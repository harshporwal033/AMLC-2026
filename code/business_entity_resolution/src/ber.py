"""Business Entity Resolution: shared pipeline code.

Stages
  1. prep_file      normalise a raw source TSV into parquet (name_n, addr_n, non_latin)
  2. Blocker        per-country TF-IDF top-K search: S2/S3 record -> S1 candidates
  3. build_features pair features for (query, S1 candidate)
  4. assign         one S1 per S2/S3 record (best probability >= threshold)
  5. macro_f05 / write_outputs

Country is only used to partition the search (every true pair shares a country);
it is never a model feature, so unseen countries (France) work unchanged.
"""
import re
from collections import Counter
from multiprocessing import Pool

import numpy as np
import pandas as pd
from anyascii import anyascii
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

# ---------------------------------------------------------------- normalisation
LEGAL = {"private", "pvt", "pvtltd", "limited", "ltd", "llc", "inc", "incorporated", "corp",
         "corporation", "co", "company", "llp", "plc", "pllc", "pc", "lp", "sa", "sas", "sarl",
         "eurl", "gmbh", "the"}
HONOR = {"mr", "mrs", "ms", "m", "s", "dr", "messrs", "shri", "sri", "smt"}
DBA = re.compile(r"\b(trading as|doing business as|d b a|dba|t a|aka)\b")
ID_JUNK = re.compile(r"\(\s*id\b[^)]*\)|#\s*\d+", re.I)
ADDR_MAP = {"street": "st", "saint": "st", "road": "rd", "avenue": "ave", "court": "ct",
            "lane": "ln", "drive": "dr", "boulevard": "blvd", "place": "pl", "nagar": "ngr",
            "sector": "sec", "near": "nr", "opposite": "opp", "floor": "flr",
            "apartment": "apt", "house": "h", "hno": "h"}
ADDR_DROP = {"null", "nan", "no", "number", "/"}
NUM_SUFFIX = re.compile(r"^(\d+)[a-z]{1,2}$")


def base(s):
    s = anyascii(s).lower()
    s = re.sub(r"\.(com|in|net|org|co|fr)\b", " ", s)
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9/ ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def norm_name(s):
    if not isinstance(s, str) or not s:
        return ""
    s = base(ID_JUNK.sub(" ", s))
    parts = DBA.split(s)
    if len(parts) > 1:
        s = parts[-1].strip()
    toks = [t for t in s.split() if t not in LEGAL]
    while len(toks) > 1 and toks[0] in HONOR:
        toks = toks[1:]
    return " ".join(toks) or s


def norm_addr(s):
    if not isinstance(s, str) or not s:
        return ""
    toks = base(s).replace("/", " / ").split()
    toks = [ADDR_MAP.get(t, t) for t in toks if t not in ADDR_DROP]
    return " ".join(NUM_SUFFIX.sub(r"\1", t) for t in toks)


def _norm_chunk(args):
    names, addrs = args
    return [norm_name(x) for x in names], [norm_addr(x) for x in addrs]


def prep_file(path_in, path_out, n_jobs=4, chunk=200_000):
    """Raw TSV -> parquet with normalised columns."""
    df = pd.read_csv(path_in, sep="\t", dtype=str, keep_default_na=False)
    names, addrs = df["business_name"].tolist(), df["business_address"].tolist()
    jobs = [(names[i:i + chunk], addrs[i:i + chunk]) for i in range(0, len(df), chunk)]
    with Pool(n_jobs) as p:
        res = p.map(_norm_chunk, jobs)
    df["name_n"] = [x for r in res for x in r[0]]
    df["addr_n"] = [x for r in res for x in r[1]]
    df["non_latin"] = [any(ord(c) > 0x24F for c in s) for s in names]
    df.to_parquet(path_out, index=False)
    return len(df)


def load(path):
    df = pd.read_parquet(path)
    df["both_n"] = df["name_n"] + " " + df["addr_n"]
    return df


# ---------------------------------------------------------------- blocking
CHAR3 = dict(analyzer="char_wb", ngram_range=(3, 3), min_df=2, sublinear_tf=True, dtype=np.float32)
VIEWS = {
    "name": ("name_n", CHAR3),
    "addr": ("addr_n", CHAR3),
    "word": ("both_n", dict(analyzer="word", token_pattern=r"\S+", min_df=1,
                            sublinear_tf=True, dtype=np.float32)),
}
VIEW_K = {"word": 10, "name": 5, "addr": 5}   # final candidate set fed to the model


class Blocker:
    """Fits one TF-IDF index per (country, view) over S1; queries return top-k per view."""

    def __init__(self, S1, views=VIEWS, n_threads=4):
        self.views, self.nt, self.idx = views, n_threads, {}
        for c in pd.unique(S1["country"]):
            rows = np.flatnonzero(S1["country"].values == c)
            fitted = {}
            for v, (col, params) in views.items():
                vec = TfidfVectorizer(**params)
                fitted[v] = (vec, vec.fit_transform(S1[col].values[rows]).T.tocsr())
            self.idx[c] = (rows, fitted)

    def query(self, Q, k=20, q_chunk=200_000, view_k=None):
        """Returns DataFrame(q=row in Q, s1=row in S1, {view}_score, {view}_rank).
        view_k: optional {view: K} keeping only pairs within K of at least one view."""
        out = []
        for c in pd.unique(Q["country"]):
            if c not in self.idx:
                continue
            rows, fitted = self.idx[c]
            qc = np.flatnonzero(Q["country"].values == c)
            merged = None
            for v, (col, _) in self.views.items():
                vec, X1T = fitted[v]
                parts = []
                for i in range(0, len(qc), q_chunk):
                    qq = qc[i:i + q_chunk]
                    C = sp_matmul_topn(vec.transform(Q[col].values[qq]), X1T, top_n=k,
                                       threshold=0.01, sort=True, n_threads=self.nt).tocoo()
                    parts.append(pd.DataFrame({"q": qq[C.row].astype(np.int32),
                                               "s1": rows[C.col].astype(np.int32),
                                               f"{v}_score": C.data.astype(np.float32)}))
                d = pd.concat(parts, ignore_index=True)
                d = d.sort_values(["q", f"{v}_score"], ascending=[True, False], kind="stable")
                d[f"{v}_rank"] = (d.groupby("q").cumcount() + 1).astype(np.int16)
                merged = d if merged is None else merged.merge(d, on=["q", "s1"], how="outer")
            out.append(merged)
        cols = ["q", "s1"] + [f"{v}_{s}" for v in self.views for s in ("score", "rank")]
        if not out:
            return pd.DataFrame(columns=cols)
        cand = pd.concat(out, ignore_index=True)
        for v in self.views:
            cand[f"{v}_score"] = cand[f"{v}_score"].fillna(0).astype(np.float32)
            cand[f"{v}_rank"] = cand[f"{v}_rank"].fillna(999).astype(np.int16)
        if view_k:
            keep = np.zeros(len(cand), bool)
            for v, kk in view_k.items():
                keep |= cand[f"{v}_rank"].values <= kk
            cand = cand[keep]
        return cand[cols].sort_values(["q", "s1"]).reset_index(drop=True)


# ---------------------------------------------------------------- features
def name_vocab(S1):
    """Document frequency of name tokens over the S1 index (detects junk names)."""
    return Counter(t for s in S1["name_n"].values for t in set(s.split()))


def query_stats(Q, vocab):
    """Per-query-record stats, computed once per Q frame."""
    toks = Q["name_n"].str.split()
    return pd.DataFrame({
        "q_known_frac": toks.map(lambda ts: np.mean([vocab[t] > 0 for t in ts]) if ts else 0.0),
        "q_min_tokfreq": np.log1p(toks.map(lambda ts: min((vocab[t] for t in ts), default=0))),
    }).astype(np.float32).values


def _cd(scorer, a, b):
    return process.cpdist(a, b, scorer=scorer, workers=-1).astype(np.float32)


_NUMS = re.compile(r"\d+")


def build_features(cand, Q, S1, qstats):
    """cand must hold every candidate of each query it contains (group features)."""
    qi, si = cand["q"].values, cand["s1"].values
    n = len(cand)
    qn, sn = Q["name_n"].values[qi], S1["name_n"].values[si]
    qa, sa = Q["addr_n"].values[qi], S1["addr_n"].values[si]

    F = cand[[c for c in cand.columns if c.endswith(("_score", "_rank"))]].copy()
    F["n_tset"] = _cd(fuzz.token_set_ratio, qn, sn)
    F["n_tsort"] = _cd(fuzz.token_sort_ratio, qn, sn)
    F["n_partial"] = _cd(fuzz.partial_ratio, qn, sn)
    F["n_jw"] = _cd(JaroWinkler.normalized_similarity, qn, sn)
    F["a_tset"] = _cd(fuzz.token_set_ratio, qa, sa)
    F["a_tsort"] = _cd(fuzz.token_sort_ratio, qa, sa)
    F["a_partial"] = _cd(fuzz.partial_ratio, qa, sa)

    qN = [frozenset(_NUMS.findall(s)) for s in qa]
    sN = [frozenset(_NUMS.findall(s)) for s in sa]
    inter = np.fromiter((len(a & b) for a, b in zip(qN, sN)), np.float32, n)
    union = np.fromiter((len(a | b) for a, b in zip(qN, sN)), np.float32, n)
    F["num_inter"], F["num_union"], F["num_jacc"] = inter, union, inter / np.maximum(union, 1)
    qF = [m.group() if (m := _NUMS.search(s)) else "" for s in qa]
    sF = [m.group() if (m := _NUMS.search(s)) else "" for s in sa]
    F["first_eq"] = np.fromiter((a != "" and a == b for a, b in zip(qF, sF)), bool, n)
    F["first_pref"] = np.fromiter((bool(a and b and (a.startswith(b) or b.startswith(a)))
                                   for a, b in zip(qF, sF)), bool, n)

    qT = [set(s.split()) for s in qn]
    sT = [set(s.split()) for s in sn]
    ti = np.fromiter((len(a & b) for a, b in zip(qT, sT)), np.float32, n)
    tu = np.fromiter((len(a | b) for a, b in zip(qT, sT)), np.float32, n)
    F["ntok_inter"], F["ntok_jacc"] = ti, ti / np.maximum(tu, 1)
    F["q_known_frac"], F["q_min_tokfreq"] = qstats[qi, 0], qstats[qi, 1]

    F["q_nlen"] = np.fromiter(map(len, qn), np.float32, n)
    F["s_nlen"] = np.fromiter(map(len, sn), np.float32, n)
    F["q_alen"] = np.fromiter(map(len, qa), np.float32, n)
    F["s_alen"] = np.fromiter(map(len, sa), np.float32, n)
    F["q_nonlatin"] = Q["non_latin"].values[qi]
    F["q_is_s3"] = Q["entity_id"].str.startswith("S3").values[qi]

    g = cand.groupby("q")
    for v in ("word", "name", "addr"):
        F[f"{v}_gap_best"] = cand[f"{v}_score"].values - g[f"{v}_score"].transform("max").values
    o = cand[["q", "word_score"]].sort_values(["q", "word_score"], ascending=[True, False])
    second = o[o.groupby("q").cumcount() == 1].set_index("q")["word_score"]
    best = g["word_score"].transform("max").values
    ws = cand["word_score"].values
    F["word_margin"] = np.where(ws >= best, ws - cand["q"].map(second).fillna(0).values, ws - best)
    F["q_ncand"] = g["s1"].transform("size").values
    return F.astype(np.float32)


# ---------------------------------------------------------------- decision + scoring
def assign(cand, p, thr):
    """Each S2/S3 record goes to its single best S1 if p >= thr (ground truth is one-to-one)."""
    d = pd.DataFrame({"q": cand["q"].values, "s1": cand["s1"].values, "p": p})
    top = d.sort_values("p", ascending=False, kind="stable").drop_duplicates("q")
    return top[top["p"] >= thr]


def macro_f05(pred_s1, correct, true_cnt, eval_mask):
    """pred_s1: S1 row of each assigned query; correct: bool per assignment;
    true_cnt: #true matches per S1 row; eval_mask: S1 rows to average over."""
    n1 = len(true_cnt)
    pred = np.bincount(pred_s1, minlength=n1)
    tp = np.bincount(pred_s1[correct], minlength=n1)
    fp, fn = pred - tp, true_cnt - tp
    den = 1.25 * tp + 0.25 * fn + fp
    f = np.where(den == 0, 1.0, 1.25 * tp / np.maximum(den, 1e-9))
    m = eval_mask
    P, R = tp[m].sum() / max(pred[m].sum(), 1), tp[m].sum() / max(true_cnt[m].sum(), 1)
    return f[m].mean(), P, R, f


def truth_map(gt):
    """{S2/S3 entity_id: S1 entity_id} from the ground-truth file."""
    t = gt.assign(ids=gt["matched_entity_ids"].fillna("").str.split(",")).explode("ids")
    t = t[t["ids"] != ""]
    return dict(zip(t["ids"], t["source1_entity_id"]))


def write_id_lists(s1_ids, pair_s1_ids, pair_q_ids, col, path):
    """One row per S1 entity, comma-joined unique S2/S3 ids (empty when none)."""
    lists = (pd.DataFrame({"s": pair_s1_ids, "q": pair_q_ids}).drop_duplicates()
             .groupby("s")["q"].agg(",".join))
    out = pd.DataFrame({"source1_entity_id": s1_ids})
    out[col] = out["source1_entity_id"].map(lists).fillna("")
    out.to_csv(path, sep="\t", index=False)


def block_all(blk, Q, out_prefix, chunk=1_000_000, k=20, view_k=VIEW_K, true_s1=None):
    """Blocks every row of Q in chunks, writing <out_prefix>_NNN.parquet (q = row in Q).
    If true_s1 (S1 row per Q row, NaN for decoys) is given, prints running recall."""
    import time
    files, hit, tot = [], 0, 0
    for n, i in enumerate(range(0, len(Q), chunk)):
        t = time.time()
        c = blk.query(Q.iloc[i:i + chunk], k=k, view_k=view_k)
        c["q"] = (c["q"] + i).astype(np.int32)
        f = f"{out_prefix}_{n:03d}.parquet"
        c.to_parquet(f, index=False)
        files.append(f)
        msg = f"chunk {n}: {min(i + chunk, len(Q)):,}/{len(Q):,} rows, {len(c):,} pairs, {time.time() - t:.0f}s"
        if true_s1 is not None:
            ts = true_s1[i:i + chunk]
            hit += int((c["s1"].values == true_s1[c["q"].values]).sum())
            tot += int(np.isfinite(ts).sum())
            msg += f", recall so far {hit / max(tot, 1):.4f}"
        print(msg, flush=True)
    return files


def write_pairs(s1_ids, q_ids, s1_rows, q_rows, col, path):
    """Fast writer for large pair sets given integer rows into S1 / Q.
    One row per S1 entity, comma-joined unique S2/S3 ids (empty when none)."""
    key = np.unique(s1_rows.astype(np.int64) * len(q_ids) + q_rows.astype(np.int64))
    s, q = key // len(q_ids), key % len(q_ids)
    uniq, start = np.unique(s, return_index=True)
    end = np.append(start[1:], len(s))
    lists = np.full(len(s1_ids), "", dtype=object)
    for u, a, b in zip(uniq, start, end):
        lists[u] = ",".join(q_ids[q[a:b]])
    pd.DataFrame({"source1_entity_id": s1_ids, col: lists}).to_csv(path, sep="\t", index=False)
