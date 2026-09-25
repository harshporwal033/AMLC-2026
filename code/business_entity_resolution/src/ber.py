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
import time
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
def make_views(word_max_df=0.02, char=False, char_max_df=0.05):
    """Sparse TF-IDF views. max_df drops words / n-grams present in more than that share of
    a country's S1 records: they barely identify a business but make the top-K search slow.
    The char-3gram views are optional (slow on the full data; the dense view replaces them)."""
    views = {"word": ("both_n", dict(analyzer="word", token_pattern=r"\S+", min_df=1,
                                     max_df=word_max_df, sublinear_tf=True, dtype=np.float32))}
    if char:
        char3 = dict(analyzer="char_wb", ngram_range=(3, 3), min_df=2, max_df=char_max_df,
                     sublinear_tf=True, dtype=np.float32)
        views["name"] = ("name_n", char3)
        views["addr"] = ("addr_n", char3)
    return views


VIEWS = make_views()
VIEW_K = {"word": 10, "emb": 10, "name": 5, "addr": 5}   # final candidate set (per view)
CHAR_FRAC = 0.3   # share of queries (weakest word-view margin) that also get the char views
EMB_MODEL = "intfloat/multilingual-e5-small"             # MIT licence, 118M parameters


def emb_texts(df):
    """Raw (not transliterated) text: the multilingual model reads Indic scripts directly."""
    return ("query: " + df["business_name"].fillna("").astype(str) + " | "
            + df["business_address"].fillna("").astype(str)).tolist()


class Embedder:
    """Mean-pooled, L2-normalised sentence embeddings (fp16 on GPU). Rows stay on device."""

    def __init__(self, model=EMB_MODEL, device=None, batch=512, max_len=48):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.dev.startswith("cuda") else torch.float32
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model).to(self.dev).eval()
        if self.dtype == torch.float16:
            self.model.half()
        self.dim, self.batch, self.max_len = self.model.config.hidden_size, batch, max_len

    def encode(self, texts):
        torch = self.torch
        order = np.argsort([len(t) for t in texts], kind="stable")   # similar lengths per batch
        out = torch.empty((len(texts), self.dim), dtype=self.dtype, device=self.dev)
        with torch.inference_mode():
            for i in range(0, len(texts), self.batch):
                idx = order[i:i + self.batch]
                b = self.tok([texts[j] for j in idx], padding=True, truncation=True,
                             max_length=self.max_len, return_tensors="pt").to(self.dev)
                h = self.model(**b).last_hidden_state
                m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
                e = torch.nn.functional.normalize(((h * m).sum(1) / m.sum(1)).float(), dim=-1)
                out[torch.as_tensor(idx, device=self.dev)] = e.to(self.dtype)
        return out


def _fit(params, docs):
    vec = TfidfVectorizer(**params)
    try:
        return vec, vec.fit_transform(docs).T.tocsr()
    except ValueError:                      # tiny index: max_df pruned everything
        vec = TfidfVectorizer(**{**params, "max_df": 1.0, "min_df": 1})
        return vec, vec.fit_transform(docs).T.tocsr()


class Blocker:
    """Candidate generation per country: S2/S3 record -> top-k S1 records per view.

    word  sparse TF-IDF over rare words of name + address (CPU, every query)
    emb   dense multilingual embeddings, exact top-k by cosine (GPU, every query); also
          gives an exact emb_score for every candidate found by the other views
    name/addr  optional char-3gram views, only for "weak" queries (see make_views)."""

    def __init__(self, S1, views=VIEWS, n_threads=4, char_frac=CHAR_FRAC, embedder=None,
                 verbose=False):
        self.views, self.nt, self.char_frac, self.embedder = views, n_threads, char_frac, embedder
        self.idx, self.emb = {}, {}
        self.s1_pos = np.full(len(S1), -1, np.int64)            # S1 row -> row within country
        for c in pd.unique(S1["country"]):
            t0 = time.time()
            rows = np.flatnonzero(S1["country"].values == c)
            self.s1_pos[rows] = np.arange(len(rows))
            self.idx[c] = (rows, {v: _fit(params, S1[col].values[rows])
                                  for v, (col, params) in views.items()})
            if embedder is not None:
                self.emb[c] = embedder.encode(emb_texts(S1.iloc[rows]))
            if verbose:
                print(f"  index [{c}] {len(rows):,} S1 records, {time.time() - t0:.0f}s", flush=True)
        self.view_names = list(views) + (["emb"] if embedder is not None else [])

    def _weak(self, word_cand, Q, qc):
        d = word_cand.sort_values(["q", "word_score"], ascending=[True, False], kind="stable")
        top = d.groupby("q")["word_score"].agg(["first", "size"])
        second = d[d.groupby("q").cumcount() == 1].set_index("q")["word_score"]
        margin = pd.Series(-1.0, index=qc)                      # no word hit -> weak
        margin.loc[top.index] = top["first"] - second.reindex(top.index).fillna(0)
        cut = np.quantile(margin.values, self.char_frac) if len(qc) else 0
        weak = (margin.values <= cut)
        if "non_latin" in Q.columns:
            weak |= Q["non_latin"].values[qc].astype(bool)
        return qc[weak]

    def _dense(self, Q, qc, c, merged, k, chunk=1024):
        torch = self.embedder.torch
        rows, S = self.idx[c][0], self.emb[c]
        E = self.embedder.encode(emb_texts(Q.iloc[qc]))
        kk = min(k, S.shape[0])
        sc, ix = [], []
        for j in range(0, len(qc), chunk):
            s, i = (E[j:j + chunk] @ S.T).topk(kk, dim=1)
            sc.append(s.float().cpu().numpy()); ix.append(i.cpu().numpy())
        sc, ix = np.concatenate(sc), np.concatenate(ix)
        d = pd.DataFrame({"q": np.repeat(qc, kk).astype(np.int32),
                          "s1": rows[ix.ravel()].astype(np.int32),
                          "emb_score": sc.ravel().astype(np.float32),
                          "emb_rank": np.tile(np.arange(1, kk + 1, dtype=np.int16), len(qc))})
        merged = d if merged is None else merged.merge(d, on=["q", "s1"], how="outer")
        # exact cosine for candidates that only the other views found
        miss = np.flatnonzero(merged["emb_score"].isna().values)
        if len(miss):
            qpos = np.empty(len(Q), np.int64); qpos[qc] = np.arange(len(qc))
            qi = qpos[merged["q"].values[miss]]
            si = self.s1_pos[merged["s1"].values[miss]]
            vals = np.empty(len(miss), np.float32)
            for j in range(0, len(miss), 1_000_000):
                a = torch.as_tensor(qi[j:j + 1_000_000], device=E.device)
                b = torch.as_tensor(si[j:j + 1_000_000], device=E.device)
                vals[j:j + 1_000_000] = (E[a] * S[b]).sum(1).float().cpu().numpy()
            col = merged["emb_score"].values.astype(np.float32)
            col[miss] = vals
            merged["emb_score"] = col
        del E
        return merged

    def query(self, Q, k=20, q_chunk=200_000, view_k=None, verbose=False):
        """Returns DataFrame(q=row in Q, s1=row in S1, {view}_score, {view}_rank).
        view_k: optional {view: K} keeping only pairs within K of at least one view."""
        out = []
        for c in pd.unique(Q["country"]):
            if c not in self.idx:
                continue
            rows, fitted = self.idx[c]
            qc = np.flatnonzero(Q["country"].values == c)
            merged, sub = None, qc
            for v, (col, _) in self.views.items():
                t0 = time.time()
                if merged is not None and v != "word" and self.char_frac < 1:
                    if sub is qc:
                        sub = self._weak(merged, Q, qc)
                vec, X1T = fitted[v]
                parts = []
                for i in range(0, len(sub), q_chunk):
                    qq = sub[i:i + q_chunk]
                    C = sp_matmul_topn(vec.transform(Q[col].values[qq]), X1T, top_n=k,
                                       threshold=0.01, sort=True, n_threads=self.nt).tocoo()
                    parts.append(pd.DataFrame({"q": qq[C.row].astype(np.int32),
                                               "s1": rows[C.col].astype(np.int32),
                                               f"{v}_score": C.data.astype(np.float32)}))
                d = pd.concat(parts, ignore_index=True)
                d = d.sort_values(["q", f"{v}_score"], ascending=[True, False], kind="stable")
                d[f"{v}_rank"] = (d.groupby("q").cumcount() + 1).astype(np.int16)
                merged = d if merged is None else merged.merge(d, on=["q", "s1"], how="outer")
                if verbose:
                    print(f"  [{c}] view {v}: {len(sub):,} queries, {time.time() - t0:.0f}s", flush=True)
            if self.embedder is not None:
                t0 = time.time()
                merged = self._dense(Q, qc, c, merged, k)
                if verbose:
                    print(f"  [{c}] view emb: {len(qc):,} queries, {time.time() - t0:.0f}s", flush=True)
            out.append(merged)
        cols = ["q", "s1"] + [f"{v}_{s}" for v in self.view_names for s in ("score", "rank")]
        if not out:
            return pd.DataFrame(columns=cols)
        cand = pd.concat(out, ignore_index=True)
        for v in self.view_names:
            cand[f"{v}_score"] = cand[f"{v}_score"].fillna(0).astype(np.float32)
            cand[f"{v}_rank"] = cand[f"{v}_rank"].fillna(999).astype(np.int16)
        if view_k:
            keep = np.zeros(len(cand), bool)
            for v, kk in view_k.items():
                if f"{v}_rank" in cand:
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
    for v in [c[:-6] for c in cand.columns if c.endswith("_score")]:
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
    files, hit, tot = [], 0, 0
    for n, i in enumerate(range(0, len(Q), chunk)):
        t = time.time()
        c = blk.query(Q.iloc[i:i + chunk], k=k, view_k=view_k, verbose=True)
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


def best_per_query(cand, p):
    """One row per query: its highest-probability S1 (q, s1, p)."""
    d = pd.DataFrame({"q": cand["q"].values, "s1": cand["s1"].values,
                      "p": np.asarray(p, dtype=np.float32)})
    return d.sort_values("p", ascending=False, kind="stable").drop_duplicates("q")


def decide_sets(top, p_min=0.05, power=1.0):
    """Set-level decision tuned for macro F0.5 (replaces a single global threshold).

    top: best_per_query output. For each S1, its queries are sorted by p and we keep the
    top-m (m = 0..n) maximising the plug-in expected F0.5 of that S1:
        m > 0: 1.25*TP / (1.25*TP + 0.25*(T-TP) + (m-TP)), TP = sum of kept p, T = sum of all p
        m = 0: P(no true match) = prod(1 - p)
    So a lone p=0.5 match on an otherwise empty S1 is dropped (a false merge on a singleton
    costs a full 1.0), while the same p next to confident matches may be kept.
    power < 1 inflates / > 1 deflates probabilities (calibration knob, tune on dev)."""
    d = top[top["p"] >= p_min][["q", "s1", "p"]].copy()
    if d.empty:
        return d
    d["p"] = np.clip(d["p"].values.astype(np.float64), 1e-6, 1 - 1e-6) ** power
    d = d.sort_values(["s1", "p"], ascending=[True, False], kind="stable").reset_index(drop=True)
    g = d.groupby("s1")["p"]
    tp = g.cumsum().values
    m = (g.cumcount() + 1).values
    T = g.transform("sum").values
    d["f_m"] = 1.25 * tp / (1.25 * tp + 0.25 * (T - tp) + (m - tp))
    d["m"] = m
    f0 = np.exp(np.log1p(-d["p"]).groupby(d["s1"]).sum())
    best = d.loc[d.groupby("s1")["f_m"].idxmax(), ["s1", "m", "f_m"]].set_index("s1")
    keep_m = pd.Series(np.where(best["f_m"].values > f0.reindex(best.index).values,
                                best["m"].values, 0), index=best.index)
    d = d[d["m"].values <= d["s1"].map(keep_m).values]
    return d[["q", "s1", "p"]].reset_index(drop=True)
