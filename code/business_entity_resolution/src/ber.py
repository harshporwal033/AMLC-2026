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


LEGAL_CANON = {"private": "pvt", "pvt": "pvt", "pvtltd": "pvt", "limited": "ltd", "ltd": "ltd",
               "llp": "llp", "llc": "llc", "inc": "inc", "incorporated": "inc", "corp": "corp",
               "corporation": "corp", "co": "co", "company": "co", "lp": "lp", "pllc": "pllc",
               "pc": "pc", "plc": "plc", "sa": "sa", "sas": "sas", "sarl": "sarl", "eurl": "eurl",
               "gmbh": "gmbh"}
_SINGLES = re.compile(r"\b[a-z](?: [a-z])+\b")            # "l l p" -> "llp", "o d" -> "od"


def _collapse(s):
    return _SINGLES.sub(lambda m: m.group(0).replace(" ", ""), s)


def record_info(df, rows=None):
    """Per-record fields for the decoy-sensitive pair features (computed once per frame).
    legal: id of the set of canonical legal-form words in the raw name (0 = none)
    core:  name tokens without legal words, single letters collapsed
    nums:  address numbers with leading zeros stripped ("0049" -> "49")
    rows: only fill these rows (others stay empty) to save time on a subset."""
    n = len(df)
    rows = np.arange(n) if rows is None else np.asarray(rows)
    legal = np.zeros(n, np.int32)
    core = np.full(n, "", dtype=object)
    nums = np.full(n, "", dtype=object)
    ids = {frozenset(): 0}
    raw, nn, an = df["business_name"].values, df["name_n"].values, df["addr_n"].values
    for i in rows:
        toks = _collapse(base(raw[i])).split() if isinstance(raw[i], str) else []
        key = frozenset(LEGAL_CANON[t] for t in toks if t in LEGAL_CANON)
        legal[i] = ids.setdefault(key, len(ids))
        core[i] = " ".join(t for t in _collapse(nn[i]).split() if t not in LEGAL)
        nums[i] = " ".join(x.lstrip("0") or "0" for x in _NUMS.findall(an[i]))
    return {"legal": legal, "core": core, "nums": nums, "legal_ids": ids}


def _num_rel(a, b):
    """0 missing, 1 equal, 2 truncated (prefix/suffix), 3 close (|diff|<=10), 4 different."""
    if not a or not b:
        return 0
    if a == b:
        return 1
    if a.endswith(b) or b.endswith(a) or a.startswith(b) or b.startswith(a):
        return 2
    return 3 if abs(int(a[:15]) - int(b[:15])) <= 10 else 4


def pair_extra(qi, si, qinfo, sinfo):
    """Features aimed at the synthetic decoys: near-copies of an S1 record with a changed
    house number, a changed legal form, or one name word swapped for a different word."""
    n = len(qi)
    ql, sl = qinfo["legal"][qi], sinfo["legal"][si]
    out = {"legal_both": ((ql > 0) & (sl > 0)).astype(np.float32),
           "legal_eq": ((ql == sl) & (ql > 0)).astype(np.float32),
           "legal_diff": ((ql != sl) & (ql > 0) & (sl > 0)).astype(np.float32)}
    q_extra = np.zeros(n, np.float32); s_extra = np.zeros(n, np.float32)
    hard_sub = np.zeros(n, np.float32); soft_sub = np.zeros(n, np.float32)
    sub_min_jw = np.ones(n, np.float32)
    first_rel = np.zeros(n, np.float32); n_rel = np.zeros((n, 5), np.float32)
    qc, sc = qinfo["core"][qi], sinfo["core"][si]
    qn, sn = qinfo["nums"][qi], sinfo["nums"][si]
    jw = JaroWinkler.normalized_similarity
    for k in range(n):
        qt, st = qc[k].split(), sc[k].split()
        qs, ss = set(qt), set(st)
        qx = [t for t in qt if t not in ss]
        sx = [t for t in st if t not in qs]
        q_extra[k], s_extra[k] = len(qx), len(sx)
        for t in qx:
            best = max((jw(t, u) for u in sx), default=0.0)
            sub_min_jw[k] = min(sub_min_jw[k], best)
            if best >= 0.8:
                soft_sub[k] += 1
            else:
                hard_sub[k] += 1
        a, b = qn[k].split(), sn[k].split()
        if a and b:
            first_rel[k] = _num_rel(a[0], b[0])
            bs = set(b)
            for x in a:                                   # best relation of each query number
                n_rel[k, 1 if x in bs else min((_num_rel(x, y) for y in b), default=0)] += 1
    out.update({"q_extra": q_extra, "s_extra": s_extra, "hard_sub": hard_sub, "soft_sub": soft_sub,
                "sub_min_jw": sub_min_jw,
                "first_rel": first_rel, "num_eq": n_rel[:, 1], "num_trunc": n_rel[:, 2],
                "num_close": n_rel[:, 3], "num_far": n_rel[:, 4]})
    return out


def _cd(scorer, a, b):
    return process.cpdist(a, b, scorer=scorer, workers=-1).astype(np.float32)


_NUMS = re.compile(r"\d+")


def build_features(cand, Q, S1, qstats, qinfo=None, sinfo=None):
    """cand must hold every candidate of each query it contains (group features).
    qinfo / sinfo (record_info of Q / S1) add the decoy-sensitive features."""
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
    for v in [c[:-6] for c in cand.columns if c.endswith("_score") and c[:-6] in ("word", "emb")]:
        o = cand[["q", f"{v}_score"]].sort_values(["q", f"{v}_score"], ascending=[True, False])
        second = o[o.groupby("q").cumcount() == 1].set_index("q")[f"{v}_score"]
        best = g[f"{v}_score"].transform("max").values
        ws = cand[f"{v}_score"].values
        F[f"{v}_margin"] = np.where(ws >= best, ws - cand["q"].map(second).fillna(0).values, ws - best)
    F["q_ncand"] = g["s1"].transform("size").values
    if qinfo is not None:
        for k, v in pair_extra(qi, si, qinfo, sinfo).items():
            F[k] = v
        # ambiguity: how many candidates of this query look alike (same name / same address)
        F["n_name_twins"] = (F["n_tsort"] >= 90).groupby(cand["q"].values).transform("sum").values
        F["n_addr_twins"] = (F["a_tset"] >= 90).groupby(cand["q"].values).transform("sum").values
    return F.astype(np.float32)


# ---------------------------------------------------------------- decision + scoring
def assign(cand, p, thr):
    """Each S2/S3 record goes to its single best S1 if p >= thr (ground truth is one-to-one)."""
    d = pd.DataFrame({"q": cand["q"].values, "s1": cand["s1"].values, "p": p})
    top = d.sort_values("p", ascending=False, kind="stable").drop_duplicates("q")
    return top[top["p"] >= thr]


def macro_f05(pred_s1, correct, true_cnt, eval_mask, fp_w=None):
    """pred_s1: S1 row of each assigned query; correct: bool per assignment;
    true_cnt: #true matches per S1 row; eval_mask: S1 rows to average over.
    fp_w: optional weight per assignment for its false-positive cost (e.g. 1.9 for decoy
    records, to mimic the test set, which has ~1.9x more decoys per S1 than train)."""
    n1 = len(true_cnt)
    tp = np.bincount(pred_s1[correct], minlength=n1)
    wrong = ~np.asarray(correct)
    fp = np.bincount(pred_s1[wrong], weights=None if fp_w is None else np.asarray(fp_w)[wrong],
                     minlength=n1)
    pred = tp + fp
    fn = true_cnt - tp
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
                      "p": np.asarray(p, dtype=np.float32), "row": np.arange(len(cand))})
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


def reverse_features(top):
    """Stage-2 features: competition among the queries whose best S1 is the same S1.
    top: one row per query (q, s1, p) over ALL queries (a complete claimant set per S1)."""
    d = top[["q", "s1", "p"]].reset_index(drop=True)
    g = d.groupby("s1")["p"]
    o = d.sort_values(["s1", "p"], ascending=[True, False], kind="stable")
    second = o[o.groupby("s1").cumcount() == 1].set_index("s1")["p"]
    mx = g.transform("max").values
    sec = d["s1"].map(second).fillna(0).values
    p = d["p"].values
    other = np.where(p >= mx, sec, mx)
    return pd.DataFrame({
        "rv_rank": g.rank(ascending=False, method="first").values,
        "rv_n": g.transform("size").values,
        "rv_n05": (d["p"] >= 0.5).groupby(d["s1"]).transform("sum").values,
        "rv_sum_other": g.transform("sum").values - p,
        "rv_max_other": other,
        "rv_gap": p - other,
    }, index=top.index).astype(np.float32)


LGB_PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=127, min_data_in_leaf=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  verbose=-1, num_threads=4)


def train_oof(F, y, groups, params=LGB_PARAMS, folds=3, rounds=600, weight=None):
    """Grouped out-of-fold LightGBM. Returns oof predictions, a final model on all rows.
    weight: optional per-row sample weight (e.g. up-weight decoy rows to the test prior)."""
    import lightgbm as lgb
    from sklearn.model_selection import GroupKFold
    w = np.ones(len(F), np.float32) if weight is None else np.asarray(weight, np.float32)
    oof, iters = np.zeros(len(F), np.float32), []
    for fo, (tr, va) in enumerate(GroupKFold(folds).split(F, y, groups)):
        m = lgb.train(params, lgb.Dataset(F.iloc[tr], y[tr], weight=w[tr]), rounds,
                      valid_sets=[lgb.Dataset(F.iloc[va], y[va], weight=w[va])],
                      callbacks=[lgb.early_stopping(30, verbose=False)])
        oof[va] = m.predict(F.iloc[va], num_iteration=m.best_iteration)
        iters.append(m.best_iteration)
        print(f"  fold {fo}: best iteration {m.best_iteration}", flush=True)
    final = lgb.train(params, lgb.Dataset(F, y, weight=w), max(int(np.mean(iters) * 1.1), 10))
    return oof, final


def second_per_query(cand, p):
    """Runner-up S1 of each query: (q, s1_2, p2). Queries with one candidate get no row."""
    d = pd.DataFrame({"q": cand["q"].values, "s1": cand["s1"].values,
                      "p": np.asarray(p, dtype=np.float32)})
    d = d.sort_values(["q", "p"], ascending=[True, False], kind="stable")
    return d[d.groupby("q").cumcount() == 1].rename(columns={"s1": "s1_2", "p": "p2"})


def second_choice_features(top, second):
    """Stage-2 features about the query's runner-up S1: how strong it is, and whether that S1
    is already claimed by a stronger query (if not, the runner-up may be the true match and
    the top-1 a look-alike). top / second: best_per_query / second_per_query over ALL queries."""
    claim_max = top.groupby("s1")["p"].max()
    d = top[["q", "p"]].merge(second, on="q", how="left")
    p2 = d["p2"].fillna(0).values
    alt = d["s1_2"].map(claim_max).fillna(0).values
    return pd.DataFrame({"p2": p2, "p_gap12": d["p"].values - p2,
                         "alt_claim_max": alt, "alt_margin": p2 - alt},
                        index=top.index).astype(np.float32)


# ---------------------------------------------------------------- dev set for training
def load_dev(prep_dir, dev_cand_path, dev_mask_path, true_s1_path, frac=1.0, seed=0):
    """Loads the saved dev set with only the records it needs (low RAM).

    frac < 1 keeps a random share of the dev S1 entities (plus every query that has one of
    them as its true match or in its top-2 of any view): a faster set for trying ideas
    whose scores are comparable with the full dev set.
    Returns dict: cand (q / s1 remapped to rows of Q / S1 below), Q, S1, true_s1 (row in S1
    or NaN), dev (bool per S1 row, entities to score), true_cnt, qs (query stats)."""
    cand = pd.read_parquet(dev_cand_path)
    dev_g = np.load(dev_mask_path)
    ts_g = np.load(true_s1_path)
    if frac < 1:
        keep = np.flatnonzero(dev_g)
        keep = np.random.default_rng(seed).choice(keep, max(1, int(len(keep) * frac)), replace=False)
        dev_g = np.zeros_like(dev_g); dev_g[keep] = True
        rank_cols = [c for c in cand.columns if c.endswith("_rank")]
        near = np.zeros(len(cand), bool)
        for c in rank_cols:
            near |= cand[c].values <= 2
        tq = ts_g[cand["q"].values]
        is_dev_true = np.isfinite(tq) & dev_g[np.nan_to_num(tq, nan=0).astype(np.int64)]
        touch = np.unique(cand["q"].values[(near & dev_g[cand["s1"].values]) | is_dev_true])
        cand = cand[np.isin(cand["q"].values, touch)]
    uq = np.unique(cand["q"].values)
    s1_all = pd.read_parquet(f"{prep_dir}/train_s1.parquet")
    vocab = name_vocab(s1_all)
    us = np.unique(np.concatenate([cand["s1"].values, np.flatnonzero(dev_g)]))
    S1 = s1_all.iloc[us].reset_index(drop=True); del s1_all
    Q = pd.concat([pd.read_parquet(f"{prep_dir}/train_s{k}.parquet") for k in (2, 3)], ignore_index=True)
    Q = Q.iloc[uq].reset_index(drop=True)
    for d in (Q, S1):
        d["both_n"] = d["name_n"] + " " + d["addr_n"]
    s1_map = np.full(len(dev_g), -1, np.int64); s1_map[us] = np.arange(len(us))
    q_map = np.full(len(ts_g), -1, np.int64); q_map[uq] = np.arange(len(uq))
    cand = cand.assign(q=q_map[cand["q"].values].astype(np.int32),
                       s1=s1_map[cand["s1"].values].astype(np.int32))
    cand = cand.sort_values(["q", "s1"]).reset_index(drop=True)
    t = ts_g[uq]
    t_sub = np.where(np.isfinite(t), s1_map[np.nan_to_num(t, nan=0).astype(np.int64)], -1)
    true_s1 = np.where(t_sub >= 0, t_sub, np.nan).astype(np.float64)
    cand["is_true"] = cand["s1"].values == true_s1[cand["q"].values]
    dev = dev_g[us]
    true_cnt = np.bincount(true_s1[np.isfinite(true_s1)].astype(np.int64), minlength=len(S1))
    return {"cand": cand, "Q": Q, "S1": S1, "true_s1": true_s1, "dev": dev,
            "true_cnt": true_cnt, "qs": query_stats(Q, vocab)}


def build_features_chunked(cand, Q, S1, qs, qinfo=None, sinfo=None, chunk_q=200_000):
    """build_features over query chunks (cand sorted by q): same result, far less peak RAM."""
    qv = cand["q"].values
    uq = np.unique(qv)
    parts = []
    for j in range(0, len(uq), chunk_q):
        lo = np.searchsorted(qv, uq[j])
        hi = np.searchsorted(qv, uq[min(j + chunk_q, len(uq)) - 1], side="right")
        parts.append(build_features(cand.iloc[lo:hi].reset_index(drop=True), Q, S1, qs, qinfo, sinfo))
    return pd.concat(parts, ignore_index=True)


def run_two_stage(D, w_decoy=1.9, out_dir=None, params=LGB_PARAMS, verbose=True, rounds=600):
    """Stage 1 (pair model) + stage 2 (competition / runner-up model) on a load_dev() set.

    Scores are "test-like": decoy false matches count w_decoy times (test has ~1.9x more
    decoys per S1 than train); decoy rows are up-weighted the same way in training.
    Picks the better stage and the best decision rule on dev; saves models if out_dir."""
    import json
    import os
    cand, Q, S1, true_s1, dev, true_cnt = (D[k] for k in ("cand", "Q", "S1", "true_s1", "dev", "true_cnt"))
    t = time.time()
    F = build_features_chunked(cand, Q, S1, D["qs"], record_info(Q), record_info(S1))
    if verbose:
        print(f"features {F.shape} in {time.time() - t:.0f}s", flush=True)
    y = cand["is_true"].values.astype(np.int8)
    tq_c = true_s1[cand["q"].values]
    grp = np.where(np.isfinite(tq_c), tq_c, -(cand["q"].values + 1.0)).astype(np.int64)

    def score(a):
        tq = true_s1[a["q"].values]
        return macro_f05(a["s1"].values, a["s1"].values == tq, true_cnt, dev,
                         fp_w=np.where(np.isfinite(tq), 1.0, w_decoy))[:3]

    def apply(top, rule):
        return top[top["p"] >= rule["thr"]] if rule["mode"] == "thr" else decide_sets(top, power=rule["power"])

    def sweep(top, label):
        rules = [{"mode": "thr", "thr": float(t)} for t in np.round(np.arange(0.3, 0.96, 0.05), 2)]
        rules += [{"mode": "sets", "power": pw} for pw in (1.0, 1.2, 1.5, 2.0)]
        res = [(score(apply(top, r)), r) for r in rules]
        (f, P, R), rule = max(res, key=lambda x: x[0][0])
        if verbose:
            print(f"{label}: test-like macro F0.5 {f:.4f} | P {P:.4f} R {R:.4f} | rule {rule}", flush=True)
        return f, rule

    w1 = np.where(np.isfinite(tq_c), 1.0, w_decoy).astype(np.float32)
    oof1, m1 = train_oof(F, y, grp, params, weight=w1, rounds=rounds)
    top = best_per_query(cand, oof1)
    f1, rule1 = sweep(top, "STAGE 1")

    X2 = pd.concat([F.iloc[top["row"].values].reset_index(drop=True),
                    reverse_features(top).reset_index(drop=True),
                    second_choice_features(top, second_per_query(cand, oof1)).reset_index(drop=True)],
                   axis=1)
    X2["p1"] = top["p"].values
    del F
    y2 = cand["is_true"].values[top["row"].values].astype(np.int8)
    mask = dev[top["s1"].values]
    tq_t = true_s1[top["q"].values]
    w2 = np.where(np.isfinite(tq_t), 1.0, w_decoy).astype(np.float32)
    oof2, m2 = train_oof(X2[mask].reset_index(drop=True), y2[mask], top["s1"].values[mask], params,
                         weight=w2[mask], rounds=rounds)
    f2, rule2 = sweep(top[mask].assign(p=oof2), "STAGE 2")
    imp = pd.Series(m2.feature_importance("gain"), index=X2.columns).sort_values(ascending=False)
    if verbose:
        print("stage-2 top features:", (imp / imp.sum()).head(12).round(3).to_dict())
    use2 = f2 > f1
    cfg = {"f1": list(m1.feature_name()), "f2": list(X2.columns), "stage2": bool(use2),
           "rule": rule2 if use2 else rule1, "w_decoy": w_decoy,
           "dev_f05_stage1": float(f1), "dev_f05_stage2": float(f2)}
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        m1.save_model(f"{out_dir}/lgb1.txt"); m2.save_model(f"{out_dir}/lgb2.txt")
        json.dump(cfg, open(f"{out_dir}/config2.json", "w"))
        if verbose:
            print(f"saved {out_dir} -> stage {'2' if use2 else '1'}, rule {cfg['rule']}")
    # keep what error_report() needs
    D["_run"] = {"top": top, "p2": oof2, "mask": mask, "rule": cfg["rule"], "w_decoy": w_decoy,
                 "stage2": use2}
    return cfg


def error_report(D, n_examples=0):
    """Where does the dev score go? Needs run_two_stage(D) first.

    Prints the test-like macro F0.5 of the current system and of three oracles:
      A  perfect accept/reject of each query's current top-1 S1   (scoring ceiling)
      B  perfect choice among ALL candidates of each query         (ranking ceiling)
      C  B + blocking recall of 100% (every true pair available)   (= 1.0 by definition)
    The gaps A-now, B-A, 1-B show whether to invest in the pair scorer, the ranking,
    or candidate generation. Then error buckets by record type and country."""
    r = D["_run"]
    cand, Q, S1, true_s1, dev, true_cnt = (D[k] for k in ("cand", "Q", "S1", "true_s1", "dev", "true_cnt"))
    w = r["w_decoy"]
    top = r["top"][r["mask"]].copy()
    if r["stage2"]:
        top["p"] = r["p2"]
    rule = r["rule"]
    a = top[top["p"] >= rule["thr"]] if rule["mode"] == "thr" else decide_sets(top, power=rule["power"])

    def f(a):
        tq = true_s1[a["q"].values]
        return macro_f05(a["s1"].values, a["s1"].values == tq, true_cnt, dev,
                         fp_w=np.where(np.isfinite(tq), 1.0, w))

    f_now, P, R, fv = f(a)
    orA = top[top["s1"].values == true_s1[top["q"].values]]
    t = cand[cand["is_true"].values]
    orB = t[dev[t["s1"].values]]
    print(f"NOW        test-like F0.5 {f_now:.4f}  (P {P:.4f} R {R:.4f})")
    print(f"oracle A   perfect accept/reject of top-1      {f(orA)[0]:.4f}   <- pair-scoring ceiling")
    print(f"oracle B   perfect pick among all candidates   {f(orB)[0]:.4f}   <- ranking ceiling")
    print(f"oracle C   + perfect blocking                   1.0000")

    # per-true-pair buckets (dev S1)
    tq_all = true_s1
    dq = np.flatnonzero(np.isfinite(tq_all)); dq = dq[dev[tq_all[dq].astype(np.int64)]]
    found = np.zeros(len(Q), bool); found[t["q"].values] = True
    topi = r["top"].set_index("q")
    ts_ = topi.reindex(dq)
    right = ts_["s1"].values == tq_all[dq]
    acc = np.zeros(len(Q), bool); acc[a["q"].values] = True
    buckets = {"not in candidates": ~found[dq], "top-1 is wrong S1": found[dq] & ~right,
               "right S1, rejected": right & ~acc[dq], "matched": right & acc[dq]}
    info = pd.DataFrame({"non_latin": Q["non_latin"].values[dq].astype(bool),
                         "empty_addr": Q["addr_n"].values[dq] == "",
                         "junk_name": D["qs"][dq, 0] == 0,
                         "country": Q["country"].values[dq]})
    print(f"\ntrue pairs of dev S1: {len(dq):,}")
    rows = []
    for k, m in buckets.items():
        rows.append({"bucket": k, "n": m.sum(), "share": m.mean(),
                     "non_latin": info["non_latin"][m].mean(), "empty_addr": info["empty_addr"][m].mean(),
                     "junk_name": info["junk_name"][m].mean()})
    rows.append({"bucket": "(all true pairs)", "n": len(dq), "share": 1.0, "non_latin": info["non_latin"].mean(),
                 "empty_addr": info["empty_addr"].mean(), "junk_name": info["junk_name"].mean()})
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    # false matches
    tq = true_s1[a["q"].values]
    wrong = a["s1"].values != tq
    dec = ~np.isfinite(tq)
    print(f"\naccepted {len(a):,} | wrong {wrong.sum():,} (decoys {(wrong & dec).sum():,}, other-S1 "
          f"{(wrong & ~dec).sum():,}) | on singleton S1: {(wrong & (true_cnt[a['s1'].values] == 0)).sum():,}")
    # decoy clusters: do decoys claiming the same S1 come in groups?
    top_dec = top[~np.isfinite(true_s1[top["q"].values])]
    per_s1 = top_dec.groupby("s1").size()
    print(f"decoys claiming a dev S1: {len(top_dec):,} over {len(per_s1):,} S1 | "
          f"S1 with >=2 decoy claimants: {(per_s1 >= 2).mean():.3f} | mean decoys per such S1: {per_s1.mean():.2f}")

    sc = S1["country"].values[dev]
    print("\nF0.5 by country:", pd.Series(fv[dev]).groupby(sc).mean().round(4).to_dict())
    tc = true_cnt[dev]
    print("F0.5 by #true matches of S1:", pd.Series(fv[dev]).groupby(np.minimum(tc, 6)).mean().round(4).to_dict())
    print("share of dev S1 by #true matches:", pd.Series(np.minimum(tc, 6)).value_counts(normalize=True).sort_index().round(3).to_dict())

    if n_examples:
        rng = np.random.default_rng(0)
        cs = cand.set_index(["q", "s1"])
        def show(qs, title):
            print(f"\n--- {title} ---")
            for q in rng.choice(qs, min(n_examples, len(qs)), replace=False):
                s = int(topi.loc[q, "s1"]); tt = true_s1[q]
                print(f"Q  {str(Q['business_name'].values[q])[:40]!r:42} {str(Q['business_address'].values[q])[:60]}")
                print(f"S1 {str(S1['business_name'].values[s])[:40]!r:42} {str(S1['business_address'].values[s])[:60]}  (top-1)")
                if np.isfinite(tt) and int(tt) != s:
                    print(f"T  {str(S1['business_name'].values[int(tt)])[:40]!r:42} {str(S1['business_address'].values[int(tt)])[:60]}  (TRUE)")
        show(dq[buckets["right S1, rejected"]], "right S1 but rejected")
        show(a["q"].values[wrong & dec], "accepted decoy")
