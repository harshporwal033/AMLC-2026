"""Test prediction, end to end from saved artifacts (resumable).

  stage 1 (pair model) over every test candidate  -> <WORK>/test_stage1/  (resumes if interrupted)
  stage 2 (+ coherence, + cross-encoder)          -> <WORK>/output/matching_results.tsv
                                                     <WORK>/output/candidate_pairs.tsv

Inputs are found by searching ROOTS for:
  test_s1.parquet / test_s2.parquet / test_s3.parquet   (normalised test records, ber.prep_file)
  test_000.parquet ...                                   (test candidate blocks, ber.block_all)
  config2.json + lgb1.txt                                (stage-1 model, "model3")
  config4.json + lgb2.txt (+ ce_* folders)               (stage-2 model, "model4" / "model5")

Kaggle: put this file and ber.py in /kaggle/working (or %%writefile them), then
    %run /kaggle/working/predict_test.py
Use "Save Version -> Save & Run All" with a GPU to run it in the background.
"""
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

WORK = os.environ.get("BER_WORK", "/kaggle/working")
ROOTS = os.environ.get("BER_ROOTS", "/kaggle/input:/kaggle/working").split(":")
sys.path.insert(0, WORK)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ber  # noqa: E402
import lightgbm as lgb  # noqa: E402


def find(name, required=True):
    for root in ROOTS:
        hits = sorted(glob.glob(f"{root}/**/{name}", recursive=True))
        hits = [h for h in hits if "/test_stage1/" not in h]
        if hits:
            return hits[0]
    if required:
        raise FileNotFoundError(f"{name} not found under {ROOTS}")
    return None


def main():
    t0 = time.time()
    prep = os.path.dirname(find("test_s1.parquet"))
    blk = os.path.dirname(find("test_000.parquet"))
    # stage-2 model: BER_STAGE2 (substring of its folder path) picks one when several are
    # attached; otherwise the last in sorted order (model5 after model4). Stage-1 model: the
    # config2.json in the same folder if present, else any attached one.
    c4 = sorted(h for r in ROOTS for h in glob.glob(f"{r}/**/config4.json", recursive=True))
    want = os.environ.get("BER_STAGE2", "")
    c4 = [h for h in c4 if want in h] or c4
    if not c4:
        raise FileNotFoundError("config4.json not found")
    m4 = os.path.dirname(c4[-1])
    m3 = m4 if os.path.exists(f"{m4}/config2.json") else os.path.dirname(find("config2.json"))
    files = sorted(glob.glob(f"{blk}/test_*.parquet"))
    print(f"prep {prep}\nblocks {blk} ({len(files)} files)\nstage-1 model {m3}\nstage-2 model {m4}", flush=True)

    S1 = ber.load(f"{prep}/test_s1.parquet")
    Q = pd.concat([ber.load(f"{prep}/test_s2.parquet"), ber.load(f"{prep}/test_s3.parquet")],
                  ignore_index=True)
    print(f"test S1 {len(S1):,} | S2+S3 {len(Q):,}", flush=True)

    # ---- stage 1 (resumable) ----
    cfg3 = json.load(open(f"{m3}/config2.json"))
    m1 = lgb.Booster(model_file=f"{m3}/lgb1.txt")
    s1o = f"{WORK}/test_stage1"
    need = ("top1.parquet", "second.parquet", "X1top.parquet", "cand_s1.npy", "cand_q.npy")
    done = [d for d in [s1o] + [os.path.dirname(h) for r in ROOTS
                                for h in glob.glob(f"{r}/**/test_stage1/top1.parquet", recursive=True)]
            if all(os.path.exists(f"{d}/{f}") for f in need)]
    if done:
        s1o = done[0]
        print(f"stage 1 already complete, loading from {s1o}", flush=True)
        top = pd.read_parquet(f"{s1o}/top1.parquet")
        second = pd.read_parquet(f"{s1o}/second.parquet")
        X = pd.read_parquet(f"{s1o}/X1top.parquet")
    else:
        top, second, X = ber.predict_test_stage1(files, Q, S1, m1, cfg3["f1"], s1o)

    # ---- stage 2 ----
    cfg4 = json.load(open(f"{m4}/config4.json"))
    m2 = lgb.Booster(model_file=f"{m4}/lgb2.txt")
    print(f"stage-2 variant {cfg4.get('variant')} | dev F0.5 {cfg4.get('dev_f05', float('nan')):.4f}", flush=True)
    extra = [ber.coherence_features(top, Q, S1)] if cfg4.get("coherence") else []
    if cfg4.get("ce"):
        dirs = [f"{m4}/{d}" for d in cfg4["ce"]]
        sel = np.ones(len(top), bool)
        if cfg4.get("ce_band"):
            lo, hi = cfg4["ce_band"]
            sel = (top["p"].values > lo) & (top["p"].values < hi)
        t = time.time()
        print(f"cross-encoder scoring {sel.sum():,} queries (top-1 + runner-up) with {len(dirs)} model(s)", flush=True)
        t_sel = top[sel]
        s_sel = second[second["q"].isin(t_sel["q"])]
        c1 = ber.score_ce(dirs, ber.ce_texts(Q, t_sel["q"].values), ber.ce_texts(S1, t_sel["s1"].values))
        s2 = ber.score_ce(dirs, ber.ce_texts(Q, s_sel["q"].values), ber.ce_texts(S1, s_sel["s1_2"].values))
        part = ber.ce_features(c1, t_sel["q"].map(pd.Series(s2, index=s_sel["q"].values)).values)
        ce = pd.DataFrame(np.nan, index=np.arange(len(top)), columns=part.columns, dtype=np.float32)
        ce.loc[np.flatnonzero(sel), :] = part.values
        extra.append(ce)
        print(f"  done in {time.time() - t:.0f}s", flush=True)
    X2 = ber.stage2_matrix(top, second, X, extra)
    top["p1"] = top["p"]
    top["p"] = m2.predict(X2[cfg4["f2"]])
    rule = cfg4["rule"]
    a = top[top["p"] >= rule["thr"]] if rule["mode"] == "thr" else ber.decide_sets(top, power=rule["power"])

    out = f"{WORK}/output"
    os.makedirs(out, exist_ok=True)
    top.to_parquet(f"{out}/test_top_final.parquet")
    s1_ids, q_ids = S1["entity_id"].values, Q["entity_id"].values
    ber.write_pairs(s1_ids, q_ids, a["s1"].values, a["q"].values, "matched_entity_ids",
                    f"{out}/matching_results.tsv")
    ber.write_pairs(s1_ids, q_ids, np.load(f"{s1o}/cand_s1.npy"), np.load(f"{s1o}/cand_q.npy"),
                    "candidate_entity_ids", f"{out}/candidate_pairs.tsv")

    m = pd.read_csv(f"{out}/matching_results.tsv", sep="\t", dtype=str, keep_default_na=False)
    ids = m["matched_entity_ids"].str.split(",").explode()
    ids = ids[ids != ""]
    print("CHECKS | rows == test S1:", len(m) == len(S1), "| unique S1:", m["source1_entity_id"].is_unique,
          "| ids valid:", ids.isin(set(q_ids)).all(), "| no id twice:", ids.is_unique)
    n_matched = m["matched_entity_ids"].str.split(",").map(lambda x: len([i for i in x if i])).values
    cty = S1["country"].values
    for c in pd.unique(cty):
        print(f"  {c:7s} matched share {n_matched[cty == c].sum() / (Q['country'] == c).sum():.3f}"
              f" | S1 with 0 matches {(n_matched[cty == c] == 0).mean():.3f}")
    print(f"finished in {(time.time() - t0) / 60:.0f} min -> {out}/matching_results.tsv", flush=True)


if __name__ == "__main__":
    main()
