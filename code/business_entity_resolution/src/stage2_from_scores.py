"""Dev side, part 2: stage 2 with cross-encoder scores computed elsewhere -> model folder.

Needs dev_state.pkl (dev_stage1.py) and dev_scores.parquet (score_pairs.py on dev_pack).
Tries: coherence only, each scored model alone, and the average of all models; keeps the
best (test-like macro F0.5) and saves <WORK>/<BER_NAME, default model6>/ with lgb1 / config2
(stage 1), lgb2 / config4 (stage 2, "ce_cols" = score columns to average) + zip.
predict_test.py then needs test_scores.parquet (same model columns) instead of CE models."""
import glob
import json
import os
import pickle
import shutil
import sys

WORK = os.environ.get("BER_WORK", "/kaggle/working")
ROOTS = os.environ.get("BER_ROOTS", "/kaggle/input:/kaggle/working").split(":")
sys.path.insert(0, WORK)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd  # noqa: E402
import ber  # noqa: E402


def find(name):
    for root in ROOTS:
        hits = sorted(glob.glob(f"{root}/**/{name}", recursive=True))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"{name} not found under {ROOTS}")


def main():
    name = os.environ.get("BER_NAME", "model6")
    D = pickle.load(open(find("dev_state.pkl"), "rb"))
    S = pd.read_parquet(find("dev_scores.parquet"))
    models = [c for c in S.columns if c not in ("q", "s1")]
    print(f"dev scores: {len(S):,} pairs, models {models}", flush=True)
    r = D["_run"]
    variants = {"coh": None}
    for m in models:
        variants[m] = [m]
    if len(models) > 1:
        variants["avg_all"] = models
    res = {}
    for v, cols in variants.items():
        extra = [D["COH"]] + ([ber.ce_features_from_scores(r["top"], r["second"], S, cols)] if cols else [])
        res[v] = ber.stage2_again(D, extra=extra, label=f"+coh +CE[{v}]" if cols else "+coh (no CE)")
    best = os.environ.get("BER_VARIANT") or max(res, key=lambda k: res[k][0])   # BER_VARIANT forces one
    f, rule, m2, f2 = res[best]
    print(f"\nCHOSEN {best}: test-like F0.5 {f:.4f} rule {rule}", flush=True)
    out = f"{WORK}/{name}"
    os.makedirs(out, exist_ok=True)
    r["m1"].save_model(f"{out}/lgb1.txt")
    json.dump({"f1": r["m1"].feature_name()}, open(f"{out}/config2.json", "w"))
    m2.save_model(f"{out}/lgb2.txt")
    json.dump({"f2": f2, "rule": rule, "coherence": True, "ce": [], "ce_band": None,
               "ce_cols": variants[best] or [], "dev_f05": f, "variant": best, "frac": D.get("frac")},
              open(f"{out}/config4.json", "w"))
    shutil.make_archive(out, "zip", WORK, name)
    print(f"saved {out} + {out}.zip", flush=True)


if __name__ == "__main__":
    main()
