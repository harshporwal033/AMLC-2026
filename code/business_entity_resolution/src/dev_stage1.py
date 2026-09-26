"""Dev side, part 1 (Kaggle, before the big-GPU scoring).

load_dev -> stage 1 (+ stage-2 base features) -> coherence features, saved to
<WORK>/dev_state.pkl, and the dev top-1 / runner-up pairs exported as <WORK>/dev_pack.zip
for score_pairs.py.  Later: stage2_from_scores.py (dev_state.pkl + dev_scores.parquet).

Inputs under ROOTS: dev_cand.parquet, dev_mask.npy, train_true_s1.npy, prep/train_s*.parquet.
Env: BER_FRAC (default 1.0), BER_WORK, BER_ROOTS.
"""
import glob
import os
import pickle
import sys
import time

WORK = os.environ.get("BER_WORK", "/kaggle/working")
ROOTS = os.environ.get("BER_ROOTS", "/kaggle/input:/kaggle/working").split(":")
sys.path.insert(0, WORK)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ber  # noqa: E402


def find(name):
    for root in ROOTS:
        hits = sorted(glob.glob(f"{root}/**/{name}", recursive=True))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"{name} not found under {ROOTS}")


def main():
    t0 = time.time()
    frac = float(os.environ.get("BER_FRAC", "1.0"))
    prep = os.path.dirname(find("train_s1.parquet"))
    dev = dict(dev_cand_path=find("dev_cand.parquet"), dev_mask_path=find("dev_mask.npy"),
               true_s1_path=find("train_true_s1.npy"))
    print(f"prep {prep}\n{dev}\nfrac {frac}", flush=True)
    D = ber.load_dev(prep, frac=frac, **dev)
    print(f"dev S1 {D['dev'].sum():,} | queries {len(D['Q']):,} | pairs {len(D['cand']):,}", flush=True)
    ber.run_two_stage(D)
    r = D["_run"]
    coh = ber.coherence_features(r["top"], D["Q"], D["S1"])
    state = {"_run": r, "true_s1": D["true_s1"], "dev": D["dev"], "true_cnt": D["true_cnt"],
             "COH": coh, "frac": frac}
    with open(f"{WORK}/dev_state.pkl", "wb") as fh:
        pickle.dump(state, fh, protocol=4)
    top, sec = r["top"], r["second"]
    q = list(top["q"].values) + list(sec["q"].values)
    s = list(top["s1"].values) + list(sec["s1_2"].values)
    ber.export_pair_pack(D["Q"], D["S1"], q, s, f"{WORK}/dev_pack")
    print(f"saved dev_state.pkl + dev_pack.zip in {(time.time() - t0) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
