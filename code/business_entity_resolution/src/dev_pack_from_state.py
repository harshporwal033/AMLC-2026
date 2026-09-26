"""Rebuild dev_pack.zip from a saved dev_state.pkl without retraining (~10 min).
Reloads the same dev frames (load_dev is deterministic for the same inputs / frac) and exports
the uncertain-band top-1 / runner-up pairs. Env BER_BAND as in dev_stage1.py."""
import glob
import os
import pickle
import sys

WORK = os.environ.get("BER_WORK", "/kaggle/working")
ROOTS = os.environ.get("BER_ROOTS", "/kaggle/input:/kaggle/working").split(":")
sys.path.insert(0, WORK)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ber  # noqa: E402
_missing = [f for f in ["load_dev", "export_pair_pack", "band_mask"] if not hasattr(ber, f)]
if _missing:
    raise SystemExit(f"ber.py is an OLD version (missing {_missing}) - paste the latest ber.py into "
                     f"{ber.__file__} and restart")


def find(name):
    for root in ROOTS:
        hits = sorted(glob.glob(f"{root}/**/{name}", recursive=True))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"{name} not found under {ROOTS}")


def main():
    st = pickle.load(open(find("dev_state.pkl"), "rb"))
    r = st["_run"]
    D = ber.load_dev(os.path.dirname(find("train_s1.parquet")), frac=st.get("frac", 1.0),
                     dev_cand_path=find("dev_cand.parquet"), dev_mask_path=find("dev_mask.npy"),
                     true_s1_path=find("train_true_s1.npy"))
    if len(D["true_s1"]) != len(st["true_s1"]):
        raise SystemExit("dev frames differ from dev_state.pkl - same inputs / frac needed")
    top, sec = r["top"], r["second"]
    sel = ber.band_mask(top["p"].values)
    print(f"queries in band: {sel.mean():.3f} ({sel.sum():,} of {len(sel):,})", flush=True)
    top = top[sel]
    sec = sec[sec["q"].isin(top["q"])]
    ber.export_pair_pack(D["Q"], D["S1"], list(top["q"].values) + list(sec["q"].values),
                         list(top["s1"].values) + list(sec["s1_2"].values), f"{WORK}/dev_pack")


if __name__ == "__main__":
    main()
