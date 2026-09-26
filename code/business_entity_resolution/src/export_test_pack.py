"""Test side: export test top-1 / runner-up pairs (from a finished test_stage1/) as
<WORK>/test_pack.zip for score_pairs.py.  CPU is enough.
Inputs under ROOTS: test_stage1/top1.parquet + second.parquet, prep/test_s*.parquet."""
import glob
import os
import sys

import pandas as pd

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
    s1o = os.path.dirname(find("test_stage1/top1.parquet"))
    prep = os.path.dirname(find("test_s1.parquet"))
    top = pd.read_parquet(f"{s1o}/top1.parquet")
    sec = pd.read_parquet(f"{s1o}/second.parquet")
    cols = ["business_name", "business_address"]
    S1 = pd.read_parquet(f"{prep}/test_s1.parquet", columns=cols)
    Q = pd.concat([pd.read_parquet(f"{prep}/test_s{k}.parquet", columns=cols) for k in (2, 3)], ignore_index=True)
    print(f"stage-1 {s1o} | prep {prep} | top {len(top):,} | second {len(sec):,}", flush=True)
    ber.export_pair_pack(Q, S1, list(top["q"].values) + list(sec["q"].values),
                         list(top["s1"].values) + list(sec["s1_2"].values), f"{WORK}/test_pack")


if __name__ == "__main__":
    main()
