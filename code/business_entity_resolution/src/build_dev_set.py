"""Step 3: dev set for training / validation + cross-encoder training pairs.

    python build_dev_set.py --work work [--dev_frac 0.02] [--ce_frac 0.06] [--ce_pairs 4000000]

dev set  : a random dev_frac of the train S1 entities, plus every S2/S3 record whose true
           match is one of them or that has one of them in its top-2 (word or embedding
           view), with ALL candidates of those records -> <work>/dev_cand.parquet, dev_mask.npy.
           False matches onto dev S1 are therefore counted as they would be on test.
CE pairs : text pairs from ce_frac of the NON-dev S1 entities (so cross-encoder scores are
           out-of-sample on dev) -> <work>/ce_pairs.parquet (train_ce_big.py input).
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ber  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="work")
    ap.add_argument("--dev_frac", type=float, default=0.02)
    ap.add_argument("--ce_frac", type=float, default=0.06)
    ap.add_argument("--ce_pairs", type=int, default=4_000_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    blk, prep = f"{args.work}/blocks", f"{args.work}/prep"
    files = sorted(glob.glob(f"{blk}/train_[0-9][0-9][0-9].parquet"))
    true_s1 = np.load(f"{blk}/train_true_s1.npy")
    n1 = len(pd.read_parquet(f"{prep}/train_s1.parquet", columns=["entity_id"]))

    dev = np.zeros(n1, bool)
    dev[np.random.default_rng(args.seed).choice(n1, int(n1 * args.dev_frac), replace=False)] = True
    touch = set(np.flatnonzero(np.isfinite(true_s1) & dev[np.nan_to_num(true_s1, nan=0).astype(int)]))
    for f in files:
        c = pd.read_parquet(f, columns=["q", "s1", "word_rank", "emb_rank"])
        near = ((c["word_rank"].values <= 2) | (c["emb_rank"].values <= 2)) & dev[c["s1"].values]
        touch.update(c["q"].values[near].tolist())
    touch = np.array(sorted(touch))
    parts = [c[np.isin(c["q"].values, touch)] for c in (pd.read_parquet(f) for f in files)]
    cand = pd.concat(parts, ignore_index=True)
    cand["is_true"] = cand["s1"].values == true_s1[cand["q"].values]
    cand.to_parquet(f"{args.work}/dev_cand.parquet", index=False)
    np.save(f"{args.work}/dev_mask.npy", dev)
    print(f"dev set: {dev.sum():,} S1 | {len(touch):,} records | {len(cand):,} candidate pairs", flush=True)

    ber.export_ce_pairs(files, prep, true_s1, dev, f"{args.work}/ce_pairs.parquet",
                        frac=args.ce_frac, max_pairs=args.ce_pairs)


if __name__ == "__main__":
    main()
