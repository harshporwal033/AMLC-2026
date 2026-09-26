"""Step 1-2: normalise the raw records and generate candidates (blocking) for one split.

    python run_blocking.py --data <dir with train/ and test/ .tsv> --split train --work work
    python run_blocking.py --data <dir> --split test  --work work

Writes <work>/prep/<split>_s1/2/3.parquet (normalised records) and
<work>/blocks/<split>_NNN.parquet (candidates: word view + multilingual-e5 embedding view,
top-10 of each per S2/S3 record, same country). For train also <work>/blocks/train_true_s1.npy.
A GPU is used for the embedding view if available. Resumable: finished files are skipped.
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ber  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder containing train/ and test/ with the .tsv files")
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--work", default="work")
    ap.add_argument("--chunk", type=int, default=1_000_000)
    ap.add_argument("--n_jobs", type=int, default=4)
    args = ap.parse_args()
    prep, blk = f"{args.work}/prep", f"{args.work}/blocks"
    os.makedirs(prep, exist_ok=True)
    os.makedirs(blk, exist_ok=True)

    for k in (1, 2, 3):
        dst = f"{prep}/{args.split}_s{k}.parquet"
        if not os.path.exists(dst):
            t = time.time()
            n = ber.prep_file(f"{args.data}/{args.split}/{args.split}_source{k}.tsv", dst, n_jobs=args.n_jobs)
            print(f"prepared {dst}: {n:,} rows, {time.time() - t:.0f}s", flush=True)

    S1 = ber.load(f"{prep}/{args.split}_s1.parquet")
    Q = pd.concat([ber.load(f"{prep}/{args.split}_s2.parquet"), ber.load(f"{prep}/{args.split}_s3.parquet")],
                  ignore_index=True)
    true_s1 = None
    if args.split == "train":
        gt = pd.read_csv(f"{args.data}/train/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
        pos = pd.Series(np.arange(len(S1)), index=S1["entity_id"])
        true_s1 = Q["entity_id"].map(ber.truth_map(gt)).map(pos).values.astype(np.float64)
        np.save(f"{blk}/train_true_s1.npy", true_s1)
    n_chunks = (len(Q) + args.chunk - 1) // args.chunk
    if len(glob.glob(f"{blk}/{args.split}_[0-9][0-9][0-9].parquet")) == n_chunks:
        print("blocks already complete", flush=True)
        return
    t = time.time()
    blk_obj = ber.Blocker(S1, n_threads=args.n_jobs, embedder=ber.Embedder(), verbose=True)
    print(f"index built in {time.time() - t:.0f}s", flush=True)
    ber.block_all(blk_obj, Q, f"{blk}/{args.split}", chunk=args.chunk, true_s1=true_s1)


if __name__ == "__main__":
    main()
