"""Score a pair pack with one or more cross-encoders on a big GPU (standalone, resumable).

Needs only: torch, transformers, pandas, pyarrow, numpy.

    python score_pairs.py --pack dev_pack   --models ce_qwen/final_fp16 [ce_big/final_fp16 ...] --out dev_scores.parquet
    python score_pairs.py --pack test_pack  --models ce_qwen/final_fp16 [...]                  --out test_scores.parquet

--pack is the unzipped folder with q.parquet, s1.parquet, pairs.parquet (ber.export_pair_pack).
Output: parquet with columns q, s1 and one float column per model (named after --names or the
model folder). Work is saved in chunks next to --out, so an interrupted run resumes.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load(model_dir, dev):
    tok = AutoTokenizer.from_pretrained(model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=1,
                                                           torch_dtype=torch.bfloat16 if dev == "cuda" else torch.float32)
    m.config.pad_token_id = tok.pad_token_id
    return tok, m.to(dev).eval()


@torch.inference_mode()
def score(tok, model, a, b, dev, bs, max_len):
    order = np.argsort([len(x) + len(y) for x, y in zip(a, b)], kind="stable")
    out = np.empty(len(a), np.float32)
    for i in range(0, len(a), bs):
        idx = order[i:i + bs]
        enc = tok([a[j] for j in idx], [b[j] for j in idx], padding=True, truncation=True,
                  max_length=max_len, return_tensors="pt").to(dev)
        out[idx] = model(**enc).logits.squeeze(-1).float().cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--names", nargs="*", default=None, help="column names (default: model folder names)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--max_len", type=int, default=96)
    ap.add_argument("--chunk", type=int, default=1_000_000)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True

    qt = pd.read_parquet(f"{args.pack}/q.parquet").set_index("q")["text"]
    st = pd.read_parquet(f"{args.pack}/s1.parquet").set_index("s1")["text"]
    pairs = pd.read_parquet(f"{args.pack}/pairs.parquet")
    names = args.names or [os.path.basename(os.path.normpath(m)) if os.path.basename(os.path.normpath(m)) != "final_fp16"
                           else os.path.basename(os.path.dirname(os.path.normpath(m))) for m in args.models]
    print(f"device {dev} | {len(pairs):,} pairs | models {dict(zip(names, args.models))}", flush=True)
    part_dir = args.out + ".parts"
    os.makedirs(part_dir, exist_ok=True)
    t0 = time.time()
    for name, md in zip(names, args.models):
        tok, model = load(md, dev)
        for k, i in enumerate(range(0, len(pairs), args.chunk)):
            f = f"{part_dir}/{name}_{k:04d}.npy"
            if os.path.exists(f):
                continue
            p = pairs.iloc[i:i + args.chunk]
            t = time.time()
            s = score(tok, model, qt.loc[p["q"].values].tolist(), st.loc[p["s1"].values].tolist(),
                      dev, args.bs, args.max_len)
            np.save(f + ".tmp.npy", s)
            os.replace(f + ".tmp.npy", f)
            done = min(i + args.chunk, len(pairs))
            print(f"  {name}: {done:,}/{len(pairs):,} pairs | {len(p) / (time.time() - t):,.0f} pairs/s "
                  f"| elapsed {(time.time() - t0) / 60:.0f} min", flush=True)
        del model
        if dev == "cuda":
            torch.cuda.empty_cache()
    out = pairs.copy()
    n = (len(pairs) + args.chunk - 1) // args.chunk
    for name in names:
        out[name] = np.concatenate([np.load(f"{part_dir}/{name}_{k:04d}.npy") for k in range(n)])
    out.to_parquet(args.out, index=False)
    print(f"done -> {args.out} ({(time.time() - t0) / 60:.0f} min)", flush=True)


if __name__ == "__main__":
    main()
