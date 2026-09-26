"""Train the stage-2 cross-encoder on a big GPU (A100 / H100 / any CUDA GPU).

Standalone: needs only  torch, transformers, pandas, pyarrow, numpy.

Input  : ce_pairs.parquet with columns a (record text), b (S1 text), y (1 match / 0 not),
         produced by ber.export_ce_pairs() on the main machine.
Output : <out>/final_fp16/      the model to use (config + tokenizer + fp16 weights)
         <out>/final_fp16.zip   same, zipped for upload
         <out>/log.txt          training / validation log

Example:
    pip install -U torch transformers pandas pyarrow
    python train_ce_big.py --pairs ce_pairs.parquet --out ce_big --epochs 2

The text format and max_len must match ber.score_ce (defaults do): "name | address", 96 tokens.
"""
import argparse
import math
import os
import shutil
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)


class Pairs(Dataset):
    def __init__(self, a, b, y):
        self.a, self.b, self.y = a, b, y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.a[i], self.b[i], self.y[i]


def make_collate(tok, max_len):
    def collate(batch):
        a, b, y = zip(*batch)
        enc = tok(list(a), list(b), padding=True, truncation=True, max_length=max_len,
                  return_tensors="pt")
        enc["labels"] = torch.tensor(y, dtype=torch.float32)
        return enc
    return collate


def auc(y, s):
    """ROC AUC without sklearn."""
    order = np.argsort(s, kind="stable")
    r = np.empty(len(s)); r[order] = np.arange(1, len(s) + 1)
    pos = y > 0.5
    n1, n0 = pos.sum(), (~pos).sum()
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))


@torch.inference_mode()
def evaluate(model, loader, dev, dtype):
    model.eval()
    ys, ss = [], []
    for enc in loader:
        y = enc.pop("labels")
        enc = {k: v.to(dev, non_blocking=True) for k, v in enc.items()}
        with torch.autocast("cuda", dtype=dtype, enabled=dev == "cuda"):
            s = model(**enc).logits.squeeze(-1).float().cpu().numpy()
        ys.append(y.numpy()); ss.append(s)
    model.train()
    y, s = np.concatenate(ys), np.concatenate(ss)
    p = 1 / (1 + np.exp(-np.clip(s, -30, 30)))
    ll = float(-np.mean(y * np.log(p + 1e-7) + (1 - y) * np.log(1 - p + 1e-7)))
    acc = float(((p > 0.5) == (y > 0.5)).mean())
    return ll, auc(y, s), acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", default="ce_big")
    ap.add_argument("--base", default="intfloat/multilingual-e5-small")  # MIT, 118M params
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max_len", type=int, default=96)
    ap.add_argument("--val_frac", type=float, default=0.01)
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="use only the first N pairs (smoke test)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, "log.txt"), "a")

    def say(msg):
        print(msg, flush=True); log.write(msg + "\n"); log.flush()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bf16 = dev == "cuda" and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    say(f"device {dev} ({torch.cuda.get_device_name(0) if dev == 'cuda' else 'cpu'}), "
        f"precision {'bf16' if bf16 else 'fp16'}, base {args.base}")

    P = pd.read_parquet(args.pairs)
    if args.limit:
        P = P.iloc[:args.limit]
    rng = np.random.default_rng(0)
    is_val = rng.random(len(P)) < args.val_frac
    tr, va = P[~is_val], P[is_val]
    say(f"pairs: train {len(tr):,} ({tr['y'].mean():.2%} positive), val {len(va):,}")

    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSequenceClassification.from_pretrained(args.base, num_labels=1).to(dev)
    collate = make_collate(tok, args.max_len)
    dl = DataLoader(Pairs(tr["a"].tolist(), tr["b"].tolist(), tr["y"].values.astype(np.float32)),
                    batch_size=args.bs, shuffle=True, num_workers=args.workers, collate_fn=collate,
                    pin_memory=dev == "cuda", drop_last=True, persistent_workers=args.workers > 0)
    dl_va = DataLoader(Pairs(va["a"].tolist(), va["b"].tolist(), va["y"].values.astype(np.float32)),
                       batch_size=args.bs * 2, shuffle=False, num_workers=args.workers,
                       collate_fn=collate)

    steps = args.epochs * len(dl)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda" and not bf16)
    lossf = torch.nn.BCEWithLogitsLoss()
    say(f"{steps:,} steps ({len(dl):,} per epoch, batch {args.bs})")

    model.train()
    step, run, t0 = 0, None, time.time()
    for ep in range(args.epochs):
        for enc in dl:
            y = enc.pop("labels").to(dev, non_blocking=True)
            enc = {k: v.to(dev, non_blocking=True) for k, v in enc.items()}
            with torch.autocast("cuda", dtype=dtype, enabled=dev == "cuda"):
                logit = model(**enc).logits.squeeze(-1)
            loss = lossf(logit.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sch.step()
            step += 1
            run = loss.item() if run is None else 0.98 * run + 0.02 * loss.item()
            if step % 200 == 0:
                el = time.time() - t0
                say(f"ep {ep} step {step:,}/{steps:,} loss {run:.4f} | {step * args.bs / el:,.0f} pairs/s "
                    f"| eta {el / step * (steps - step) / 60:.0f} min")
            if step % args.eval_every == 0 or step == steps:
                ll, a, acc = evaluate(model, dl_va, dev, dtype)
                say(f"  VAL step {step:,}: logloss {ll:.4f} | AUC {a:.5f} | acc {acc:.4f}")
        model.save_pretrained(os.path.join(args.out, f"epoch{ep}")); tok.save_pretrained(os.path.join(args.out, f"epoch{ep}"))

    final = os.path.join(args.out, "final_fp16")
    model.half().save_pretrained(final); tok.save_pretrained(final)
    shutil.make_archive(final, "zip", args.out, "final_fp16")
    say(f"done in {(time.time() - t0) / 60:.0f} min -> {final}.zip")


if __name__ == "__main__":
    main()
