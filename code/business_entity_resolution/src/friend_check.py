"""5-minute check before the long GPU runs. Tests everything the real runs use, on a small
slice, and prints how long the full runs will take.

    python friend_check.py --pairs ce_pairs.parquet

Checks: GPU usable by torch -> model download -> training (train_ce_big.py) -> scoring
(score_pairs.py).  Prints ALL CHECKS PASSED + time estimates, or the step that failed.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def run(cmd):
    print("\n$ " + " ".join(cmd), flush=True)
    t = time.time()
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = [l for l in p.stdout.splitlines()
             if re.search(r"device|pairs|step|VAL|done|Error|error|No usable GPU", l)]
    print("\n".join("   " + l for l in lines[-12:]), flush=True)
    if p.returncode != 0:
        print(p.stdout[-3000:])
        sys.exit(f"\nFAILED: {' '.join(cmd[:2])} (exit {p.returncode}) - send this output")
    return p.stdout, time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="ce_pairs.parquet")
    ap.add_argument("--base", default="Qwen/Qwen3-Reranker-0.6B")
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--n", type=int, default=40_000, help="pairs used for the test")
    ap.add_argument("--allow_cpu", action="store_true")
    args = ap.parse_args()

    # 1. GPU
    import torch
    ok = torch.cuda.is_available()
    print(f"[1] torch {torch.__version__} (CUDA build {torch.version.cuda}) | GPU usable: {ok}"
          + (f" | {torch.cuda.get_device_name(0)}" if ok else ""), flush=True)
    if not ok and not args.allow_cpu:
        sys.exit("FAILED [1]: torch cannot use the GPU. Run nvidia-smi, read 'CUDA Version' and install the "
                 "matching torch (python -m pip install --force-reinstall torch --index-url "
                 "https://download.pytorch.org/whl/cu121  | cu118 | cu124 | cu126)")
    cpu = ["--allow_cpu"] if args.allow_cpu else []

    # 2 + 3. download + short training
    import pandas as pd
    P = pd.read_parquet(args.pairs)
    print(f"[2] {args.pairs}: {len(P):,} pairs ({P['y'].mean():.1%} positive)", flush=True)
    small = "check_pairs.parquet"
    P.sample(n=min(args.n, len(P)), random_state=0).to_parquet(small, index=False)
    shutil.rmtree("check_ce", ignore_errors=True)
    out, _ = run([sys.executable, f"{HERE}/train_ce_big.py", "--pairs", small, "--out", "check_ce", "--base", args.base,
                  "--bs", str(args.bs), "--epochs", "1", "--eval_every", "100000", "--val_frac", "0.05"] + cpu)
    m = re.search(r"\(([\d,]+) pairs/s overall\)", out)
    train_speed = float(m.group(1).replace(",", "")) if m else float("nan")
    print(f"[3] training OK: {train_speed:,.0f} pairs/s (includes model load; the full run is a bit faster)", flush=True)

    # 4. scoring on a small pack made from the same pairs
    os.makedirs("check_pack", exist_ok=True)
    S = P.sample(n=min(args.n, len(P)), random_state=1).reset_index(drop=True)
    pd.DataFrame({"q": range(len(S)), "text": S["a"]}).to_parquet("check_pack/q.parquet", index=False)
    pd.DataFrame({"s1": range(len(S)), "text": S["b"]}).to_parquet("check_pack/s1.parquet", index=False)
    pd.DataFrame({"q": range(len(S)), "s1": range(len(S))}).to_parquet("check_pack/pairs.parquet", index=False)
    shutil.rmtree("check_scores.parquet.parts", ignore_errors=True)
    out, _ = run([sys.executable, f"{HERE}/score_pairs.py", "--pack", "check_pack", "--models", "check_ce/final_fp16",
                  "--names", "qwen", "--out", "check_scores.parquet"] + cpu)
    speeds = [float(x.replace(",", "")) for x in re.findall(r"\| ([\d,]+) pairs/s", out)]
    score_speed = max(speeds) if speeds else float("nan")
    sc = pd.read_parquet("check_scores.parquet")
    print(f"[4] scoring OK: {score_speed:,.0f} pairs/s | score range {sc['qwen'].min():.2f} .. {sc['qwen'].max():.2f}", flush=True)

    h = lambda n, s: f"{n / s / 3600:.1f} h" if s == s and s > 0 else "?"
    print("\n==================== ALL CHECKS PASSED ====================")
    print(f"estimated full training (4M pairs, 1 epoch): ~{h(len(P), train_speed)}")
    print(f"estimated scoring: dev_pack (~1.3M pairs) ~{h(1.3e6, score_speed)} | "
          f"test_pack (~19.5M pairs) ~{h(19.5e6, score_speed)}")
    print("send these lines back; then start the full training command.")
    for f in ("check_pairs.parquet", "check_scores.parquet"):
        os.remove(f)
    for d in ("check_ce", "check_pack", "check_scores.parquet.parts"):
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    main()
