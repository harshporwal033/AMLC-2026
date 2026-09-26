# Business Entity Resolution – reproduction guide

Pipeline: **normalise → block (word TF-IDF + multilingual embeddings) → stage-1 pair model
(LightGBM) → cross-encoder on uncertain pairs (fine-tuned Qwen3-Reranker-0.6B) → stage-2
competition model (LightGBM) → per-S1 expected-F0.5 decision**.
`src/ber.py` is the shared library; every other file in `src/` is one runnable step.

## Environment

- Python 3.10+, `pip install -r requirements.txt`
- CPU steps: 4 cores, ~30 GB RAM (a Kaggle notebook).
- GPU steps: blocking embeddings (any CUDA GPU; Kaggle T4 was used) and the cross-encoder
  (trained and scored on one A100 40 GB; `train_ce_big.py` / `score_pairs.py` stop with a clear
  message if torch cannot use the GPU).
- Pretrained models (Hugging Face, downloaded on first use; no other external data or lookup):
  `intfloat/multilingual-e5-small` (MIT, 118M) – blocking embeddings;
  `Qwen/Qwen3-Reranker-0.6B` (Apache-2.0, 0.6B) – fine-tuned cross-encoder;
  LightGBM (MIT) – stage 1 and stage 2.

## Data layout

```
DATA/train/train_source{1,2,3}.tsv   DATA/train/train_ground_truth.tsv
DATA/test/test_source{1,2,3}.tsv
```

## Run order

Everything is written to one working folder (`work/`). Every step can be re-run and resumes /
skips finished parts. Scripts 5–11 find their inputs by searching `BER_ROOTS` (colon-separated
folders) and write to `BER_WORK`; below both are `work`.

```bash
export BER_WORK=work BER_ROOTS=work
```

| # | Command | Produces | Time |
|---|---|---|---|
| 1 | `python src/run_blocking.py --data DATA --split train --work work` | `work/prep/train_s*.parquet`, `work/blocks/train_*.parquet`, `train_true_s1.npy` | ~2.5 h, T4 |
| 2 | `python src/run_blocking.py --data DATA --split test --work work` | `work/prep/test_s*.parquet`, `work/blocks/test_*.parquet` | ~2 h, T4 |
| 3 | `python src/build_dev_set.py --work work` | `work/dev_cand.parquet`, `dev_mask.npy`, `ce_pairs.parquet` | ~15 min |
| 4 | `python src/train_ce_big.py --pairs work/ce_pairs.parquet --out work/ce_qwen --base Qwen/Qwen3-Reranker-0.6B --bs 128 --lr 2e-5 --epochs 1 --limit 1000000` | `work/ce_qwen/final_fp16/` | ~75 min, A100 |
| 5 | `python src/dev_stage1.py` | `work/dev_state.pkl`, `work/dev_pack.zip` | ~1.5 h |
| 6 | `unzip -o work/dev_pack.zip -d work && python src/score_pairs.py --pack work/dev_pack --models work/ce_qwen/final_fp16 --names qwen --out work/dev_scores.parquet` | `work/dev_scores.parquet` | ~12 min, A100 |
| 7 | `python src/stage2_from_scores.py` | `work/model6/` (stage-1 + stage-2 models, decision rule) | ~20 min |
| 8 | `BER_STAGE1_ONLY=1 python src/predict_test.py` | `work/test_stage1/` (stage-1 scores of every test candidate) | ~2.3 h |
| 9 | `python src/export_test_pack.py` | `work/test_pack.zip` | ~10 min |
| 10 | `unzip -o work/test_pack.zip -d work && python src/score_pairs.py --pack work/test_pack --models work/ce_qwen/final_fp16 --names qwen --out work/test_scores.parquet` | `work/test_scores.parquet` | ~3.2 h, A100 |
| 11 | `python src/predict_test.py` | `work/output/matching_results.tsv`, `work/output/candidate_pairs.tsv` | ~20 min |

Step 11 prints sanity checks (one row per test S1, only valid S2/S3 IDs, no ID twice) and the
matched share per country. Validate with the organisers' `utils/validate_submission.py`.
`candidate_pairs.tsv` is exactly the candidate set the matching models score (every final match
is a candidate).

## Files

| File | Role |
|---|---|
| `ber.py` | library: normalisation, `Blocker`, features, stage-1/2 training, decision rule, scoring, pair packs |
| `run_blocking.py` | steps 1–2: normalise records, generate candidates |
| `build_dev_set.py` | step 3: dev set + cross-encoder training pairs |
| `train_ce_big.py` | step 4: cross-encoder fine-tuning (standalone, GPU) |
| `dev_stage1.py` | step 5: stage 1 on the dev set + dev pair pack |
| `score_pairs.py` | steps 6, 10: score pair packs with cross-encoder(s) (standalone, GPU, resumable) |
| `stage2_from_scores.py` | step 7: stage 2 with cross-encoder scores → model folder |
| `predict_test.py` | steps 8, 11: test stage 1 (resumable) + stage 2 + decision → submission files |
| `export_test_pack.py` | step 9: test pair pack |
| `train_stage2.py` | alternative to 5–7 when the cross-encoder is scored inside the notebook (small models) |
| `dev_pack_from_state.py` | rebuild `dev_pack.zip` from `dev_state.pkl` |
| `friend_check.py` | 5-minute GPU / training / scoring smoke test with run-time estimates |
