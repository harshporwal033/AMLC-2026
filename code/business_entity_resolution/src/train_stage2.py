"""Final training on the dev set with an out-of-sample cross-encoder -> a complete model folder.

  stage 1 (pair model)            trained on the dev candidate set (frac of dev S1)
  stage 2 (+ coherence, + CE)     CE = cross-encoder(s) trained on NON-dev entities
                                  (train_ce_big.py on export_ce_pairs output), so their dev
                                  scores are out-of-sample and no cross-fitting is needed
  output  <WORK>/<name>/          lgb1.txt config2.json (stage 1)  lgb2.txt config4.json (stage 2)
                                  + copies of the CE model folders -> use with predict_test.py

Inputs found under ROOTS: dev_cand.parquet, dev_mask.npy, train_true_s1.npy, train_s1.parquet
(prep dir with train_s1/2/3.parquet), and cross-encoder folders (a config.json whose folder
path contains one of BER_CE, default "final_fp16").

Env: BER_FRAC (default 1.0 = full dev set), BER_CE (comma list of folder-name substrings),
     BER_NAME (output folder, default model5), BER_WORK, BER_ROOTS.
Kaggle: %%writefile ber.py and this file into /kaggle/working, then
    %run /kaggle/working/train_stage2.py      (GPU; "Save & Run All" to run in background)
"""
import glob
import json
import os
import shutil
import sys
import time

import numpy as np

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
    name = os.environ.get("BER_NAME", "model5")
    pats = [p for p in os.environ.get("BER_CE", "final_fp16").split(",") if p]
    ce_dirs = sorted({os.path.dirname(h) for r in ROOTS
                      for h in glob.glob(f"{r}/**/config.json", recursive=True)
                      if any(p in os.path.dirname(h) for p in pats)
                      and os.path.exists(os.path.join(os.path.dirname(h), "tokenizer_config.json"))})
    prep = os.path.dirname(find("train_s1.parquet"))
    dev = dict(dev_cand_path=find("dev_cand.parquet"), dev_mask_path=find("dev_mask.npy"),
               true_s1_path=find("train_true_s1.npy"))
    print(f"prep {prep}\n{dev}\ncross-encoders {ce_dirs}\nfrac {frac} -> {WORK}/{name}", flush=True)
    if not ce_dirs:
        raise FileNotFoundError(f"no cross-encoder folder matching {pats}")

    D = ber.load_dev(prep, frac=frac, **dev)
    print(f"dev S1 {D['dev'].sum():,} | queries {len(D['Q']):,} | pairs {len(D['cand']):,}", flush=True)
    ber.run_two_stage(D)
    COH = ber.coherence_features(D["_run"]["top"], D["Q"], D["S1"])
    t = time.time()
    CE = ber.ce_features_for(D, ce_dirs)
    print(f"cross-encoder dev scoring {time.time() - t:.0f}s", flush=True)
    p1 = D["_run"]["top"]["p"].values
    band = (p1 > 0.02) & (p1 < 0.995)
    CEb = CE.copy()
    CEb.loc[~band, :] = np.nan
    print(f"share of queries in CE band: {band.mean():.3f}", flush=True)
    res = {"coh": ber.stage2_again(D, extra=[COH], label="+coh (no CE)"),
           "coh_ce": ber.stage2_again(D, extra=[COH, CE], label="+coh +CE (all rows)"),
           "coh_ce_band": ber.stage2_again(D, extra=[COH, CEb], label="+coh +CE (band)")}
    # band is much faster on test; take it unless it is clearly worse
    use = "coh_ce_band" if res["coh_ce_band"][0] >= res["coh_ce"][0] - 0.001 else "coh_ce"
    if res["coh"][0] > res[use][0]:
        use = "coh"
    f, rule, m2, f2 = res[use]
    print(f"\nCHOSEN {use}: test-like F0.5 {f:.4f} rule {rule}", flush=True)

    out = f"{WORK}/{name}"
    os.makedirs(out, exist_ok=True)
    D["_run"]["m1"].save_model(f"{out}/lgb1.txt")
    json.dump({"f1": D["_run"]["m1"].feature_name()}, open(f"{out}/config2.json", "w"))
    m2.save_model(f"{out}/lgb2.txt")
    ce_names = []
    if use != "coh":
        for k, d in enumerate(ce_dirs):
            shutil.copytree(d, f"{out}/ce_{k}", dirs_exist_ok=True)
            ce_names.append(f"ce_{k}")
    json.dump({"f2": f2, "rule": rule, "coherence": True, "ce": ce_names,
               "ce_band": [0.02, 0.995] if use == "coh_ce_band" else None,
               "dev_f05": f, "variant": use, "frac": frac, "ce_src": ce_dirs},
              open(f"{out}/config4.json", "w"))
    shutil.make_archive(out, "zip", WORK, name)
    print(f"saved {out} and {out}.zip in {(time.time() - t0) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
