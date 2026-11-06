# JointGAD — thesis evaluation code (Part B, Chapter 4)

Evaluation, statistics and results for **JointGAD**: single-camera joint compression and multi-task
perception (detection + segmentation) on WoodScape fisheye images, from the PhD thesis
*Task-Aware Learned Compression and Perception in Automotive Fisheye Camera Systems* (B. A. H. Barakat, Cairo University).

This repository holds what is needed to **reproduce the thesis numbers**; it plugs into the
OmniDet / JointGAD training code base (`omnidet/`), whose models and loaders it imports.

## Layout

| Path | Content |
|---|---|
| `eval/evaluate_mtl_v4.py` | corrected evaluation (strict checkpoint loading, BatchNorm handling, 8-bit protocol, sender-side mode, GMACs/latency) |
| `eval/run_sweep.py` | runs `evaluate_mtl_v4.py` once per setting and collects tables + statistics |
| `eval/stats_tests.py`, `eval/bootstrap_ci.py` | paired bootstrap CIs, paired t / Wilcoxon tests, Holm correction |
| `scripts/` | input and split checks used during the audit |
| `results/sweep_<date>/` | `summary.csv` / `summary.md` per setting, `stats_<setting>.txt`, `sweep.json` |
| `results/results_final.csv`, `results/stats_P1_as_trained.csv` | earlier 84-frame tables kept for traceability |

## Run

```bash
cd omnidet                              # JointGAD / OmniDet code base
cp <this repo>/eval/*.py eval/
python eval/run_sweep.py --config data/params.yaml            # all settings
python eval/run_sweep.py --only FA1 --config data/params.yaml  # one setting
```
Settings (`run_sweep.py`): `F1` as trained, `F4`/`F5` BatchNorm re-estimated (seeds 0/1), `F7` re-estimation
for every encoder, `FA1` sender side (heads on the encoder features of the original image, as on board).

## Main result (full WoodScape validation list, sender side, `results/sweep_20261104_111940`)

| Model | bpp | PSNR | MS-SSIM | mAP | mIoU |
|---|---|---|---|---|---|
| ResNet-50 baseline, uncompressed | – | – | – | 0.1997 | 0.4919 |
| JointGAD A | 0.750 | 26.47 | 0.940 | 0.2041 | 0.4967 |
| JointGAD B | 0.527 | 25.19 | 0.927 | 0.1959 | 0.5070 |
| JointGAD C | 0.241 | 25.61 | 0.926 | 0.2028 | 0.5065 |
| JointGAD D | 0.097 | 21.95 | 0.858 | 0.2104 | 0.5078 |

Statistics for every row are in `results/sweep_20261104_111940/stats_FA1_full_sender.txt`.

## Not in this repository
Checkpoints and the WoodScape dataset (licence); see the thesis for their sources.
