"""
bootstrap_ci.py  -  paired bootstrap confidence intervals for thesis requirement R4.

Reads the per-image pickles written by evaluate_mtl_v4.py and reports, for each model
against the baseline, the difference in mAP, mIoU (all classes) and mIoU (no void) with a
95% interval. No network is run: resampling reuses the stored detections.

Usage (from omnidet/, so train_utils resolves):
  python bootstrap_ci.py <per_image_dir> [--baseline RES50_Baseline_no_compression]
                         [--n 1000] [--block 0] [--void 0] [--seed 0]

--block N > 0 resamples blocks of N consecutive frame indices instead of single images,
because neighbouring WoodScape frames are correlated (use e.g. 10 as a sensitivity check).
"""
import argparse
import glob
import os
import pickle
import re

import numpy as np
from train_utils.detection_utils import ap_per_class


def metrics(rec, keys, void):
    stats, labels, conf = [], [], None
    for k in keys:
        d = rec["det"].get(k)
        if d is not None:
            stats.extend(d["stats"])
            labels.extend(d["target_cls"])
        c = rec["seg"].get(k)
        if c is not None:
            conf = c.copy() if conf is None else conf + c
    out = {"mAP": np.nan, "mIoU_all": np.nan, "mIoU_no_void": np.nan}
    if stats and labels:
        tp, sc, pc = [np.concatenate(x, 0) for x in zip(*stats)]
        out["mAP"] = float(np.mean(ap_per_class(tp, sc, pc, np.array(labels))[2]))
    if conf is not None:
        tpd = np.diag(conf).astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = tpd / (conf.sum(0) + conf.sum(1) - tpd)
        out["mIoU_all"] = float(np.nanmean(iou))
        out["mIoU_no_void"] = float(np.nanmean(np.delete(iou, void)))
    return out


def groups(keys, block):
    if block <= 0:
        return [[k] for k in keys]
    g = {}
    for k in keys:
        m = re.match(r"(\d+)", k)
        idx = int(m.group(1)) // block if m else hash(k)
        g.setdefault(idx, []).append(k)
    return list(g.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("per_image_dir")
    ap.add_argument("--baseline", default="RES50_Baseline_no_compression")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--block", type=int, default=0)
    ap.add_argument("--void", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    load = lambda p: pickle.load(open(p, "rb"))
    base = load(os.path.join(a.per_image_dir, a.baseline + ".pkl"))
    rng = np.random.default_rng(a.seed)
    print(f"{'model':<34}{'metric':<14}{'model':>9}{'base':>9}{'diff':>9}{'2.5%':>9}{'97.5%':>9}  CI excludes 0")
    for path in sorted(glob.glob(os.path.join(a.per_image_dir, "*.pkl"))):
        name = os.path.basename(path)[:-4]
        if name == a.baseline:
            continue
        rec = load(path)
        keys = sorted(set(rec["det"]) | set(rec["seg"]))
        keys = [k for k in keys if k in base["det"] or k in base["seg"]]
        grp = groups(keys, a.block)
        point_m, point_b = metrics(rec, keys, a.void), metrics(base, keys, a.void)
        diffs = {m: [] for m in point_m}
        for _ in range(a.n):
            pick = rng.integers(0, len(grp), len(grp))
            ks = [k for i in pick for k in grp[i]]
            mm, mb = metrics(rec, ks, a.void), metrics(base, ks, a.void)
            for m in diffs:
                diffs[m].append(mm[m] - mb[m])
        for m in diffs:
            d = np.array(diffs[m], dtype=float)
            if np.all(np.isnan(d)):
                continue
            lo, hi = np.nanpercentile(d, [2.5, 97.5])
            print(f"{name:<34}{m:<14}{point_m[m]:>9.4f}{point_b[m]:>9.4f}{point_m[m]-point_b[m]:>+9.4f}"
                  f"{lo:>+9.4f}{hi:>+9.4f}  {'yes' if (lo > 0 or hi < 0) else 'no'}")


if __name__ == "__main__":
    main()
