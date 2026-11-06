"""
check_inputs.py - find out why the WoodScape inputs differ between results.txt and the 3 Nov run.

The Fisheye8K rows of the two runs are identical, but on WoodScape even the HEVC / JPEG2000
rows changed. Codecs don't use any checkpoint, so the WoodScape images reaching the script
are different. This prints exactly what the WoodScape loader returns for every sample, so two
setups (old commit / config vs current) can be diffed.

Usage (from omnidet/, same config as evaluate_mtl):
  python check_inputs.py --config <cfg.yaml>  > inputs_now.txt
  (check out the commit / config used for results.txt, run again > inputs_old.txt)
  diff inputs_old.txt inputs_now.txt
"""
import hashlib
import os

import numpy as np
import torch
import yaml

from data_loader.woodscape_loader import WoodScapeRawDataset
from main import collect_args
from utils import Tupperware

config = collect_args()
args = Tupperware(yaml.safe_load(open(config.config)))
print("config:", os.path.abspath(config.config))
for k in ("dataset_dir", "val_file", "input_height", "input_width", "semantic_num_classes",
          "crop", "train", "dataset", "fishey8k_val_file"):
    print(f"  {k} = {getattr(args, k, '<unset>')}")
lines = [l.strip() for l in open(args.val_file) if l.strip()]
print("val_file md5:", hashlib.md5(open(args.val_file, "rb").read()).hexdigest(), "lines:", len(lines))
cams = {}
for l in lines:
    c = l.split("_")[-1].split(".")[0]
    cams[c] = cams.get(c, 0) + 1
print("cameras in val_file:", cams)

ds = WoodScapeRawDataset(data_path=args.dataset_dir, path_file=args.val_file, is_train=False, config=args)
print("dataset length:", len(ds))
print("idx | val_file line | image shape | pixel mean | img md5 | label shape | label classes | n boxes")
for i in range(len(ds)):
    s = ds[i]
    img = s[("color", 0, 0)]
    lab = s.get(("semantic_labels", 0, 0))
    box = s.get(("detection_labels", 0))
    img_md5 = hashlib.md5((img.numpy() * 255).round().astype(np.uint8).tobytes()).hexdigest()[:10]
    lab_cls = sorted(torch.unique(lab).tolist()) if isinstance(lab, torch.Tensor) else None
    nbox = int(box.view(-1, 6).shape[0]) if isinstance(box, torch.Tensor) else 0
    print(f"{i:3d} | {lines[i] if i < len(lines) else '-'} | {tuple(img.shape)} | {img.mean():.4f} | {img_md5} | "
          f"{tuple(lab.shape) if lab is not None else None} | {lab_cls} | {nbox}")
