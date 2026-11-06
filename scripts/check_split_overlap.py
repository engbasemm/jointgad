"""Confirms the BN-recalibration images (train list) and the evaluation images (val list) do not overlap.
Usage: python check_split_overlap.py data/train_FR_only.txt data/val_FR_only.txt"""
import os, sys
name = lambda l: os.path.basename(l.split()[0]).replace(".png", "")
tr = {name(l) for l in open(sys.argv[1]) if l.strip()}
va = {name(l) for l in open(sys.argv[2]) if l.strip()}
print(f"train {len(tr)} | val {len(va)} | overlap {len(tr & va)}", sorted(tr & va)[:20])
