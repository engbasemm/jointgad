"""
compare_old_new.py - is the WoodScape image list the same as in results.txt, and are the images the same?

Usage (from omnidet/):
  python compare_old_new.py old_run_84_images.csv data/val_FR_only.txt \
         WoodScape_ICCV19/output/<eval_v4_folder>/Woodscape/per_image/CODEC_HM_HEVC_QP33.pkl
"""
import csv, os, pickle, sys

old = {r["image"]: r for r in csv.DictReader(open(sys.argv[1]))}
lines = [l.strip() for l in open(sys.argv[2]) if l.strip()]
new_names = [os.path.basename(l.split()[0]) for l in lines]
new_names = [n if n.endswith(".png") else n + ".png" for n in new_names]
o, n = set(old), set(new_names)
print(f"old run: {len(o)} images | val file now: {len(n)} entries | in both: {len(o & n)}")
print("only in old run :", sorted(o - n)[:20])
print("only in val file:", sorted(n - o)[:20])
print("first 5 val-file lines as written:", lines[:5])
if len(sys.argv) > 3:
    comp = pickle.load(open(sys.argv[3], "rb"))["comp"]
    print("\nsame image, HEVC QP33 bpp  old -> new   (equal = same image and preprocessing)")
    diffs = []
    for name in sorted(o & set(comp)):
        b_old, b_new = float(old[name]["qp33_bpp_old"]), comp[name]["BPP"]
        diffs.append(abs(b_old - b_new))
        print(f"  {name}: {b_old:.3f} -> {b_new:.3f}")
    if diffs:
        print(f"max |diff| = {max(diffs):.4f} over {len(diffs)} common images")
    else:
        print("  no common image names between old run and new pickle")
