"""
run_sweep.py - runs evaluate_mtl_v4.py once per configuration, then collects everything.

Each configuration only sets the switches of evaluate_mtl_v4.py (environment variables), so
every run uses the same script, checkpoints, data and config. After the runs it writes, in
sweep_<timestamp>/:
  summary.csv       one row per (run, model or codec): mAP, AP per class, mIoU, bpp, PSNR, ...
  summary.md        the same as readable tables
  stats_<run>.txt   stats_tests.py output (bootstrap CI, paired t-test, Wilcoxon) for each run
  logs/<run>.log    full console output of each run
  sweep.json        which run folder belongs to which configuration

Usage (from omnidet/, same arguments you give evaluate_mtl.py, e.g. --config ...):
  python eval/run_sweep.py [--only R3,R4] [--script eval/evaluate_mtl.py] [evaluate args ...]
Runs already finished are skipped when you restart with --resume sweep_<timestamp>.
"""
import argparse
import csv
import datetime
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# name: (purpose, switches). Switches not given take the defaults in evaluate_mtl_v4.py's
# RUN SETTINGS block, so every run states all of them explicitly here.
BASE = {"EVAL_LEGACY_BN": "0", "EVAL_LEGACY_SKIP_EMPTY": "0", "EVAL_LEGACY_PIXELS": "0",
        "EVAL_PERCEPTION_INPUT": "float", "EVAL_BN_RECALIB_N": "0", "EVAL_BN_RECALIB_SCOPE": "no_stats",
        "EVAL_BN_RECALIB_SEED": "0", "EVAL_PIPELINE": "compress_then_perceive", "EVAL_SKIP_CODECS": "1"}
RUNS = {
    # --- reproduction and attribution of each fix (response to R1) ---
    "R0_legacy_all": ("old script behaviour: must reproduce results.txt (except Model A epoch)",
                      dict(EVAL_LEGACY_BN="1", EVAL_LEGACY_SKIP_EMPTY="1", EVAL_LEGACY_PIXELS="1",
                           EVAL_PERCEPTION_INPUT="8bit", EVAL_SKIP_CODECS="0")),
    "R1_as_trained_8bit": ("BatchNorm as each checkpoint was trained, 8-bit perception input",
                           dict(EVAL_PERCEPTION_INPUT="8bit", EVAL_SKIP_CODECS="0")),
    "R2_as_trained_float": ("as R1, perception on the decoder's exact output", dict()),
    "R3_perimage_bn_all": ("sensitivity: identical per-image BatchNorm for every model",
                           dict(EVAL_LEGACY_BN="1", EVAL_SKIP_CODECS="0")),
    # --- BatchNorm recalibration (main proposal) and its robustness ---
    "R4_recal_s0": ("BN recalibration of encoders without statistics, 1000 train images, seed 0",
                    dict(EVAL_BN_RECALIB_N="1000")),
    "R5_recal_s1": ("as R4, seed 1", dict(EVAL_BN_RECALIB_N="1000", EVAL_BN_RECALIB_SEED="1")),
    "R6_recal_s2": ("as R4, seed 2", dict(EVAL_BN_RECALIB_N="1000", EVAL_BN_RECALIB_SEED="2")),
    "R7_recal_all": ("as R4, but EVERY encoder (baseline too) recalibrated identically",
                     dict(EVAL_BN_RECALIB_N="1000", EVAL_BN_RECALIB_SCOPE="all", EVAL_SKIP_CODECS="0")),
    "R8_recal_8bit": ("as R4, 8-bit perception input (isolates the float-input effect)",
                      dict(EVAL_BN_RECALIB_N="1000", EVAL_PERCEPTION_INPUT="8bit")),
    "R9_recal_n300": ("as R4 with 300 images (sensitivity to N)", dict(EVAL_BN_RECALIB_N="300")),
    # --- inference-only ablation of the JointGAD pipeline ---
    "A1_no_reencode": ("ablation: perception on shared encoder features, no re-encoding of x_hat",
                       dict(EVAL_BN_RECALIB_N="1000", EVAL_PIPELINE="uncompressed_features")),
    # --- fixed-rate Factorized (FPDA) models; set EVAL_WEIGHTS to include their checkpoint root ---
    "Z1_factorized_as_trained": ("Factorized models (+ baseline), each as trained",
                                 dict(EVAL_MODELS="Model_AA_Factorized,Model_A_Factorized,Model_B_Factorized,Model_C_Factorized")),
    "Z4_factorized_recal": ("Factorized models, BN re-estimation where statistics are missing",
                            dict(EVAL_MODELS="Model_AA_Factorized,Model_A_Factorized,Model_B_Factorized,Model_C_Factorized",
                                 EVAL_BN_RECALIB_N="1000")),
    "Z7_factorized_recal_all": ("Factorized models, every encoder incl. baseline re-estimated",
                                dict(EVAL_MODELS="Model_AA_Factorized,Model_A_Factorized,Model_B_Factorized,Model_C_Factorized",
                                     EVAL_BN_RECALIB_N="1000", EVAL_BN_RECALIB_SCOPE="all")),
    # --- full validation list (all cameras); needs --val-file ---
    "F1_full_as_trained": ("FULL LIST: each model as trained (+ codecs through the baseline)",
                           dict(EVAL_VAL_FILE="@FULL", EVAL_SKIP_CODECS="0")),
    "F4_full_recal_s0": ("FULL LIST: BN re-estimation, seed 0 (main)", dict(EVAL_VAL_FILE="@FULL", EVAL_BN_RECALIB_N="1000")),
    "F5_full_recal_s1": ("FULL LIST: BN re-estimation, seed 1",
                         dict(EVAL_VAL_FILE="@FULL", EVAL_BN_RECALIB_N="1000", EVAL_BN_RECALIB_SEED="1")),
    "F7_full_recal_all": ("FULL LIST: BN re-estimation for every encoder incl. baseline",
                          dict(EVAL_VAL_FILE="@FULL", EVAL_BN_RECALIB_N="1000", EVAL_BN_RECALIB_SCOPE="all")),
    "FA1_full_sender": ("FULL LIST: heads on sender features f_enc(x)",
                        dict(EVAL_VAL_FILE="@FULL", EVAL_BN_RECALIB_N="1000", EVAL_PIPELINE="uncompressed_features")),
}
FULL_VAL_FILE = None   # set by --val-file
CAMERA_GROUPS = [("all", ""), ("front_rear", "FV,RV"), ("side", "MVL,MVR")]

METRICS = ["BPP", "PSNR", "MS_SSIM", "mAP", "mIoU_all", "mIoU_no_void"]


def run_one(name, switches, script, passthrough, sweep_dir):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([os.path.dirname(HERE), env.get("PYTHONPATH", "")])  # omnidet/
    env["PYTHONUNBUFFERED"] = "1"
    env.update(BASE)
    env.update(switches)
    if env.get("EVAL_VAL_FILE") == "@FULL":
        if not FULL_VAL_FILE:
            raise SystemExit(f"{name} needs --val-file (the full WoodScape validation list)")
        env["EVAL_VAL_FILE"] = FULL_VAL_FILE
    log_path = os.path.join(sweep_dir, "logs", name + ".log")
    out_dir = None
    print(f"\n===== {name}: {RUNS[name][0]}\n      {switches}")
    with open(log_path, "w") as log:
        p = subprocess.Popen([sys.executable, script] + passthrough, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            log.write(line)
            if "[settings]" in line or "results in" in line or ": mAP=" in line or "Error" in line:
                print("   ", line.rstrip())
            m = re.search(r"=> results in (.+)$", line.strip())
            if m:
                out_dir = m.group(1).strip()
        p.wait()
    if p.returncode != 0 or out_dir is None:
        print(f"   !! {name} failed (exit {p.returncode}); see {log_path}")
        return None
    return out_dir


def collect(sweep, sweep_dir):
    rows = []
    for name, out_dir in sweep.items():
        if not out_dir:
            continue
        res = json.load(open(os.path.join(out_dir, "results.json")))
        ws = res.get("Woodscape", {})
        for kind in ("models", "codecs"):
            for model, m in ws.get(kind, {}).items():
                row = {"run": name, "kind": kind[:-1], "name": model}
                row.update({k: m.get(k) for k in METRICS})
                for c, v in (m.get("AP_per_class") or {}).items():
                    row[f"AP_class{c}"] = v
                row["bn_mode_encoder"] = (m.get("bn_mode") or {}).get("encoder")
                row["checkpoint"] = os.path.basename(os.path.dirname(m.get("checkpoint", ""))) + "/" + \
                    os.path.basename(m.get("checkpoint", "")) if m.get("checkpoint") else None
                rows.append(row)
    if not rows:
        return
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(sweep_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(sweep_dir, "summary.md"), "w") as f:
        for name in sweep:
            sub = [r for r in rows if r["run"] == name]
            if not sub:
                continue
            f.write(f"\n### {name}: {RUNS[name][0]}\n\n| | " + " | ".join(METRICS) + " |\n|" + "---|" * (len(METRICS) + 1) + "\n")
            for r in sub:
                vals = ["–" if r[k] is None or (isinstance(r[k], float) and r[k] != r[k]) else f"{r[k]:.4f}" for k in METRICS]
                f.write(f"| {r['name']} | " + " | ".join(vals) + " |\n")
        # one cross-run table per metric for the learned models
        models = list(dict.fromkeys(r["name"] for r in rows if r["kind"] == "model"))
        for metric in ("mAP", "mIoU_all", "BPP"):
            f.write(f"\n### {metric} across runs\n\n| model | " + " | ".join(sweep) + " |\n|" + "---|" * (len(sweep) + 1) + "\n")
            for mod in models:
                cells = []
                for name in sweep:
                    v = next((r[metric] for r in rows if r["run"] == name and r["name"] == mod), None)
                    cells.append("–" if v is None or v != v else f"{v:.4f}")
                f.write(f"| {mod} | " + " | ".join(cells) + " |\n")
    print(f"\n=> {os.path.join(sweep_dir, 'summary.md')} and summary.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated run names, e.g. R4_recal_s0,R7_recal_all")
    ap.add_argument("--script", default=os.path.join(HERE, "evaluate_mtl.py"))
    ap.add_argument("--resume", default=None, help="existing sweep_<timestamp> folder")
    ap.add_argument("--no-stats", action="store_true")
    ap.add_argument("--val-file", default=None, help="full WoodScape validation list for the F* runs")
    a, passthrough = ap.parse_known_args()
    global FULL_VAL_FILE
    FULL_VAL_FILE = a.val_file

    names = [n for n in RUNS if (n in a.only.split(",") if a.only else not n.startswith(("F", "Z")))]
    sweep_dir = a.resume or os.path.join("sweeps", "sweep_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(os.path.join(sweep_dir, "logs"), exist_ok=True)
    jpath = os.path.join(sweep_dir, "sweep.json")
    sweep = json.load(open(jpath)) if os.path.exists(jpath) else {}

    for name in names:
        if sweep.get(name) and os.path.exists(os.path.join(sweep[name], "results.json")):
            print(f"[resume] {name} already done: {sweep[name]}")
            continue
        sweep[name] = run_one(name, RUNS[name][1], a.script, passthrough, sweep_dir)
        json.dump(sweep, open(jpath, "w"), indent=2)
        if sweep[name] and not a.no_stats:
            per_img = os.path.join(sweep[name], "Woodscape", "per_image")
            env = dict(os.environ, PYTHONPATH=os.pathsep.join([os.path.dirname(HERE),
                                                               os.environ.get("PYTHONPATH", "")]))
            groups = CAMERA_GROUPS if name.startswith("F") else [("all", "")]
            for gname, cams in groups:
                tag = name if gname == "all" else f"{name}_{gname}"
                stats_out = os.path.join(sweep_dir, f"stats_{tag}.txt")
                cmd = [sys.executable, os.path.join(HERE, "stats_tests.py"), per_img, "--n", "1000",
                       "--csv", os.path.join(sweep_dir, f"stats_{tag}.csv")] + (["--cams", cams] if cams else [])
                with open(stats_out, "w") as f:
                    subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
                print(f"    stats -> {stats_out}")
    collect({n: sweep.get(n) for n in RUNS if n in sweep}, sweep_dir)


if __name__ == "__main__":
    main()
