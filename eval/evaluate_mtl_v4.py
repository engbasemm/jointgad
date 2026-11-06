"""
evaluate_mtl_v4.py  -  corrected Part B evaluation (thesis requirements R1, R3, R4, R14).

Derived from evaluate_mtl_v3.py and the later evaluate_mtl version. Place it next to them
(omnidet/eval/) so the same imports resolve. Numbers in [S#] refer to the supervisor's code review.

Fixes
  [S1]  BatchNorm runs as each checkpoint was trained: with its running statistics when the
        checkpoint has them (baseline), with per-image statistics when it was trained with
        track_running_stats=False and so has none (the Cheng models). Recorded as bn_mode.
  [S2]  Predictions on images without ground-truth boxes count as false positives.
  [S3]  Every checkpoint loads strictly; a 'module.' prefix is stripped; any missing or
        unexpected key stops the run.
  [S4]  One distortion protocol for every method: the network input and every reconstruction
        (learned or codec) are 8-bit images, rounded from [0, 1]. PSNR / MS-SSIM use them.
  [S5]  Image names come from the dataset's own index, not from the validation file line order.
  [S6]  Complexity is reported as GMACs and as FLOPs (= 2 x GMACs). Parameters and model size
        include the detection and segmentation heads.
  [S7]  Fixed-rate bpp is measured from the transmitted tensor (numel x bits / H x W); the
        configured value is kept beside it, and an entropy estimate if the module provides one.
  [S8]  mIoU is reported with and without the void class.
  [S9]  Latency: warm-up, synchronised, median and P95, for (i) encoder, (ii) compression
        forward, (iii) the full perception pipeline including the second encoder pass,
        (iv) real arithmetic coding when the module has compress()/decompress().
        Codec latency is measured in a separate sequential pass with no parallel workers.
  [S10] Runs on CPU (CUDA synchronisation only when CUDA is used).
  [S11] HEVC gets the frame size and frame count on the command line.
  [S12] Optional visual outputs show what the network actually saw: the 8-bit input it was
        given, boxes drawn on that input, no min-max normalisation.
  [S13] The perception input is an explicit choice (EVAL_PIPELINE), written to the results.
  Also: codec images go through the uncompressed baseline (codec mAP/mIoU are measured, not
  typed in); per-image results are saved for bootstrap_ci.py; run_meta.json records platform,
  GPU, git commit, flags and image list; JPEG2000 settings are named by compression level.

Legacy switches reproduce results.txt, so each change can be attributed (response to R1):
  EVAL_LEGACY_BN=1          per-image BatchNorm statistics for every model, also those
                            that have running statistics (as v3 did)  [S1]
  EVAL_LEGACY_SKIP_EMPTY=1  empty images skipped                      [S2]
  EVAL_LEGACY_PIXELS=1      float input, unclamped x_hat, v3 codec input conversion  [S4]
Other settings:
  EVAL_PIPELINE=compress_then_perceive (default, as v3) | uncompressed_features
  EVAL_SKIP_CODECS=1        skip HEVC / JPEG2000
  EVAL_CODEC_TIMING_N=30    images used for the sequential codec-latency pass
  EVAL_SAVE_VIS=1           save perception inputs and detection overlays
  EVAL_J2K_PIXFMT=yuv444p   optional JPEG2000 diagnostic (default: RGB as before)
  EVAL_VOID_INDEX=0         index of the void class
  EVAL_WEIGHTS=/a/MTL_bitrates:/b/other  checkpoint roots, searched in order (':'-separated)
  EVAL_MODELS=Model_A_Factorized,...      evaluate only these models (+ baseline)
  EVAL_VAL_FILE=data/val.txt  WoodScape list to evaluate (default: val_file of the config)
  MODEL_A_SUFFIX=...        Model A checkpoint (default: first of MODEL_A_CANDIDATES that exists)
  Defaults for all switches are in the RUN SETTINGS block below the imports.

Run order (inference only):
  1. all three EVAL_LEGACY_* = 1, 84-image list -> must reproduce results.txt
     (strict loading stays on; if it stops with missing keys, results.txt came from a partly
     random encoder and that must be reported)
  2. switch the legacy flags off one at a time  -> size of each correction
  3. no legacy flags, full 1,766-frame list     -> the thesis numbers
"""
import os
import sys
import math
import time
import json
import yaml
import shutil
import socket
import pickle
import platform
import datetime
import subprocess
import tempfile
import multiprocessing

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from pytorch_msssim import ms_ssim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from thop import profile

from data_loader.woodscape_loader import WoodScapeRawDataset
from train_utils.detection_utils import non_max_suppression, xywh2xyxy, get_batch_statistics, ap_per_class
from main import collect_args
from models.detection_decoderORG import YoloDecoder
from models.resnet import ResnetEncoder
from models.semantic_decoder import SemanticDecoder
from utils import Tupperware
from models.compressAI.priors import FactorizedPriorDecoderAtten, FactorizedPriorDecoderAttenQtz
from models.compressAI.waseda import Cheng2020AttentionDecoder
import warnings as _warnings
import models.compressAI.priors as _priors_mod
import models.compressAI.waseda as _waseda_mod

# The local CompressAI copies call warnings.warn() in compress()/update() without importing
# 'warnings' (seen as NameError in the 3 Nov v4 run); supply it so real entropy coding can run.
for _m in (_priors_mod, _waseda_mod):
    if not hasattr(_m, "warnings"):
        _m.warnings = _warnings


# --------------------------------------------------------------------------------------
# Switches
# --------------------------------------------------------------------------------------
def _flag(name, default=False):
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes")


# ===== RUN SETTINGS: edit here (an environment variable of the same name overrides) =====
RUN_LEGACY_BN = False           # EVAL_LEGACY_BN: per-image BatchNorm for every model (as v3 / results.txt)
RUN_LEGACY_SKIP_EMPTY = False   # EVAL_LEGACY_SKIP_EMPTY
RUN_LEGACY_PIXELS = False       # EVAL_LEGACY_PIXELS
# What the second encoder pass sees for learned models (EVAL_PERCEPTION_INPUT):
#   "8bit"  : x_hat clamped and rounded to 8-bit, as if the reconstruction were stored
#   "float" : x_hat exactly as the decoder outputs it (how the networks were trained, if
#             training fed encoder(x_hat) without clamping). PSNR / MS-SSIM always use 8-bit.
RUN_PERCEPTION_INPUT = "float"
# BatchNorm recalibration (EVAL_BN_RECALIB_N): no weight is trained. Encoders whose checkpoint
# has no running statistics get them computed once from N TRAINING images (forward passes only,
# cumulative average), then run in standard eval mode like the baseline. 0 = off.
# Scope (EVAL_BN_RECALIB_SCOPE): "no_stats" = only those encoders; "all" = every encoder,
# baseline included, with the identical procedure (strictest like-for-like comparison).
RUN_BN_RECALIB_N = 1000
RUN_BN_RECALIB_SCOPE = "no_stats"
RUN_BN_RECALIB_BATCH = 8
RUN_BN_RECALIB_SEED = 0          # EVAL_BN_RECALIB_SEED: which training images are drawn
# Model A: first checkpoint in this list that exists on disk is used (MODEL_A_SUFFIX overrides).
# 98/96300 = results.txt / Table 4.4; 103/101100 = later script / 3 Nov run; 107/105000 also exists.
MODEL_A_CANDIDATES = ["98/96300", "103/101100"]
# =========================================================================================

LEGACY_BN_PER_IMAGE = _flag("EVAL_LEGACY_BN", RUN_LEGACY_BN)
LEGACY_SKIP_EMPTY = _flag("EVAL_LEGACY_SKIP_EMPTY", RUN_LEGACY_SKIP_EMPTY)
LEGACY_PIXELS = _flag("EVAL_LEGACY_PIXELS", RUN_LEGACY_PIXELS)
PERCEPTION_INPUT = os.environ.get("EVAL_PERCEPTION_INPUT", RUN_PERCEPTION_INPUT)
assert PERCEPTION_INPUT in ("8bit", "float"), PERCEPTION_INPUT
# EVAL_MODELS: comma-separated model names to evaluate (the baseline is always included)
MODEL_FILTER = [m.strip() for m in os.environ.get("EVAL_MODELS", "").split(",") if m.strip()]
BN_RECALIB_N = int(os.environ.get("EVAL_BN_RECALIB_N", RUN_BN_RECALIB_N))
BN_RECALIB_SCOPE = os.environ.get("EVAL_BN_RECALIB_SCOPE", RUN_BN_RECALIB_SCOPE)
BN_RECALIB_SEED = int(os.environ.get("EVAL_BN_RECALIB_SEED", RUN_BN_RECALIB_SEED))
assert BN_RECALIB_SCOPE in ("no_stats", "all"), BN_RECALIB_SCOPE
assert not (BN_RECALIB_N and LEGACY_BN_PER_IMAGE), "BN recalibration and EVAL_LEGACY_BN exclude each other"
print(f"[settings] BN_RECALIB_N={BN_RECALIB_N} BN_RECALIB_SCOPE={BN_RECALIB_SCOPE}")
print(f"[settings] LEGACY_BN={LEGACY_BN_PER_IMAGE} LEGACY_SKIP_EMPTY={LEGACY_SKIP_EMPTY} "
      f"LEGACY_PIXELS={LEGACY_PIXELS} PERCEPTION_INPUT={PERCEPTION_INPUT}")
SKIP_CODECS = _flag("EVAL_SKIP_CODECS")
SAVE_VIS = _flag("EVAL_SAVE_VIS")
PIPELINE = os.environ.get("EVAL_PIPELINE", "compress_then_perceive")
VOID_INDEX = int(os.environ.get("EVAL_VOID_INDEX", "0"))
CODEC_TIMING_N = int(os.environ.get("EVAL_CODEC_TIMING_N", "30"))
J2K_PIXFMT = os.environ.get("EVAL_J2K_PIXFMT", "")
assert PIPELINE in ("compress_then_perceive", "uncompressed_features"), PIPELINE

# --------------------------------------------------------------------------------------
# Models. lam = lambda from the folder name (R9): the checkpoints say 0.08 / 0.05 / 0.02 / 0.01.
# "yolo12" in some folder names is a leftover label; every detection head is YOLOv3.
# bits = bits per transmitted value for fixed-rate models (32 = unquantised float).
# --------------------------------------------------------------------------------------
DL_MODEL_CONFIGS = [
    dict(name="RES50_Baseline_no_compression", compression_decoder_type="not_applicable",
         pretrained_path_segment="res50_no_compress_FR_2Classes_yolo3_dbg128_conf30_20_mtl_regul_0.1_anchort5",
         batch_suffix="99/97500", resnet_type="RES50", fixed_bpp=None, bits=None, lam=None),
    # Model A was evaluated at epochs 98 (results.txt, Table 4.4; checkpoint no longer on disk),
    # 107 (thesis "second evaluation") and 103 (later script). Default: 103. State the rule used.
    dict(name="Model_A_ChengAtten", compression_decoder_type="chengAttenDecoder",
         pretrained_path_segment="res50_cheng_attn0.08_FR_2Classes_yolo3_dbg128_conf30_20_mtl_regul_0.1_anchort5",
         batch_suffix=os.environ.get("MODEL_A_SUFFIX", MODEL_A_CANDIDATES[0]), resnet_type="RES50",
         batch_suffix_candidates=([os.environ["MODEL_A_SUFFIX"]] if os.environ.get("MODEL_A_SUFFIX")
                                  else MODEL_A_CANDIDATES),
         fixed_bpp=None, bits=None, lam=0.08),
    dict(name="Model_B_ChengAtten", compression_decoder_type="chengAttenDecoder",
         pretrained_path_segment="res50_chengAttenDecoder.05_FR_2Classes_yolo3_dbg128_conf30_20_mtl_regul_0.1_anchort5",
         batch_suffix="105/103800", resnet_type="RES50", fixed_bpp=None, bits=None, lam=0.05),
    dict(name="Model_C_ChengAtten", compression_decoder_type="chengAttenDecoder",
         pretrained_path_segment="res50_cheng_attn0.02_FR_2Classes_yolo12_dbg128_conf30_20_mtl_regul_0.1_anchort5",
         batch_suffix="124/122100", resnet_type="RES50", fixed_bpp=None, bits=None, lam=0.02),
    dict(name="Model_D_ChengAtten", compression_decoder_type="chengAttenDecoder",
         pretrained_path_segment="res50_chengAttenDecoder.01_FR_2Classes_yolo3_dbg128_conf30_20_mtl_regul_0.1_anchort5",
         batch_suffix="114/111900", resnet_type="RES50", fixed_bpp=None, bits=None, lam=0.01),
    dict(name="Model_AA_Factorized", compression_decoder_type="factorizedPriorDecoder",
         pretrained_path_segment="res50_factorized_b8_FR_2Classes_yolo3_dbg128_conf30_20_mtl_regul_0.1_anchort5",
         batch_suffix="138/68100", resnet_type="RES50", fixed_bpp=8.0, bits=32, lam=None,
         note="unquantised features: a control, not a compressed model"),
    dict(name="Model_A_Factorized", compression_decoder_type="factorizedPriorDecoderQuantz",
         pretrained_path_segment="res50_factorizedQunatz8_0.01_FR_2Classes_yolo12_dbg128_conf30_20_mtl_regul_0.1_anchort5",
         batch_suffix="75/74400", resnet_type="RES50", fixed_bpp=2.0, bits=8, lam=None),
    dict(name="Model_B_Factorized", compression_decoder_type="factorizedPriorDecoderQuantz",
         pretrained_path_segment="res50_factorizedQunatz6_0.01_FR_2Classes_yolo12_dbg128_conf30_20_mtl_regul_0.1_anchort5",
         batch_suffix="86/84600", resnet_type="RES50", fixed_bpp=1.5, bits=6, lam=None),
    # TODO: fill in Model_C_Factorized (ResNet-18, 2.0 bpp, in results.txt). Skipped while missing.
    dict(name="Model_C_Factorized", compression_decoder_type="factorizedPriorDecoderQuantz",
         pretrained_path_segment="TODO_res18_factorized_folder", batch_suffix="TODO", resnet_type="RES18",
         fixed_bpp=2.0, bits=8, lam=None),
]

HEVC_QPS = {"HM HEVC QP31": 31, "HM HEVC QP33": 33, "HM HEVC QP36": 36, "HM HEVC QP39": 39,
            "HM HEVC QP42": 42, "HM HEVC QP45": 45}
# JPEG2000 is named by ffmpeg compression_level, because "Q" meant different levels in
# different script versions (v3: Q4 = 95; later: Q4 = 75, Q5 = 95). libopenjpeg uses
# compression ratio = 2 x level, so level 95 -> 24 / 190 = 0.126 bpp (the thesis "Q4").
J2K_QS = {"JPEG2000 cl10": 10, "JPEG2000 cl15": 15, "JPEG2000 cl25": 25, "JPEG2000 cl50": 50,
          "JPEG2000 cl75": 75, "JPEG2000 cl95": 95}


# --------------------------------------------------------------------------------------
# Pixels [S4]
# --------------------------------------------------------------------------------------
def to_8bit(x):
    """Round a [0, 1] tensor to 8-bit levels (what a stored or transmitted image holds)."""
    return x if LEGACY_PIXELS else (x.clamp(0, 1) * 255).round() / 255


def tensor_to_pil(x):
    x = x.detach().cpu()
    if x.dim() == 4:
        x = x.squeeze(0)
    if LEGACY_PIXELS:  # v3: ToPILImage truncates (x * 255).byte()
        return transforms.ToPILImage()(x)
    return Image.fromarray((x.clamp(0, 1) * 255).round().byte().permute(1, 2, 0).numpy())


def psnr(a, b):
    mse = F.mse_loss(a.float(), b.float()).item()
    return float("inf") if mse == 0 else -10 * math.log10(mse)


# --------------------------------------------------------------------------------------
# Data [S5]
# --------------------------------------------------------------------------------------
class ImageOnlyDataset(Dataset):
    def __init__(self, data_path, path_file, is_train, config):
        self.data_path = data_path
        self.filenames = [l.strip() for l in open(path_file) if l.strip()]
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize((config.input_height, config.input_width))

    def __getitem__(self, index):
        p = os.path.join(self.data_path, self.filenames[index])
        return {("color", 0, 0): self.to_tensor(self.resize(Image.open(p).convert("RGB"))),
                ("detection_labels", 0): None, ("semantic_labels", 0, 0): None, "image_path": p}

    def __len__(self):
        return len(self.filenames)


class Named(Dataset):
    """Attaches the sample's own file name, taken from the dataset itself."""

    def __init__(self, ds, val_file):
        self.ds = ds
        names = None
        for attr in ("image_paths", "filenames", "file_names", "paths"):
            if isinstance(getattr(ds, attr, None), (list, tuple)):
                names = list(getattr(ds, attr))
                break
        if names is None:
            names = [l.strip() for l in open(val_file) if l.strip()]
        if len(names) != len(ds):
            raise RuntimeError(f"name list ({len(names)}) and dataset ({len(ds)}) differ in length")
        self.names = names

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        s = self.ds[i]
        p = s.get("image_path") if isinstance(s.get("image_path"), str) else self.names[i]
        s["_key"] = os.path.basename(p.strip())
        return s


def collate(batch):
    out, labels = {}, []
    for sample in batch:
        lab = sample.get(("detection_labels", 0))
        if lab is not None:
            lab = lab.view(-1, 6)
            if lab.numel():
                labels.append(lab)
        for k, v in sample.items():
            if k != ("detection_labels", 0) and isinstance(v, torch.Tensor):
                out.setdefault(k, []).append(v)
    out = {k: torch.stack(v) for k, v in out.items()}
    out["detection_labels"] = torch.cat(labels, 0) if labels else torch.empty((0, 6))
    out["_key"] = [s["_key"] for s in batch]
    return out


# --------------------------------------------------------------------------------------
# Metrics [S2, S8]
# --------------------------------------------------------------------------------------
def seg_confusion(pred, gt, k):
    p = pred.reshape(-1).cpu().numpy().astype(np.int64)
    g = gt.reshape(-1).cpu().numpy().astype(np.int64)
    m = (g >= 0) & (g < k)
    return np.bincount(k * g[m] + p[m], minlength=k * k).reshape(k, k)


def summarize(det_records, seg_confs):
    res = {"mAP": float("nan"), "AP_per_class": {}, "mIoU_all": float("nan"), "mIoU_no_void": float("nan")}
    recs = [r for r in det_records.values() if r is not None]
    stats = [s for r in recs for s in r["stats"]]
    labels = [c for r in recs for c in r["target_cls"]]
    if stats and labels:
        tp, conf, pcls = [np.concatenate(x, 0) for x in zip(*stats)]
        _, _, ap, _, ap_cls = ap_per_class(tp, conf, pcls, np.array(labels))
        res["mAP"] = float(np.mean(ap))
        res["AP_per_class"] = {int(c): float(a) for c, a in zip(ap_cls, ap)}
    res["n_images_det"] = len(recs)
    res["n_images_without_objects"] = sum(1 for r in recs if not r["target_cls"])
    if seg_confs:
        conf = sum(seg_confs.values())
        tp = np.diag(conf).astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = tp / (conf.sum(0) + conf.sum(1) - tp)
        res["mIoU_all"] = float(np.nanmean(iou))
        res["mIoU_no_void"] = float(np.nanmean(np.delete(iou, VOID_INDEX)))
        res["IoU_per_class"] = [float(x) for x in iou]
    return res


# --------------------------------------------------------------------------------------
# Loading [S3]
# --------------------------------------------------------------------------------------
def load_strict(module, path, device, filter_extra=False):
    sd = torch.load(path, map_location=device)
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    if filter_extra:  # encoder.pth also stores non-weight entries (e.g. height, width)
        own = module.state_dict()
        dropped = [k for k in sd if k not in own]
        sd = {k: v for k, v in sd.items() if k in own}
        if dropped:
            print(f"  {os.path.basename(path)}: ignoring non-weight entries {dropped}")
    try:
        missing, unexpected = module.load_state_dict(sd, strict=False)
    except TypeError:
        # CompressAI models override load_state_dict(state_dict) without a 'strict' argument
        # (it first resizes the entropy-model buffers). Their override loads strictly by
        # default, so any missing or unexpected key raises here.
        module.load_state_dict(sd)
        return "strict (CompressAI loader)"
    # Checkpoints trained with track_running_stats=False hold no BatchNorm running statistics.
    # That is the only kind of missing key accepted: those layers can only run on the
    # statistics of the image itself, which is how the trained network normalised (batch stats).
    stat_keys = [k for k in missing if k.endswith(("running_mean", "running_var", "num_batches_tracked"))]
    other = [k for k in missing if k not in stat_keys]
    if other or unexpected:
        raise RuntimeError(f"{path}\n  missing: {other}\n  unexpected: {unexpected}")
    no_stats = sorted({k.rsplit(".", 1)[0] for k in stat_keys})
    if no_stats:
        bn_all = [n for n, m in module.named_modules() if isinstance(m, nn.BatchNorm2d)]
        if set(no_stats) != set(bn_all):
            raise RuntimeError(f"{path}: running statistics missing for only some BatchNorm layers "
                               f"({len(no_stats)} of {len(bn_all)}); refusing to mix modes")
        bn_legacy_per_image(module)
        print(f"  {os.path.basename(path)}: no BatchNorm running statistics in checkpoint "
              f"-> {len(no_stats)} layers use per-image statistics (as trained)")
        return "per_image (checkpoint has no running statistics)"
    return "running_stats"


def make_comp_decoder(kind, n_layers):
    if kind == "chengAttenDecoder":
        return Cheng2020AttentionDecoder(64)
    if kind == "factorizedPriorDecoder":
        return FactorizedPriorDecoderAtten(64, 1024 if n_layers == 50 else 256)
    if kind == "factorizedPriorDecoderQuantz":
        return FactorizedPriorDecoderAttenQtz(64, 1024 if n_layers == 50 else 256)
    raise ValueError(kind)


@torch.no_grad()
def bn_recalibrate(bundle, args, device, n_images, batch_size, seed=0):
    """Give the encoder BatchNorm running statistics from training images (no weight update).

    The statistics cover every pass the encoder makes at test time: the input image and,
    for compress_then_perceive, the perception input built from x_hat.
    """
    enc = bundle["enc"]
    for m in enc.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.track_running_stats = True
            m.running_mean = torch.zeros(m.num_features, device=device)
            m.running_var = torch.ones(m.num_features, device=device)
            m.num_batches_tracked = torch.tensor(0, dtype=torch.long, device=device)
            m.momentum = None  # cumulative average over all recalibration batches
    ds = Named(WoodScapeRawDataset(data_path=args.dataset_dir, path_file=args.train_file,
                                   is_train=False, config=args), args.train_file)  # no augmentation
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:n_images].tolist()
    loader = DataLoader(torch.utils.data.Subset(ds, idx), batch_size=batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate, drop_last=False)
    enc.train()
    for b in tqdm(loader, desc="BN recalibration"):
        x = to_8bit(b[("color", 0, 0)].to(device))
        f = enc(x)
        if bundle["comp"] is not None and PIPELINE == "compress_then_perceive":
            xh = bundle["comp"](f)["x_hat"]
            enc(xh if PERCEPTION_INPUT == "float" else to_8bit(xh))
    enc.eval()
    return f"running_stats (recalibrated on {len(idx)} training images, batch {batch_size}, seed {seed})"


def bn_legacy_per_image(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None


def roots(args, cfg=None):
    """Checkpoint roots to search: a model's own 'root' first, then every EVAL_WEIGHTS entry."""
    r = [cfg["root"]] if cfg and cfg.get("root") else []
    return r + [x for x in args.pretrained_weights.split(os.pathsep) if x]


def run_dir_of(cfg, args, suffix=None):
    suffix = suffix or cfg["batch_suffix"]
    cands = [os.path.join(rt, cfg["pretrained_path_segment"], "models", f"weights_{suffix}") for rt in roots(args, cfg)]
    return next((d for d in cands if os.path.isdir(d)), cands[0])


def models_dir_of(cfg, args):
    cands = [os.path.join(rt, cfg["pretrained_path_segment"], "models") for rt in roots(args, cfg)]
    return next((d for d in cands if os.path.isdir(d)), cands[0])


def resolve_suffix(cfg, args):
    """For models with candidate checkpoints, pick the first that exists and record it."""
    for suf in cfg.get("batch_suffix_candidates", []):
        d = run_dir_of(cfg, args, suf)
        if os.path.isdir(d):
            if suf != cfg["batch_suffix"]:
                print(f"  {cfg['name']}: weights_{cfg['batch_suffix']} not found -> using weights_{suf}")
            cfg["batch_suffix"] = suf
            return


def load_bundle(cfg, args, device, with_det, with_sem):
    resolve_suffix(cfg, args)
    run_dir = run_dir_of(cfg, args)
    if not os.path.isdir(run_dir):
        return None
    n_layers = int(cfg["resnet_type"].replace("RES", ""))
    enc = ResnetEncoder(num_layers=n_layers, pretrained=False).to(device)
    enc.eval()
    bn_mode = {"encoder": load_strict(enc, os.path.join(run_dir, "encoder.pth"), device, filter_extra=True)}
    if LEGACY_BN_PER_IMAGE and bn_mode["encoder"] == "running_stats":
        bn_legacy_per_image(enc)
        bn_mode["encoder"] = "per_image (forced by EVAL_LEGACY_BN)"
    det = sem = comp = None
    if with_det:
        det = YoloDecoder(enc.num_ch_enc, args).to(device)
        bn_mode["det"] = load_strict(det, os.path.join(run_dir, "detection.pth"), device)
        det.eval()
    if with_sem:
        sem = SemanticDecoder(enc.num_ch_enc, n_classes=args.semantic_num_classes).to(device)
        bn_mode["sem"] = load_strict(sem, os.path.join(run_dir, "semantic.pth"), device)
        sem.eval()
    if cfg["compression_decoder_type"] != "not_applicable":
        comp = make_comp_decoder(cfg["compression_decoder_type"], n_layers).to(device)
        bn_mode["comp"] = load_strict(comp, os.path.join(run_dir, "decoder.pth"), device)
        comp.eval()
    files = ["encoder.pth", "detection.pth", "semantic.pth", "decoder.pth"]
    size_mb = sum(os.path.getsize(os.path.join(run_dir, f)) for f in files
                  if os.path.exists(os.path.join(run_dir, f))) / 2 ** 20
    return dict(run_dir=run_dir, enc=enc, det=det, sem=sem, comp=comp, n_layers=n_layers, size_mb=size_mb,
                bn_mode=bn_mode)


# --------------------------------------------------------------------------------------
# Rates [S7]
# --------------------------------------------------------------------------------------
LATENT_KEYS = ("y_hat", "y_q", "y", "latent", "features_q", "q")


def rate_fields(cfg, out, h, w):
    r = {"BPP_entropy": float("nan"), "BPP_fixed_measured": float("nan"),
         "BPP_fixed_configured": cfg["fixed_bpp"], "latent_key": None,
         "output_keys": ",".join(f"{k}:{tuple(v.shape)}" for k, v in out.items() if isinstance(v, torch.Tensor))}
    if isinstance(out.get("likelihoods"), dict) and out["likelihoods"]:
        r["BPP_entropy"] = sum(torch.log(l).sum() / (-math.log(2) * h * w)
                               for l in out["likelihoods"].values()).item()
    if cfg["bits"] is not None:
        for k in LATENT_KEYS:
            t = out.get(k)
            if isinstance(t, torch.Tensor):
                r["BPP_fixed_measured"] = t[0].numel() * cfg["bits"] / (h * w)
                r["latent_key"] = k
                break
    if cfg["fixed_bpp"] is None:
        r["BPP"] = r["BPP_entropy"]
    elif not math.isnan(r["BPP_fixed_measured"]):
        r["BPP"] = r["BPP_fixed_measured"]
    else:
        r["BPP"] = cfg["fixed_bpp"]  # flagged in the summary as not measured
    return r


# --------------------------------------------------------------------------------------
# Perception [S2, S12]
# --------------------------------------------------------------------------------------
@torch.no_grad()
def perceive(bundle, feats, inputs, img_size, args):
    det_rec, conf, nms = None, None, None
    if bundle["det"] is not None:
        targets = inputs["detection_labels"].cpu().clone()
        target_cls = targets[:, 1].tolist() if targets.numel() else []
        if not (LEGACY_SKIP_EMPTY and targets.numel() == 0):
            if targets.numel():
                targets[:, 2:6] = xywh2xyxy(targets[:, 2:6])
                targets[:, [2, 4]] *= img_size[0]
                targets[:, [3, 5]] *= img_size[1]
            yolo = bundle["det"](feats, img_dim=img_size)["yolo_outputs"]
            nms = non_max_suppression(yolo, conf_thres=args.detection_conf_thres, nms_thres=args.detection_nms_thres)
            det_rec = {"stats": get_batch_statistics(nms, targets, iou_threshold=0.5, args=args),
                       "target_cls": target_cls}
    if bundle["sem"] is not None and ("semantic_labels", 0, 0) in inputs:
        pred = bundle["sem"](feats)[("semantic", 0)].argmax(1)
        conf = seg_confusion(pred, inputs[("semantic_labels", 0, 0)], args.semantic_num_classes)
    return det_rec, conf, nms


def save_vis(folder, key, img, nms):
    os.makedirs(folder, exist_ok=True)
    stem = os.path.splitext(key)[0]
    pil = tensor_to_pil(img)
    pil.save(os.path.join(folder, f"{stem}_input.png"))  # exactly what the network saw
    if nms is not None and nms[0] is not None:
        arr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        for b in torch.cat(nms, 0).cpu().numpy():
            cv2.rectangle(arr, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
        cv2.imwrite(os.path.join(folder, f"{stem}_detections.png"), arr)


# --------------------------------------------------------------------------------------
# Codecs [S4, S11]
# --------------------------------------------------------------------------------------
NAN_CODEC = {"BPP": np.nan, "PSNR": np.nan, "MS_SSIM": np.nan, "encoding_time": np.nan, "decoding_time": np.nan}


def _run(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _quality(orig, png):
    rec = transforms.ToTensor()(Image.open(png).convert("RGB")).unsqueeze(0)
    o = orig.float().cpu()
    return psnr(o, rec), ms_ssim(o, rec, data_range=1.0).item()


def compress_hevc(orig, qp, cfg_main, cfg_seq, enc_bin, dec_bin, keep_png=None):
    pil = tensor_to_pil(orig)
    w, h = pil.size
    with tempfile.TemporaryDirectory() as d:
        png, yuv, bit, dyuv, out = (os.path.join(d, n) for n in ("in.png", "in.yuv", "b.bin", "d.yuv", "out.png"))
        pil.save(png)
        try:
            _run(["ffmpeg", "-y", "-loglevel", "error", "-i", png, "-s", f"{w}x{h}", "-pix_fmt", "yuv420p", yuv])
            t = time.perf_counter()
            _run([enc_bin, "-c", cfg_main, "-c", cfg_seq, "-i", yuv, "-wdt", str(w), "-hgt", str(h), "-f", "1",
                  f"--QP={qp}", "--ConformanceWindowMode=1", "-o", os.devnull, "-b", bit])
            te = time.perf_counter() - t
            bpp = os.path.getsize(bit) * 8 / (w * h)
            t = time.perf_counter()
            _run([dec_bin, "-b", bit, "-o", dyuv])
            td = time.perf_counter() - t
            _run(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{w}x{h}",
                  "-i", dyuv, out])
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"HEVC QP{qp} failed: {e}")
            return dict(NAN_CODEC)
        p, s = _quality(orig, out)
        if keep_png:
            os.makedirs(os.path.dirname(keep_png), exist_ok=True)
            shutil.copy(out, keep_png)
        return {"BPP": bpp, "PSNR": p, "MS_SSIM": s, "encoding_time": te, "decoding_time": td}


def compress_j2k(orig, level, keep_png=None):
    pil = tensor_to_pil(orig)
    w, h = pil.size
    with tempfile.TemporaryDirectory() as d:
        png, j2k, out = (os.path.join(d, n) for n in ("in.png", "e.j2k", "out.png"))
        pil.save(png)
        enc = ["ffmpeg", "-y", "-loglevel", "error", "-i", png]
        if J2K_PIXFMT:
            enc += ["-pix_fmt", J2K_PIXFMT]
        enc += ["-c:v", "libopenjpeg", "-compression_level", str(level), j2k]
        try:
            t = time.perf_counter()
            _run(enc)
            te = time.perf_counter() - t
            bpp = os.path.getsize(j2k) * 8 / (w * h)
            t = time.perf_counter()
            _run(["ffmpeg", "-y", "-loglevel", "error", "-i", j2k, "-pix_fmt", "rgb24", out])
            td = time.perf_counter() - t
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"JPEG2000 level {level} failed: {e}")
            return dict(NAN_CODEC)
        p, s = _quality(orig, out)
        if keep_png:
            os.makedirs(os.path.dirname(keep_png), exist_ok=True)
            shutil.copy(out, keep_png)
        return {"BPP": bpp, "PSNR": p, "MS_SSIM": s, "encoding_time": te, "decoding_time": td}


def codec_worker(task):
    key, orig, codec_dir, hm = task
    stem = os.path.splitext(key)[0]
    out = {}
    for name, qp in HEVC_QPS.items():
        out[name] = compress_hevc(orig, qp, *hm, keep_png=os.path.join(codec_dir, name, stem + ".png"))
    for name, level in J2K_QS.items():
        out[name] = compress_j2k(orig, level, keep_png=os.path.join(codec_dir, name, stem + ".png"))
    return key, out


# --------------------------------------------------------------------------------------
# Complexity and latency [S6, S9, S10]
# --------------------------------------------------------------------------------------
def _timer(device, n_warm=50, n_run=500):
    def timeit(fn):
        sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
        for _ in range(n_warm):
            fn()
        sync()
        ts = []
        for _ in range(n_run):
            t = time.perf_counter()
            fn()
            sync()
            ts.append((time.perf_counter() - t) * 1e3)
        return float(np.median(ts)), float(np.percentile(ts, 95))
    return timeit


@torch.no_grad()
def profile_bundle(bundle, h, w, device):
    x = torch.randn(1, 3, h, w, device=device).clamp(0, 1)
    img_size = [w, h]
    parts = {"encoder": bundle["enc"], "compression": bundle["comp"],
             "detection_head": bundle["det"], "segmentation_head": bundle["sem"]}
    r = {"size_MB_all_checkpoints": bundle["size_mb"]}
    for name, m in parts.items():
        r[f"Params_M_{name}"] = sum(p.numel() for p in m.parameters()) / 1e6 if m is not None else 0.0
    r["Params_M_total"] = sum(v for k, v in r.items() if k.startswith("Params_M_"))
    feats = bundle["enc"](x)
    macs = {"encoder": profile(bundle["enc"], inputs=(x,), verbose=False)[0]}
    if bundle["comp"] is not None:
        macs["compression"] = profile(bundle["comp"], inputs=(feats,), verbose=False)[0]
    for k, v in macs.items():
        r[f"GMACs_{k}"] = v / 1e9
        r[f"GFLOPs_{k}"] = 2 * v / 1e9

    timeit = _timer(device)

    def pipeline():
        f = bundle["enc"](x)
        if bundle["comp"] is not None:
            xh = bundle["comp"](f)["x_hat"]
            xh = xh if PERCEPTION_INPUT == "float" else to_8bit(xh)
            f = bundle["enc"](xh) if PIPELINE == "compress_then_perceive" else f
        if bundle["det"] is not None:
            bundle["det"](f, img_dim=img_size)
        if bundle["sem"] is not None:
            bundle["sem"](f)

    r["encoder_ms"] = timeit(lambda: bundle["enc"](x))
    r["full_pipeline_ms"] = timeit(pipeline)
    if bundle["comp"] is not None:
        r["compression_forward_ms"] = timeit(lambda: bundle["comp"](feats))
        comp = bundle["comp"]
        if hasattr(comp, "compress") and hasattr(comp, "decompress"):
            try:
                if hasattr(comp, "update"):
                    comp.update(force=True)
                enc_out = comp.compress(feats)
                n_bytes = sum(len(s[0]) for s in enc_out["strings"])
                r["BPP_actual_bitstream"] = n_bytes * 8 / (h * w)
                r["arithmetic_encode_ms"] = timeit(lambda: comp.compress(feats))
                r["arithmetic_decode_ms"] = timeit(lambda: comp.decompress(enc_out["strings"], enc_out["shape"]))
            except Exception as e:  # module has no working entropy coder
                r["arithmetic_coding"] = f"not available: {type(e).__name__}: {e}"
        else:
            r["arithmetic_coding"] = "not available: module has no compress()/decompress()"
    r["latency_note"] = "(median, P95) ms, batch 1, after 50 warm-up runs; random input of the stated size"
    return r


def codec_latency(keys_and_images, hm, n):
    """Sequential, single process, so other work does not distort the times [S9]."""
    rows = {}
    for key, orig in keys_and_images[:n]:
        for name, qp in HEVC_QPS.items():
            rows.setdefault(name, []).append(compress_hevc(orig, qp, *hm))
        for name, level in J2K_QS.items():
            rows.setdefault(name, []).append(compress_j2k(orig, level))
    return {name: {"encode_s_median": float(np.nanmedian([r["encoding_time"] for r in v])),
                   "decode_s_median": float(np.nanmedian([r["decoding_time"] for r in v])),
                   "n_images": len(v),
                   "note": "wall time incl. process start-up and file I/O (reference software)"}
            for name, v in rows.items()}


# --------------------------------------------------------------------------------------
# One dataset
# --------------------------------------------------------------------------------------
@torch.no_grad()
def evaluate_dataset(args, ds, out_root, device):
    h, w = int(args.input_height), int(args.input_width)
    img_size = [w, h]
    labelled = ds["has_detection"] or ds["has_semantic"]
    base_ds = (WoodScapeRawDataset(data_path=ds["data_path"], path_file=ds["val_file"], is_train=False, config=args)
               if labelled else ImageOnlyDataset(ds["data_path"], ds["val_file"], False, args))
    dataset = Named(base_ds, ds["val_file"])
    loader = lambda: DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers,
                                pin_memory=True, drop_last=False, collate_fn=collate)
    out_dir = os.path.join(out_root, ds["name"])
    per_img_dir = os.path.join(out_dir, "per_image")
    codec_dir = os.path.join(out_dir, "codec_decoded")
    os.makedirs(per_img_dir, exist_ok=True)
    summary = {"models": {}, "codecs": {}, "codec_latency": {}}

    # ---- codecs: rate and distortion (parallel) + latency (sequential) ----
    codec_metrics = {}
    if not SKIP_CODECS:
        hm = (args.hevc_encoder_cfg_main, args.hevc_encoder_cfg_per_sequence,
              args.tapp_encoder_path, args.tapp_decoder_path)
        items = [(b["_key"][0], to_8bit(b[("color", 0, 0)]).cpu()) for b in loader()]
        tasks = [(k, img, codec_dir, hm) for k, img in items]
        with multiprocessing.Pool(max(1, (os.cpu_count() or 2) // 2)) as pool:
            for key, m in tqdm(pool.imap_unordered(codec_worker, tasks), total=len(tasks), desc="codecs"):
                codec_metrics[key] = m
        summary["codec_latency"] = codec_latency(items, hm, CODEC_TIMING_N)

    # ---- learned models ----
    for cfg in DL_MODEL_CONFIGS:
        if MODEL_FILTER and cfg is not DL_MODEL_CONFIGS[0] and cfg["name"] not in MODEL_FILTER:
            continue
        bundle = load_bundle(cfg, args, device, ds["has_detection"], ds["has_semantic"])
        if bundle is None:
            missing = run_dir_of(cfg, args)
            if cfg is DL_MODEL_CONFIGS[0]:
                raise SystemExit(f"Baseline checkpoint not found (every comparison needs it):\n  {missing}")
            models_dir = models_dir_of(cfg, args)
            avail = []
            if os.path.isdir(models_dir):
                for e in sorted(os.listdir(models_dir)):
                    sub = os.path.join(models_dir, e)
                    avail += [f"{e[len('weights_'):]}/{s2}" for s2 in sorted(os.listdir(sub))] \
                        if e.startswith("weights_") and os.path.isdir(sub) else []
            print(f"[skip] {cfg['name']}: checkpoint folder not found:\n  {missing}\n"
                  f"  available batch_suffix values: {avail if avail else 'none (run folder missing)'}")
            continue
        if BN_RECALIB_N and (BN_RECALIB_SCOPE == "all" or bundle["bn_mode"]["encoder"].startswith("per_image")):
            bundle["bn_mode"]["encoder"] = bn_recalibrate(bundle, args, device, BN_RECALIB_N, RUN_BN_RECALIB_BATCH, BN_RECALIB_SEED)
            print(f"  {cfg['name']}: {bundle['bn_mode']['encoder']}")
        det_recs, seg_confs, comp_rows = {}, {}, {}
        for inputs in tqdm(loader(), desc=cfg["name"]):
            key = inputs.pop("_key")[0]
            inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
            x = to_8bit(inputs[("color", 0, 0)])
            feats0 = bundle["enc"](x)
            seen = x
            if bundle["comp"] is not None:
                out = bundle["comp"](feats0)
                x_hat = to_8bit(out["x_hat"])
                x_hat_perc = out["x_hat"] if PERCEPTION_INPUT == "float" else x_hat
                row = rate_fields(cfg, out, h, w)
                row.update({"PSNR": psnr(x, x_hat), "MS_SSIM": ms_ssim(x, x_hat, data_range=1.0).item()})
                comp_rows[key] = row
                if PIPELINE == "compress_then_perceive":
                    feats, seen = bundle["enc"](x_hat_perc), x_hat
                else:
                    feats = feats0
            else:
                feats = feats0
            if labelled:
                det_recs[key], conf, nms = perceive(bundle, feats, inputs, img_size, args)
                if conf is not None:
                    seg_confs[key] = conf
                if SAVE_VIS:
                    save_vis(os.path.join(out_dir, "vis", cfg["name"]), key, seen, nms)
        res = summarize(det_recs, seg_confs) if labelled else {}
        for m in ("BPP", "BPP_entropy", "BPP_fixed_measured", "PSNR", "MS_SSIM"):
            vals = [r[m] for r in comp_rows.values() if r.get(m) is not None]
            res[m] = float(np.nanmean(vals)) if vals else float("nan")
        if comp_rows:
            first = next(iter(comp_rows.values()))
            res["decoder_output_keys"] = first.get("output_keys")
            res["latent_key_for_fixed_bpp"] = first.get("latent_key")
        if comp_rows and cfg["fixed_bpp"] is not None:
            res["BPP_fixed_configured"] = cfg["fixed_bpp"]
            res["BPP_is_measured"] = not math.isnan(res["BPP_fixed_measured"])
        res.update({"n_images": len(dataset), "lambda": cfg["lam"], "checkpoint": bundle["run_dir"], "bn_mode": bundle["bn_mode"],
                    "note": cfg.get("note")})
        res.update(profile_bundle(bundle, h, w, device))
        summary["models"][cfg["name"]] = res
        with open(os.path.join(per_img_dir, cfg["name"] + ".pkl"), "wb") as f:
            pickle.dump({"det": det_recs, "seg": seg_confs, "comp": comp_rows}, f)
        print(f"{cfg['name']}: mAP={res.get('mAP')} mIoU_all={res.get('mIoU_all')} "
              f"mIoU_no_void={res.get('mIoU_no_void')} bpp={res['BPP']}")
        del bundle
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- codecs: perception through the uncompressed baseline ----
    if codec_metrics:
        base = load_bundle(DL_MODEL_CONFIGS[0], args, device, ds["has_detection"], ds["has_semantic"])
        for name in list(HEVC_QPS) + list(J2K_QS):
            det_recs, seg_confs = {}, {}
            if labelled and base is not None:
                for inputs in tqdm(loader(), desc=f"perception {name}"):
                    key = inputs.pop("_key")[0]
                    png = os.path.join(codec_dir, name, os.path.splitext(key)[0] + ".png")
                    if not os.path.exists(png):
                        continue
                    inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
                    x_dec = transforms.ToTensor()(Image.open(png).convert("RGB")).unsqueeze(0).to(device)
                    det_recs[key], conf, nms = perceive(base, base["enc"](x_dec), inputs, img_size, args)
                    if conf is not None:
                        seg_confs[key] = conf
                    if SAVE_VIS:
                        save_vis(os.path.join(out_dir, "vis", name.replace(" ", "_")), key, x_dec, nms)
            res = summarize(det_recs, seg_confs) if labelled else {}
            rows = [codec_metrics[k][name] for k in codec_metrics]
            for m in ("BPP", "PSNR", "MS_SSIM"):
                res[m] = float(np.nanmean([r[m] for r in rows]))
            summary["codecs"][name] = res
            with open(os.path.join(per_img_dir, "CODEC_" + name.replace(" ", "_") + ".pkl"), "wb") as f:
                pickle.dump({"det": det_recs, "seg": seg_confs,
                             "comp": {k: codec_metrics[k][name] for k in codec_metrics}}, f)
    return summary


def run_meta(args, datasets, device):
    def sh(cmd):
        try:
            return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return None
    return {
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "host": socket.gethostname(), "platform": platform.platform(), "python": sys.version.split()[0],
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "git_commit": sh(["git", "rev-parse", "HEAD"]), "git_dirty": bool(sh(["git", "status", "--porcelain"])),
        "script": os.path.abspath(__file__),
        "flags": {"LEGACY_BN_PER_IMAGE": LEGACY_BN_PER_IMAGE, "LEGACY_SKIP_EMPTY": LEGACY_SKIP_EMPTY,
                  "LEGACY_PIXELS": LEGACY_PIXELS, "PERCEPTION_INPUT": PERCEPTION_INPUT, "BN_RECALIB_N": BN_RECALIB_N,
                  "BN_RECALIB_SCOPE": BN_RECALIB_SCOPE, "BN_RECALIB_SEED": BN_RECALIB_SEED, "PIPELINE": PIPELINE, "VOID_INDEX": VOID_INDEX,
                  "SKIP_CODECS": SKIP_CODECS, "J2K_PIXFMT": J2K_PIXFMT or "rgb (default)"},
        "input_hw": [int(args.input_height), int(args.input_width)],
        "bpp_definition": "bits / (input_height x input_width) of the network input",
        "detection_conf_thres": args.detection_conf_thres, "detection_nms_thres": args.detection_nms_thres,
        "semantic_num_classes": args.semantic_num_classes,
        "loader_settings": {k: getattr(args, k, None) for k in
                            ("dataset", "crop", "discard_mirrors", "ego_mask", "input_height", "input_width",
                             "dataset_dir", "rgb_images", "val_file")},
        "models": {c["name"]: f"{c['pretrained_path_segment']}/weights_{c['batch_suffix']}" for c in DL_MODEL_CONFIGS},
        "val_files": {d["name"]: d["val_file"] for d in datasets},
    }


if __name__ == "__main__":
    config = collect_args()
    args = Tupperware(yaml.safe_load(open(config.config)))
    hm_base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    args.hevc_encoder_cfg_main = os.path.join(hm_base, "HM", "cfg", "encoder_intra_main.cfg")
    args.hevc_encoder_cfg_per_sequence = os.path.join(hm_base, "HM", "cfg", "per-sequence", "Traffic_Fisheye8K.cfg")
    args.tapp_encoder_path = os.path.join(hm_base, "HM", "bin", "TAppEncoderStatic")
    args.tapp_decoder_path = os.path.join(hm_base, "HM", "bin", "TAppDecoderStatic")
    # Same folder the earlier evaluate_mtl used; override with EVAL_WEIGHTS=/path/to/MTL_bitrates/
    # EVAL_VAL_FILE: evaluate on another WoodScape list (e.g. the full four-camera validation list)
    if os.environ.get("EVAL_VAL_FILE"):
        args.val_file = os.environ["EVAL_VAL_FILE"]
        print("[settings] WoodScape val_file =", args.val_file)
    args.pretrained_weights = os.environ.get(
        "EVAL_WEIGHTS",
        "/media/PorableSSD/Work/Academic/PHDWork/development/MTL/WoodScapeGit/omnidet/"
        "pretrained_models/pretrained_models/MyModel/MTL_bitrates/")
    if not any(os.path.isdir(x) for x in args.pretrained_weights.split(os.pathsep) if x):
        sys.exit(f"Checkpoint root not found: {args.pretrained_weights}\n"
                 f"Is the portable SSD mounted? Otherwise set EVAL_WEIGHTS to the MTL_bitrates folder.")
    print("checkpoint root:", args.pretrained_weights)

    tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag += "".join(f"_{n}" for n, on in (("legacyBN", LEGACY_BN_PER_IMAGE), ("legacyEmpty", LEGACY_SKIP_EMPTY),
                                           ("legacyPix", LEGACY_PIXELS), ("percFloat", PERCEPTION_INPUT == "float"),
                                           (f"bnRecal{BN_RECALIB_N}{BN_RECALIB_SCOPE}s{BN_RECALIB_SEED}", BN_RECALIB_N > 0)) if on)
    out_root = os.path.join(args.output_directory, f"eval_v4_{tag}")
    os.makedirs(out_root, exist_ok=True)
    shutil.copy(config.config, os.path.join(out_root, "config.yaml"))

    datasets = [
        dict(name="Woodscape", data_path=args.dataset_dir, val_file=args.val_file,
             has_semantic=True, has_detection=True),
        dict(name="Fisheye8k_Image_Only", data_path=args.fishey8k_dataset_dir, val_file=args.fishey8k_val_file,
             has_semantic=False, has_detection=False),
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for _cfg in DL_MODEL_CONFIGS:  # resolve checkpoint fallbacks so run_meta records what is used
        resolve_suffix(_cfg, args)
    meta = run_meta(args, datasets, device)
    with open(os.path.join(out_root, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    results = {"meta": meta}
    for ds in datasets:
        results[ds["name"]] = evaluate_dataset(args, ds, out_root, device)
        with open(os.path.join(out_root, "results.json"), "w") as f:
            json.dump(results, f, indent=2, default=float)
    print(f"=> results in {out_root}")
