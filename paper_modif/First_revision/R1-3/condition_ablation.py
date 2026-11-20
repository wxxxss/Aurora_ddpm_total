#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reviewer 1 Comment 3: paired solar-wind condition ablation.

This runner keeps the final paper checkpoint fixed and breaks the event-to-event
correspondence of selected condition variables by permutation.  All variants of
a given test sample use the same artificial mask and the same diffusion random
seed, so the comparison is paired.

Default experiment:
- held-out January 2005 OVATION Prime maps;
- 6 h candidate spacing, evenly reduced to 48 samples across the month;
- manuscript mask: 60--80 MLAT and 18--06 MLT;
- variants: full, permute Bx/By/Bz/Vsw/Pdyn, permute all;
- optional first-order linearized solar-wind encoder diagnostic;
- paired-bootstrap 95% CIs and paired Wilcoxon p-values.

Run from repository root:
    python paper_modif/R1-3/condition_ablation.py

Quick smoke test:
    python paper_modif/R1-3/condition_ablation.py --max-samples 4 --skip-linearized
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    import torch_npu  # noqa: F401
except Exception:
    torch_npu = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
REPO_ROOT = SCRIPT_PATH.parents[2]
for _p in (str(REPO_ROOT), str(SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models.unet import UNet
from models.ddpm import DDPM
from condition_ablation_utils import (
    CONDITION_NAMES,
    apply_condition_variant,
    bootstrap_mean_ci,
    compute_masked_metrics,
    create_paper_mask,
    make_derangement,
    metric_degradation,
    paired_wilcoxon_pvalue,
)

DEFAULT_OVATION = Path(
    "/home/docker/data/private/AuroraData/generated_aurora_data/2005_omni_aurora/"
    "aurora_img_20050101.npy"
)
DEFAULT_OMNI = Path(
    "/home/docker/data/private/AuroraData/omni_real_data/omni_1min_pro/2005/"
    "omni_20050101_1min.npy"
)
DEFAULT_CHECKPOINT = REPO_ROOT / "ckpt/cond/aurora_diff_best.pth"
DEFAULT_NORM_PARAMS = REPO_ROOT / "ckpt/cond/norm_params.pkl"
DEFAULT_TRAIN_OMNI_ROOT = Path(
    "/home/docker/data/private/AuroraData/omni_real_data/omni_5min/1996"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "r1_3_results"

VARIABLE_VARIANTS = tuple(f"permute_{name}" for name in CONDITION_NAMES)
BASE_VARIANTS = ("full",) + VARIABLE_VARIANTS + ("permute_all",)
METRICS = ("ssim", "psnr", "rmse", "r2", "mae")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="R1.3 solar-wind condition ablation")
    p.add_argument("--ovation", type=Path, default=DEFAULT_OVATION)
    p.add_argument("--omni", type=Path, default=DEFAULT_OMNI)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--norm-params", type=Path, default=DEFAULT_NORM_PARAMS)
    p.add_argument("--training-omni-root", type=Path, default=DEFAULT_TRAIN_OMNI_ROOT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--device", default="auto")
    p.add_argument("--max-samples", type=int, default=48)
    p.add_argument("--spacing-minutes", type=int, default=360)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--permutation-seed", type=int, default=202603)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--inference-steps", type=int, default=300)
    p.add_argument("--jump-length", type=int, default=10)
    p.add_argument("--jump-repeats", type=int, default=10)
    p.add_argument("--linearization-eps", type=float, default=1e-3)
    p.add_argument("--skip-linearized", action="store_true")
    return p.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if hasattr(torch, "npu"):
        try:
            if torch.npu.is_available():
                return torch.device("npu:0")
        except Exception:
            pass
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "npu" and hasattr(torch, "npu"):
        torch.npu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def set_all_seeds(seed: int, device: torch.device) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if device.type == "npu" and hasattr(torch, "npu"):
        try:
            torch.npu.manual_seed_all(int(seed))
        except Exception:
            pass
    elif device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))


def load_norm_params(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Normalization parameters not found: {path}")
    with path.open("rb") as f:
        norm = pickle.load(f)
    if "aurora" not in norm or "omni" not in norm:
        raise KeyError(f"Unexpected normalization dictionary keys: {norm.keys()}")
    return norm


def normalize_aurora(flux: np.ndarray, norm: Dict[str, Any]) -> np.ndarray:
    x = np.clip(np.asarray(flux, dtype=np.float32), 0.0, None)
    log_x = np.log1p(x)
    y = (log_x - float(norm["aurora"]["min"])) / float(norm["aurora"]["range"])
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def denormalize_aurora(y: np.ndarray, norm: Dict[str, Any]) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    log_x = y * float(norm["aurora"]["range"]) + float(norm["aurora"]["min"])
    return np.clip(np.expm1(log_x), 0.0, None).astype(np.float32)


def normalize_conditions(raw: np.ndarray, norm: Dict[str, Any]) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float32)
    minimum = np.asarray(norm["omni"]["min"], dtype=np.float32).reshape(-1)
    value_range = np.asarray(norm["omni"]["range"], dtype=np.float32).reshape(-1)
    if minimum.size != 5 or value_range.size != 5:
        raise ValueError(
            f"Paper normalization must contain five OMNI dimensions; got {minimum.size}"
        )
    return np.clip((raw - minimum) / value_range, 0.0, 1.0).astype(np.float32)


def _scalar_field(arr: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(arr[name])
    if values.ndim == 1:
        return values.astype(np.float32)
    return values.reshape(values.shape[0], -1)[:, 0].astype(np.float32)


def _first_existing_field(names: Sequence[str], candidates: Iterable[str]) -> str | None:
    for field in candidates:
        if field in names:
            return field
    return None


def extract_conditions(omni: np.ndarray) -> np.ndarray:
    """Extract [Bx, By, Bz, Vsw, Pdyn] from old or new OMNI files."""
    names = omni.dtype.names
    if names is None:
        raise TypeError("OMNI input must be a structured NumPy array")

    bx = _first_existing_field(names, ("Bx", "BX_GSE", "BX_GSM"))
    by = _first_existing_field(names, ("By", "BY_GSM", "BY_GSE"))
    bz = _first_existing_field(names, ("Bz", "BZ_GSM", "BZ_GSE"))
    pdyn = _first_existing_field(names, ("P", "Pressure", "PRESSURE", "Pdyn"))
    missing = [label for label, field in (("Bx", bx), ("By", by), ("Bz", bz), ("Pdyn", pdyn)) if field is None]
    if missing:
        raise KeyError(f"Missing required OMNI fields {missing}; available={names}")

    speed_field = _first_existing_field(names, ("V", "flow_speed", "FLOW_SPEED"))
    if speed_field is not None:
        speed = _scalar_field(omni, speed_field)
    else:
        vx = _first_existing_field(names, ("Vx", "VX_GSE", "VX_GSM"))
        vy = _first_existing_field(names, ("Vy", "VY_GSE", "VY_GSM"))
        vz = _first_existing_field(names, ("Vz", "VZ_GSE", "VZ_GSM"))
        if vx is None or vy is None or vz is None:
            raise KeyError(f"No usable solar-wind speed fields; available={names}")
        vxv = _scalar_field(omni, vx)
        vyv = _scalar_field(omni, vy)
        vzv = _scalar_field(omni, vz)
        speed = np.sqrt(vxv * vxv + vyv * vyv + vzv * vzv).astype(np.float32)

    return np.column_stack(
        [
            _scalar_field(omni, bx),
            _scalar_field(omni, by),
            _scalar_field(omni, bz),
            speed,
            _scalar_field(omni, pdyn),
        ]
    ).astype(np.float32)


def condition_is_valid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return (
        np.all(np.isfinite(x), axis=1)
        & np.all(np.abs(x[:, :3]) < 200.0, axis=1)
        & (x[:, 3] > 100.0)
        & (x[:, 3] < 2500.0)
        & (x[:, 4] >= 0.0)
        & (x[:, 4] < 100.0)
    )


def extract_timestamps(omni: np.ndarray) -> np.ndarray:
    names = omni.dtype.names or ()
    field = _first_existing_field(names, ("utc", "Epoch", "epoch"))
    if field is None:
        return np.asarray([str(i) for i in range(len(omni))], dtype=object)
    return np.asarray(omni[field])


def evenly_select(candidates: np.ndarray, max_samples: int) -> np.ndarray:
    candidates = np.asarray(candidates, dtype=int)
    if candidates.size < 2:
        raise RuntimeError("Fewer than two valid held-out samples are available")
    if max_samples <= 0 or max_samples >= candidates.size:
        return candidates
    pos = np.rint(np.linspace(0, candidates.size - 1, int(max_samples))).astype(int)
    return candidates[np.unique(pos)]


def load_paper_model(checkpoint_path: Path, device: torch.device) -> DDPM:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    ddpm = DDPM(UNet(1, 1), num_train_steps=1000, schedule="cosine")
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    incompatible = ddpm.load_state_dict(state, strict=False)
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    if missing or unexpected:
        print(f"Checkpoint load note: missing={len(missing)}, unexpected={len(unexpected)}")
        if missing:
            print("  first missing keys:", missing[:8])
        if unexpected:
            print("  first unexpected keys:", unexpected[:8])
    ddpm.eval().to(device)
    return ddpm


class LinearizedSolarWindEncoder(nn.Module):
    """Finite-difference first-order linearization of the trained encoder."""

    def __init__(self, original: nn.Module, center: torch.Tensor, eps: float = 1e-3) -> None:
        super().__init__()
        if center.ndim != 1 or center.numel() != 5:
            raise ValueError(f"center must have five elements; got {tuple(center.shape)}")
        if eps <= 0:
            raise ValueError("eps must be positive")
        original.eval()
        center = center.detach().clone()
        with torch.no_grad():
            f0 = original(center.unsqueeze(0)).squeeze(0)
            columns = []
            for j in range(5):
                plus = center.clone()
                minus = center.clone()
                plus[j] = torch.clamp(plus[j] + eps, 0.0, 1.0)
                minus[j] = torch.clamp(minus[j] - eps, 0.0, 1.0)
                denom = float((plus[j] - minus[j]).item())
                if denom <= 0:
                    columns.append(torch.zeros_like(f0))
                else:
                    fp = original(plus.unsqueeze(0)).squeeze(0)
                    fm = original(minus.unsqueeze(0)).squeeze(0)
                    columns.append((fp - fm) / denom)
            jac = torch.stack(columns, dim=1)
        self.register_buffer("center", center)
        self.register_buffer("f0", f0)
        self.register_buffer("jac", jac)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.f0.unsqueeze(0) + (x - self.center.unsqueeze(0)) @ self.jac.transpose(0, 1)


def training_condition_center(root: Path, norm: Dict[str, Any]) -> Tuple[np.ndarray, str, int]:
    files = [root / "omni_19960401_5min.npy", root / "omni_19960501_5min.npy"]
    chunks: List[np.ndarray] = []
    used: List[str] = []
    for path in files:
        if not path.exists():
            continue
        raw = extract_conditions(np.load(path, allow_pickle=True))
        raw = raw[condition_is_valid(raw)]
        if len(raw):
            chunks.append(normalize_conditions(raw, norm))
            used.append(path.name)
    if not chunks:
        raise FileNotFoundError("1996 Apr--May training OMNI files were not found")
    merged = np.concatenate(chunks, axis=0)
    return np.mean(merged, axis=0).astype(np.float32), ",".join(used), int(len(merged))


def reconstruct_one(
    ddpm: DDPM,
    truth_flux: np.ndarray,
    mask: np.ndarray,
    condition_norm: np.ndarray,
    norm: Dict[str, Any],
    device: torch.device,
    *,
    seed: int,
    inference_steps: int,
    jump_length: int,
    jump_repeats: int,
) -> np.ndarray:
    corrupted = np.asarray(truth_flux, dtype=np.float32).copy()
    corrupted[mask == 0] = 0.0
    image_norm = normalize_aurora(corrupted, norm)
    image_t = torch.from_numpy(image_norm[None, None]).float().to(device)
    mask_t = torch.from_numpy(mask[None, None].astype(np.float32)).to(device)
    cond_t = torch.from_numpy(np.asarray(condition_norm, dtype=np.float32)[None]).to(device)

    set_all_seeds(seed, device)
    synchronize(device)
    with torch.no_grad():
        out = ddpm.sample(
            image_t,
            mask_t,
            cond_t,
            num_inference_steps=int(inference_steps),
            n_sample=1,
            j=int(jump_length),
            r=int(jump_repeats),
        )
    synchronize(device)
    return denormalize_aurora(out.detach().cpu().numpy().squeeze(), norm)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, np.ndarray):
        return [json_safe(v) for v in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def paired_summary(
    rows: List[Dict[str, Any]],
    variants: Sequence[str],
    bootstrap_n: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    grouped: Dict[str, Dict[int, Dict[str, Any]]] = {v: {} for v in variants}
    for row in rows:
        grouped[str(row["variant"])][int(row["sample_id"])] = row
    full = grouped["full"]
    summary_rows: List[Dict[str, Any]] = []
    paired_detail: Dict[str, Any] = {}

    for vi, variant in enumerate(variants):
        sample_ids = sorted(grouped[variant])
        record: Dict[str, Any] = {"variant": variant, "n": len(sample_ids)}
        paired_detail[variant] = {}
        for metric in METRICS:
            vals = np.asarray([float(grouped[variant][i][metric]) for i in sample_ids], dtype=float)
            vals = vals[np.isfinite(vals)]
            record[f"mean_{metric}"] = float(np.mean(vals)) if len(vals) else float("nan")
            if variant == "full":
                record[f"degradation_{metric}"] = 0.0
                record[f"degradation_{metric}_ci_low"] = 0.0
                record[f"degradation_{metric}_ci_high"] = 0.0
                record[f"degradation_{metric}_p"] = 1.0
                paired_detail[variant][metric] = {"paired_degradation": [0.0] * len(sample_ids)}
                continue

            paired_ids = sorted(set(sample_ids) & set(full))
            deltas = np.asarray(
                [metric_degradation(full[i], grouped[variant][i])[metric] for i in paired_ids],
                dtype=float,
            )
            ci = bootstrap_mean_ci(
                deltas,
                n_boot=int(bootstrap_n),
                seed=int(seed + 1000 * vi + METRICS.index(metric)),
            )
            pvalue = paired_wilcoxon_pvalue(deltas)
            record[f"degradation_{metric}"] = ci["mean"]
            record[f"degradation_{metric}_ci_low"] = ci["ci_low"]
            record[f"degradation_{metric}_ci_high"] = ci["ci_high"]
            record[f"degradation_{metric}_p"] = pvalue
            paired_detail[variant][metric] = {
                "paired_degradation": deltas.tolist(),
                "bootstrap": ci,
                "wilcoxon_two_sided_p": pvalue,
            }
        summary_rows.append(record)
    return summary_rows, paired_detail


def variant_label(variant: str) -> str:
    return {
        "full": "Full condition",
        "permute_Bx": "Permute Bx",
        "permute_By": "Permute By",
        "permute_Bz": "Permute Bz",
        "permute_Vsw": "Permute Vsw",
        "permute_Pdyn": "Permute Pdyn",
        "permute_all": "Permute all",
        "linearized_encoder": "Linearized encoder",
    }.get(variant, variant)


def write_latex_table(path: Path, rows: List[Dict[str, Any]]) -> None:
    endrow = r"\\"
    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\caption{Conditional-variable ablation on held-out January 2005 OVATION Prime samples. Each permutation preserves the marginal distribution of the selected solar-wind variable while breaking its correspondence with the target auroral map. Positive $\Delta$RMSE denotes degradation relative to the full five-variable condition. Confidence intervals are paired-bootstrap 95\% intervals.}",
        r"\label{tab:condition_ablation}",
        r"\begin{tabular}{lrrrr}",
        r"\hline",
        "Condition & SSIM & RMSE & $R^2$ & $\\Delta$RMSE (95\\% CI) " + endrow,
        r"\hline",
    ]
    for row in rows:
        lines.append(
            f"{variant_label(str(row['variant']))} & "
            f"{float(row['mean_ssim']):.3f} & {float(row['mean_rmse']):.3f} & "
            f"{float(row['mean_r2']):.3f} & {float(row['degradation_rmse']):+.3f} "
            f"[{float(row['degradation_rmse_ci_low']):+.3f}, {float(row['degradation_rmse_ci_high']):+.3f}] "
            + endrow
        )
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table*}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_importance(path: Path, rows: List[Dict[str, Any]]) -> None:
    rows = [r for r in rows if r["variant"] != "full"]
    labels = [variant_label(str(r["variant"])) for r in rows]
    means = np.asarray([float(r["degradation_rmse"]) for r in rows])
    lows = np.asarray([float(r["degradation_rmse_ci_low"]) for r in rows])
    highs = np.asarray([float(r["degradation_rmse_ci_high"]) for r in rows])
    yerr = np.vstack([means - lows, highs - means])
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(rows))
    ax.bar(x, means, yerr=yerr, capsize=4)
    ax.axhline(0.0, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Paired RMSE degradation (Delta RMSE)")
    ax.set_title("Solar-wind condition importance")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def timestamp_text(value: Any) -> str:
    if isinstance(value, np.datetime64):
        return str(value.astype("datetime64[s]"))
    return str(value)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)

    print("=" * 92)
    print("Reviewer 1 Comment 3 -- CONDITIONAL VARIABLE ABLATION")
    print("=" * 92)
    print(f"Device:          {device}")
    print(f"Checkpoint:      {args.checkpoint}")
    print(f"OVATION:         {args.ovation}")
    print(f"OMNI:            {args.omni}")
    print(f"Norm params:     {args.norm_params}")
    print(f"Output:          {args.output_dir}")
    print(f"Max samples:     {args.max_samples}")
    print(f"Spacing:         {args.spacing_minutes} min")
    print(f"Linearized test: {'disabled' if args.skip_linearized else 'enabled'}")

    for required in (args.ovation, args.omni, args.checkpoint, args.norm_params):
        if not required.exists():
            raise FileNotFoundError(required)

    norm = load_norm_params(args.norm_params)
    ovation = np.load(args.ovation, allow_pickle=False).astype(np.float32)
    omni = np.load(args.omni, allow_pickle=True)
    if ovation.ndim != 3:
        raise ValueError(f"Expected OVATION array (N,H,W), got {ovation.shape}")
    if len(ovation) != len(omni):
        raise ValueError(
            f"OVATION and 1-min OMNI lengths differ: {len(ovation)} vs {len(omni)}. "
            "Refusing implicit index alignment."
        )

    all_conditions = extract_conditions(omni)
    valid = condition_is_valid(all_conditions)
    spacing = int(args.spacing_minutes)
    if spacing <= 0:
        raise ValueError("--spacing-minutes must be positive")
    candidates = np.arange(0, len(ovation), spacing, dtype=int)
    candidates = candidates[valid[candidates]]
    if len(candidates) == 0:
        raise RuntimeError("No valid 6-h candidates remain after OMNI screening")
    finite_maps = np.all(np.isfinite(ovation[candidates].reshape(len(candidates), -1)), axis=1)
    candidates = candidates[finite_maps]
    selected = evenly_select(candidates, int(args.max_samples))

    timestamps = extract_timestamps(omni)
    selected_raw = all_conditions[selected]
    permutation = make_derangement(len(selected), seed=int(args.permutation_seed))
    variant_raw = {v: apply_condition_variant(selected_raw, v, permutation) for v in BASE_VARIANTS}
    variant_norm = {v: normalize_conditions(x, norm) for v, x in variant_raw.items()}

    print(f"Valid candidates:  {len(candidates)}")
    print(f"Selected samples:  {len(selected)}")
    print(f"Selected range:    {int(selected[0])} -> {int(selected[-1])}")

    ddpm = load_paper_model(args.checkpoint, device)
    original_encoder = ddpm.generator.solar_wind_encoder
    linearized_encoder = None
    center_info: Dict[str, Any] = {"enabled": not args.skip_linearized}
    if not args.skip_linearized:
        try:
            center_np, center_source, center_n = training_condition_center(args.training_omni_root, norm)
        except Exception as exc:
            print(f"WARNING: training mean unavailable ({exc}); using selected-test mean fallback.")
            center_np = np.mean(variant_norm["full"], axis=0).astype(np.float32)
            center_source = "selected-test mean fallback"
            center_n = len(selected)
        center_t = torch.from_numpy(center_np).float().to(device)
        linearized_encoder = LinearizedSolarWindEncoder(
            original_encoder, center_t, eps=float(args.linearization_eps)
        ).to(device)
        center_info.update(
            {
                "source": center_source,
                "n_conditions": int(center_n),
                "normalized_center": center_np.tolist(),
                "finite_difference_eps": float(args.linearization_eps),
            }
        )
        print(f"Linearization center: {center_source} (N={center_n})")

    variants = list(BASE_VARIANTS)
    if linearized_encoder is not None:
        variants.append("linearized_encoder")

    mask = create_paper_mask(tuple(ovation.shape[1:]), mlat_range=(60.0, 80.0))
    rows: List[Dict[str, Any]] = []
    wall_start = time.time()

    for local_i, data_idx in enumerate(selected):
        truth = ovation[data_idx]
        sample_seed = int(args.seed + local_i)
        print("-" * 92)
        print(
            f"Sample {local_i + 1:02d}/{len(selected):02d} | index={int(data_idx)} | "
            f"time={timestamp_text(timestamps[data_idx])}"
        )
        for variant in variants:
            if variant == "linearized_encoder":
                ddpm.generator.solar_wind_encoder = linearized_encoder
                cond_norm = variant_norm["full"][local_i]
                cond_raw = selected_raw[local_i]
                perm_source = local_i
            else:
                ddpm.generator.solar_wind_encoder = original_encoder
                cond_norm = variant_norm[variant][local_i]
                cond_raw = variant_raw[variant][local_i]
                perm_source = int(permutation[local_i]) if variant != "full" else local_i

            run_start = time.time()
            pred = reconstruct_one(
                ddpm,
                truth,
                mask,
                cond_norm,
                norm,
                device,
                seed=sample_seed,
                inference_steps=int(args.inference_steps),
                jump_length=int(args.jump_length),
                jump_repeats=int(args.jump_repeats),
            )
            elapsed = time.time() - run_start
            metrics = compute_masked_metrics(truth, pred, mask)
            row: Dict[str, Any] = {
                "sample_id": int(local_i),
                "data_index": int(data_idx),
                "timestamp": timestamp_text(timestamps[data_idx]),
                "variant": variant,
                "permutation_source_sample": int(perm_source),
                "Bx": float(cond_raw[0]),
                "By": float(cond_raw[1]),
                "Bz": float(cond_raw[2]),
                "Vsw": float(cond_raw[3]),
                "Pdyn": float(cond_raw[4]),
                "runtime_s": float(elapsed),
            }
            row.update(metrics)
            rows.append(row)
            print(
                f"  {variant:18s} RMSE={metrics['rmse']:.4f} "
                f"SSIM={metrics['ssim']:.4f} R2={metrics['r2']:.4f} ({elapsed:.1f}s)"
            )

    ddpm.generator.solar_wind_encoder = original_encoder

    sample_csv = args.output_dir / "condition_ablation_samples.csv"
    write_csv(sample_csv, rows)
    summary_rows, paired_detail = paired_summary(
        rows, variants, bootstrap_n=int(args.bootstrap), seed=int(args.seed)
    )
    summary_csv = args.output_dir / "condition_ablation_summary.csv"
    write_csv(summary_csv, summary_rows)

    summary_json = args.output_dir / "condition_ablation_summary.json"
    summary_json.write_text(
        json.dumps(
            json_safe(
                {
                    "variants": variants,
                    "summary": summary_rows,
                    "paired_statistics": paired_detail,
                }
            ),
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    latex_path = args.output_dir / "condition_ablation_table.tex"
    write_latex_table(latex_path, summary_rows)
    plot_path = args.output_dir / "condition_importance.png"
    plot_importance(plot_path, summary_rows)

    metadata = {
        "checkpoint": args.checkpoint,
        "ovation": args.ovation,
        "omni": args.omni,
        "norm_params": args.norm_params,
        "device": str(device),
        "sample_spacing_minutes": int(args.spacing_minutes),
        "selected_indices": selected,
        "selected_timestamps": [timestamp_text(timestamps[i]) for i in selected],
        "permutation": permutation,
        "seed": int(args.seed),
        "permutation_seed": int(args.permutation_seed),
        "mask_missing_pixels": int(np.sum(mask == 0)),
        "inference_steps": int(args.inference_steps),
        "jump_length": int(args.jump_length),
        "jump_repeats": int(args.jump_repeats),
        "linearized_encoder": center_info,
        "wall_time_s": float(time.time() - wall_start),
    }
    metadata_path = args.output_dir / "experiment_metadata.json"
    metadata_path.write_text(
        json.dumps(json_safe(metadata), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("=" * 92)
    print("R1.3 SUMMARY (positive degradation = worse than full condition)")
    print("=" * 92)
    for row in summary_rows:
        print(
            f"{row['variant']:18s} | RMSE={float(row['mean_rmse']):.4f} | "
            f"dRMSE={float(row['degradation_rmse']):+.4f} "
            f"[{float(row['degradation_rmse_ci_low']):+.4f}, "
            f"{float(row['degradation_rmse_ci_high']):+.4f}] | "
            f"p={float(row['degradation_rmse_p']):.4g}"
        )
    print("=" * 92)
    print(f"Per-sample CSV: {sample_csv}")
    print(f"Summary CSV:    {summary_csv}")
    print(f"Summary JSON:   {summary_json}")
    print(f"LaTeX table:    {latex_path}")
    print(f"Importance fig: {plot_path}")
    print(f"Metadata:       {metadata_path}")
    print(f"Wall time:      {(time.time() - wall_start) / 60.0:.1f} min")


if __name__ == "__main__":
    main()
