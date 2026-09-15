#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reviewer 1 Comment 3: conditional-variable ablation experiment.

The experiment keeps the final paper checkpoint fixed and measures how much
reconstruction skill is lost when the correspondence between an individual
solar-wind variable and the target auroral map is broken by permutation.

Key design choices
------------------
1. Uses the paper model in models/unet.py and the final conditional checkpoint.
2. Uses held-out January 2005 OVATION maps, sampled at 6 h cadence by default.
3. Uses the same artificial 60--80 MLAT, 18--06 MLT mask as the manuscript's
   controlled OVATION evaluation.
4. Uses the same mask and identical diffusion random seed for every ablation
   variant of a given sample, so comparisons are paired.
5. Permutation preserves each variable's marginal distribution while breaking
   its event-to-event correspondence with the auroral target.
6. Reports paired bootstrap 95% confidence intervals and paired Wilcoxon tests.
7. Optionally compares the trained nonlinear solar-wind MLP with a first-order
   finite-difference linearization around the mean normalized 1996 training
   condition.

Run from repository root:
    python paper_modif/R1-3/condition_ablation.py

Recommended smoke test before the full run:
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
for p in (str(REPO_ROOT), str(SCRIPT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

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
    if device.type == "cuda":
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
    y = (raw - minimum) / value_range
    return np.clip(y, 0.0, 1.0).astype(np.float32)


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
    """Extract [Bx, By, Bz, Vsw, Pdyn] from old or new OMNI structured arrays."""
    names = omni.dtype.names
    if names is None:
        raise TypeError("OMNI input must be a structured NumPy array")

    bx_name = _first_existing_field(names, ("Bx", "BX_GSE", "BX_GSM"))
    by_name = _first_existing_field(names, ("By", "BY_GSM", "BY_GSE"))
    bz_name = _first_existing_field(names, ("Bz", "BZ_GSM", "BZ_GSE"))
    p_name = _first_existing_field(names, ("P", "Pressure", "PRESSURE", "Pdyn"))
    missing = [
        label
        for label, value in (("Bx", bx_name), ("By", by_name), ("Bz", bz_name), ("P", p_name))
        if value is None
    ]
    if missing:
        raise KeyError(f"Missing required OMNI fields {missing}; available={names}")

    v_name = _first_existing_field(names, ("V", "flow_speed", "FLOW_SPEED"))
    if v_name is not None:
        speed = _scalar_field(omni, v_name)
    else:
        component_names = [
            _first_existing_field(names, ("Vx", "VX_GSE", "VX_GSM")),
            _first_existing_field(names, ("Vy", "VY_GSE", "VY_GSM")),
            _first_existing_field(names, ("Vz", "VZ_GSE", "VZ_GSM")),
        ]
        if any(name is None for name in component_names):
            raise KeyError(
                "No scalar solar-wind speed and incomplete velocity components; "
                f"available={names}"
            )
        vx, vy, vz = (_scalar_field(omni, name) for name in component_names)
        speed = np.sqrt(vx * vx + vy * vy + vz * vz).astype(np.float32)

    return np.column_stack(
        [
            _scalar_field(omni, bx_name),
            _scalar_field(omni, by_name),
            _scalar_field(omni, bz_name),
            speed,
            _scalar_field(omni, p_name),
        ]
    ).astype(np.float32)


def condition_is_valid(x: np.ndarray) -> np.ndarray:
    """Conservative physical screening for selected 1-min test conditions."""
    x = np.asarray(x, dtype=np.float64)
    finite = np.all(np.isfinite(x), axis=1)
    magnetic = np.all(np.abs(x[:, :3]) < 200.0, axis=1)
    speed = (x[:, 3] > 100.0) & (x[:, 3] < 2500.0)
    pressure = (x[:, 4] >= 0.0) & (x[:, 4] < 100.0)
    return finite & magnetic & speed & pressure


def extract_timestamps(omni: np.ndarray) -> np.ndarray:
    names = omni.dtype.names or ()
    field = _first_existing_field(names, ("utc", "Epoch", "epoch"))
    if field is None:
        return np.array([str(i) for i in range(len(omni))], dtype=object)
    return np.asarray(omni[field])


def evenly_select_candidates(candidates: np.ndarray, max_samples: int) -> np.ndarray:
    candidates = np.asarray(candidates, dtype=int)
    if candidates.size < 2:
        raise RuntimeError("Fewer than two valid held-out samples are available")
    if max_samples <= 0 or max_samples >= candidates.size:
        return candidates
    pos = np.rint(np.linspace(0, candidates.size - 1, int(max_samples))).astype(int)
    pos = np.unique(pos)
    return candidates[pos]


def load_paper_model(checkpoint_path: Path, device: torch.device) -> DDPM:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    unet = UNet(1, 1)
    ddpm = DDPM(unet, num_train_steps=1000, schedule="cosine")
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
    ddpm.eval()
    ddpm.to(device)
    return ddpm


class LinearizedSolarWindEncoder(nn.Module):
    """First-order finite-difference linearization of a trained encoder."""

    def __init__(
        self,
        original: nn.Module,
        center: torch.Tensor,
        eps: float = 1e-3,
    ) -> None:
        super().__init__()
        if center.ndim != 1 or center.numel() != 5:
            raise ValueError(f"center must be a five-component vector; got {tuple(center.shape)}")
        if eps <= 0:
            raise ValueError("eps must be positive")

        original.eval()
        center = center.detach().clone()
        with torch.no_grad():
            f0 = original(center.unsqueeze(0)).squeeze(0)
            columns = []
            for j in range(center.numel()):
                plus = center.clone()
                minus = center.clone()
                plus[j] = torch.clamp(plus[j] + eps, 0.0, 1.0)
                minus[j] = torch.clamp(minus[j] - eps, 0.0, 1.0)
                denom = float((plus[j] - minus[j]).item())
                if denom <= 0:
                    columns.append(torch.zeros_like(f0))
                    continue
                fp = original(plus.unsqueeze(0)).squeeze(0)
                fm = original(minus.unsqueeze(0)).squeeze(0)
                columns.append((fp - fm) / denom)
            jac = torch.stack(columns, dim=1)  # output_dim x 5

        self.register_buffer("center", center)
        self.register_buffer("f0", f0)
        self.register_buffer("jac", jac)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = x - self.center.unsqueeze(0)
        return self.f0.unsqueeze(0) + delta @ self.jac.transpose(0, 1)


def training_condition_center(
    root: Path,
    norm: Dict[str, Any],
) -> Tuple[np.ndarray, str, int]:
    """Compute the normalized mean of the 1996 Apr--May training conditions."""
    candidates = [
        root / "omni_19960401_5min.npy",
        root / "omni_19960501_5min.npy",
    ]
    chunks: List[np.ndarray] = []
    used: List[str] = []
    for path in candidates:
        if not path.exists():
            continue
        arr = np.load(path, allow_pickle=True)
        raw = extract_conditions(arr)
        raw = raw[condition_is_valid(raw)]
        if raw.size:
            chunks.append(normalize_conditions(raw, norm))
            used.append(path.name)
    if not chunks:
        raise FileNotFoundError(
            "Could not compute the 1996 training-condition mean; expected files: "
            + ", ".join(str(p) for p in candidates)
        )
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
    out_np = out.detach().cpu().numpy().squeeze()
    return denormalize_aurora(out_np, norm)


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


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def paired_summary(
    rows: List[Dict[str, Any]],
    variants: Sequence[str],
    *,
    bootstrap_n: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    by_variant: Dict[str, Dict[int, Dict[str, Any]]] = {v: {} for v in variants}
    for row in rows:
        by_variant[row["variant"]][int(row["sample_id"])] = row

    full = by_variant["full"]
    summary_rows: List[Dict[str, Any]] = []
    detail: Dict[str, Any] = {}

    for vi, variant in enumerate(variants):
        sample_ids = sorted(by_variant[variant])
        record: Dict[str, Any] = {"variant": variant, "n": len(sample_ids)}
        detail[variant] = {}
        for metric in METRICS:
            vals = np.array([float(by_variant[variant][i][metric]) for i in sample_ids], dtype=float)
            finite_vals = vals[np.isfinite(vals)]
            record[f"mean_{metric}"] = float(np.mean(finite_vals)) if finite_vals.size else float("nan")

            if variant == "full":
                record[f"degradation_{metric}"] = 0.0
                record[f"degradation_{metric}_ci_low"] = 0.0
                record[f"degradation_{metric}_ci_high"] = 0.0
                record[f"degradation_{metric}_p"] = 1.0
                detail[variant][metric] = {"paired_degradation": [0.0] * len(sample_ids)}
                continue

            paired_ids = sorted(set(sample_ids) & set(full))
            deltas = []
            for i in paired_ids:
                d = metric_degradation(full[i], by_variant[variant][i])[metric]
                deltas.append(float(d))
            delta_arr = np.asarray(deltas, dtype=float)
            ci = bootstrap_mean_ci(
                delta_arr,
                n_boot=bootstrap_n,
                seed=int(seed + 1000 * vi + METRICS.index(metric)),
            )
            pvalue = paired_wilcoxon_pvalue(delta_arr)
            record[f"degradation_{metric}"] = ci["mean"]
            record[f"degradation_{metric}_ci_low"] = ci["ci_low"]
            record[f"degradation_{metric}_ci_high"] = ci["ci_high"]
            record[f"degradation_{metric}_p"] = pvalue
            detail[variant][metric] = {
                "paired_degradation": deltas,
                "bootstrap": ci,
                "wilcoxon_two_sided_p": pvalue,
            }
        summary_rows.append(record)
    return summary_rows, detail


def variant_label(variant: str) -> str:
    labels = {
        "full": "Full condition",
        "permute_Bx": r"Permute $B_x$",
        "permute_By": r"Permute $B_y$",
        "permute_Bz": r"Permute $B_z$",
        "permute_Vsw": r"Permute $V_{sw}$",
        "permute_Pdyn": r"Permute $P_{dyn}$",
        "permute_all": "Permute all",
        "linearized_encoder": "Linearized encoder",
    }
    return labels.get(variant, variant)


def write_latex_table(path: Path, summary_rows: List[Dict[str, Any]]) -> None:
    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\caption{Conditional-variable ablation on held-out January 2005 OVATION Prime samples. "
        r"Each permutation preserves the marginal distribution of the selected solar-wind variable while "
        r"breaking its correspondence with the target auroral map. Positive $\Delta$RMSE denotes degradation "
        r"relative to the full five-variable condition. Confidence intervals are paired-bootstrap 95\% intervals.}",
        r"\label{tab:condition_ablation}",
        r"\begin{tabular}{lrrrr}",
        r"\hline",
        r"Condition & SSIM & RMSE & $R^2$ & $\Delta$RMSE (95\% CI) \\",
        r"\hline",
    ]
    for row in summary_rows:
        delta = float(row["degradation_rmse"])
        low = float(row["degradation_rmse_ci_low"])
        high = float(row["degradation_rmse_ci_high"])
        lines.append(
            f"{variant_label(str(row['variant']))} & "
            f"{float(row['mean_ssim']):.3f} & {float(row['mean_rmse']):.3f} & "
            f"{float(row['mean_r2']):.3f} & {delta:+.3f} [{low:+.3f}, {high:+.3f}] \\\\" 
        )
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table*}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_importance(path: Path, summary_rows: List[Dict[str, Any]]) -> None:
    rows = [r for r in summary_rows if r["variant"] != "full"]
    labels = [variant_label(str(r["variant"])).replace("$", "") for r in rows]
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
    ax.set_ylabel(r"Paired RMSE degradation ($\Delta$RMSE)")
    ax.set_title("Solar-wind condition importance")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def timestamp_to_text(value: Any) -> str:
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

    if not args.ovation.exists():
        raise FileNotFoundError(args.ovation)
    if not args.omni.exists():
        raise FileNotFoundError(args.omni)

    norm = load_norm_params(args.norm_params)
    ovation = np.load(args.ovation, allow_pickle=False).astype(np.float32)
    omni = np.load(args.omni, allow_pickle=True)
    if ovation.ndim != 3:
        raise ValueError(f"Expected OVATION array (N,H,W), got {ovation.shape}")
    if len(ovation) != len(omni):
        raise ValueError(
            "OVATION and 1-min OMNI lengths differ; refusing index-based alignment: "
            f"{len(ovation)} vs {len(omni)}"
        )

    all_conditions = extract_conditions(omni)
    valid = condition_is_valid(all_conditions)
    spacing = int(args.spacing_minutes)
    if spacing <= 0:
        raise ValueError("--spacing-minutes must be positive")
    candidates = np.arange(0, len(ovation), spacing, dtype=int)
    candidates = candidates[valid[candidates]]
    candidates = candidates[np.all(np.isfinite(ovation[candidates].reshape(len(candidates), -1)), axis=1)]
    selected = evenly_select_candidates(candidates, int(args.max_samples))
    if len(selected) < 2:
        raise RuntimeError("Need at least two selected samples for permutation ablation")

    timestamps = extract_timestamps(omni)
    selected_raw = all_conditions[selected]
    permutation = make_derangement(len(selected), seed=args.permutation_seed)
    variant_raw = {v: apply_condition_variant(selected_raw, v, permutation) for v in BASE_VARIANTS}
    variant_norm = {v: normalize_conditions(x, norm) for v, x in variant_raw.items()}

    print(f"Valid 6-h candidates: {len(candidates)}")
    print(f"Selected samples:     {len(selected)}")
    print("Selected index range: ", int(selected[0]), "->", int(selected[-1]))

    ddpm = load_paper_model(args.checkpoint, device)
    original_encoder = ddpm.generator.solar_wind_encoder

    linearized_encoder = None
    center_info: Dict[str, Any] = {"enabled": not args.skip_linearized}
    if not args.skip_linearized:
        try:
            center_np, center_source, center_n = training_condition_center(args.training_omni_root, norm)
        except Exception as exc:
            print(f"WARNING: training-condition mean unavailable ({exc}); using selected-test mean.")
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
    t0 = time.time()

    for local_i, data_idx in enumerate(selected):
        truth = ovation[data_idx]
        sample_seed = int(args.seed + local_i)
        print("-" * 92)
        print(
            f"Sample {local_i + 1:02d}/{len(selected):02d} | index={int(data_idx)} | "
            f"time={timestamp_to_text(timestamps[data_idx])}"
        )

        for variant in variants:
            if variant == "linearized_encoder":
                ddpm.generator.solar_wind_encoder = linearized_encoder
                cond_norm = variant_norm["full"][local_i]
                cond_raw = selected_raw[local_i]
                source_local = local_i
            else:
                ddpm.generator.solar_wind_encoder = original_encoder
                cond_norm = variant_norm[variant][local_i]
                cond_raw = variant_raw[variant][local_i]
                source_local = int(permutation[local_i]) if variant != "full" else local_i

            start = time.time()
            pred = reconstruct_one(
                ddpm,
                truth,
                mask,
                cond_norm,
                norm,
                device,
                seed=sample_seed,
                inference_steps=args.inference_steps,
                jump_length=args.jump_length,
                jump_repeats=args.jump_repeats,
            )
            elapsed = time.time() - start
            metrics = compute_masked_metrics(truth, pred, mask)

            row: Dict[str, Any] = {
                "sample_id": int(local_i),
                "data_index": int(data_idx),
                "timestamp": timestamp_to_text(timestamps[data_idx]),
                "variant": variant,
                "condition_source_sample": int(source_local),
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
                f"SSIM={metrics['ssim']:.4f} R2={metrics['r2']:.4f} "
                f"({elapsed:.1f}s)"
            )

    ddpm.generator.solar_wind_encoder = original_encoder

    sample_csv = args.output_dir / "condition_ablation_samples.csv"
    write_csv(sample_csv, rows)

    summary_rows, paired_detail = paired_summary(
        rows,
        variants,
        bootstrap_n=int(args.bootstrap),
        seed=int(args.seed),
    )
    summary_csv = args.output_dir / "condition_ablation_summary.csv"
    write_csv(summary_csv, summary_rows)

    summary_json = args.output_dir / "condition_ablation_summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(
            json_safe(
                {
                    "variants": variants,
                    "summary": summary_rows,
                    "paired_statistics": paired_detail,
                }
            ),
            f,
            indent=2,
            ensure_ascii=False,
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
        "max_samples": int(args.max_samples),
        "selected_indices": selected,
        "selected_timestamps": [timestamp_to_text(timestamps[i]) for i in selected],
        "permutation": permutation,
        "seed": int(args.seed),
        "permutation_seed": int(args.permutation_seed),
        "mask_missing_pixels": int(np.sum(mask == 0)),
        "inference_steps": int(args.inference_steps),
        "jump_length": int(args.jump_length),
        "jump_repeats": int(args.jump_repeats),
        "linearized_encoder": center_info,
        "wall_time_s": float(time.time() - t0),
    }
    metadata_path = args.output_dir / "experiment_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(metadata), f, indent=2, ensure_ascii=False)

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
    print(f"Wall time:      {(time.time() - t0) / 60.0:.1f} min")


if __name__ == "__main__":
    main()
