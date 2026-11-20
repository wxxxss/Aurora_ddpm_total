#!/usr/bin/env python3
"""Held-out real-SSUSI reconstruction benchmark.

Purpose
-------
Evaluate whether the OVATION-trained diffusion models can reconstruct *real,
actually observed* SSUSI auroral structure. The script creates artificial
MLT-sector gaps only inside the UT_N-supported SSUSI swath, reconstructs those
held-out pixels, and scores predictions against the withheld SSUSI ground
truth.

Compared methods
----------------
1. Solar-wind-conditioned DDPM (paper model)
2. Unconditional DDPM
3. Time-aligned OVATION Prime interval-mean map
4. Spatial interpolation baseline

The benchmark deliberately targets aurorally active observed sectors in
60--80 deg MLAT. For each case and each requested MLT width, the script
selects the strongest non-overlapping sectors that satisfy minimum observation
coverage and signal-content criteria.

Default run:
    python paper_modif/R2-1_R2-3/ssusi_heldout_benchmark.py

Outputs:
    metrics_per_holdout.csv
    metrics_summary.csv
    selected_holdouts.json
    benchmark_summary.json
    diagnostics/*.png
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.interpolate import griddata

try:
    import torch_npu  # type: ignore  # noqa: F401
except Exception:
    torch_npu = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
REPO_ROOT = SCRIPT_PATH.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from heldout_eval_utils import compute_metrics, select_holdout_windows
from models.unet import UNet as ConditionalUNet
from models.ddpm import DDPM
from models.simplenet import UNet as UnconditionalUNet
from models.ddpm_nocond import DDPM_nocond


DEFAULT_PREPARED_ROOT = REPO_ROOT / "paper_modif/R2-1_R2-3/swath_prepared_cases"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "paper_modif/R2-1_R2-3/ssusi_heldout_benchmark_results"
DEFAULT_COND_CHECKPOINT = REPO_ROOT / "ckpt/cond/aurora_diff_best.pth"
DEFAULT_UNCOND_CHECKPOINT = REPO_ROOT / "ckpt/uncond/aurora_diff_best.pth"
DEFAULT_COND_NORM = REPO_ROOT / "ckpt/cond/norm_params.pkl"
DEFAULT_UNCOND_NORM = REPO_ROOT / "ckpt/uncond/norm_params.pkl"
DEFAULT_OMNI = Path(
    "/home/docker/data/private/AuroraData/omni_real_data/omni_5min/2005/"
    "omni_20050101_5min.npy"
)
DEFAULT_OVATION = Path(
    "/home/docker/data/private/AuroraData/generated_aurora_data/2005_omni_aurora/"
    "aurora_img_20050101.npy"
)

CASES = ("Figure6", "Figure7")
SOLAR_FIELDS = ("Bx", "By", "Bz", "V", "P")
MLAT_1D = np.linspace(50.0, 90.0, 80)
MLT_1D = np.linspace(0.0, 24.0, 96, endpoint=False)

NUM_INFERENCE_STEPS = 300
N_SAMPLE = 1
JUMP_LENGTH = 10
JUMP_REPEATS = 10

METRIC_COLUMNS = (
    "mae", "rmse", "pearson_r", "r2", "peak_ratio", "gradient_ratio",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared-root", type=Path, default=DEFAULT_PREPARED_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--cond-checkpoint", type=Path, default=DEFAULT_COND_CHECKPOINT)
    p.add_argument("--uncond-checkpoint", type=Path, default=DEFAULT_UNCOND_CHECKPOINT)
    p.add_argument("--cond-norm", type=Path, default=DEFAULT_COND_NORM)
    p.add_argument("--uncond-norm", type=Path, default=DEFAULT_UNCOND_NORM)
    p.add_argument("--omni", type=Path, default=DEFAULT_OMNI)
    p.add_argument("--ovation", type=Path, default=DEFAULT_OVATION)
    p.add_argument("--device", default="auto")
    p.add_argument("--widths", type=float, nargs="+", default=[1.0, 2.0, 4.0])
    p.add_argument("--windows-per-width", type=int, default=2)
    p.add_argument("--min-coverage", type=float, default=0.50)
    p.add_argument("--min-signal-fraction", type=float, default=0.10)
    p.add_argument("--signal-threshold", type=float, default=0.10)
    p.add_argument("--candidate-step", type=float, default=0.50)
    p.add_argument("--band-min", type=float, default=60.0)
    p.add_argument("--band-max", type=float, default=80.0)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--vmax", type=float, default=5.0)
    p.add_argument("--bootstrap", type=int, default=2000)
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


def load_pickle(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        return pickle.load(f)


def resolve_uncond_norm(path: Path, fallback: Path) -> Tuple[Path, bool]:
    if path.exists():
        return path, False
    if fallback.exists():
        return fallback, True
    raise FileNotFoundError(
        f"Neither unconditional norm file {path} nor fallback {fallback} exists."
    )


def normalize_aurora(flux: np.ndarray, norm: Dict[str, Any]) -> np.ndarray:
    x = np.clip(np.asarray(flux, dtype=np.float32), 0.0, None)
    log_x = np.log1p(x)
    return np.clip(
        (log_x - float(norm["aurora"]["min"])) / float(norm["aurora"]["range"]),
        0.0, 1.0
    ).astype(np.float32)


def denormalize_aurora(y: np.ndarray, norm: Dict[str, Any]) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    log_x = y * float(norm["aurora"]["range"]) + float(norm["aurora"]["min"])
    return np.expm1(log_x).astype(np.float32)


def normalize_solar(raw: np.ndarray, norm: Dict[str, Any]) -> np.ndarray:
    minimum = np.asarray(norm["omni"]["min"], dtype=np.float32)
    value_range = np.asarray(norm["omni"]["range"], dtype=np.float32)
    raw = np.asarray(raw, dtype=np.float32)
    if raw.shape[-1] != minimum.shape[-1]:
        raise ValueError(
            f"Solar vector has {raw.shape[-1]} fields, but training normalization "
            f"contains {minimum.shape[-1]}."
        )
    return np.clip((raw - minimum) / value_range, 0.0, 1.0).astype(np.float32)


def load_case(prepared_root: Path, case: str):
    case_dir = prepared_root / case
    npz_path = case_dir / f"{case}_swath_case.npz"
    meta_path = case_dir / f"{case}_metadata.json"
    if not npz_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Prepared case missing under {case_dir}. Run prepare_ssusi_swath_cases.py first."
        )
    data = dict(np.load(npz_path, allow_pickle=False))
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    return data, meta


def as_datetime(value: Any) -> datetime:
    return pd.Timestamp(value).to_pydatetime()


def interval_mean_solar(omni: np.ndarray, start: datetime, end: datetime) -> np.ndarray:
    times = pd.to_datetime(omni["utc"]).values.astype("datetime64[ns]")
    start64 = np.datetime64(start, "ns")
    end64 = np.datetime64(end, "ns")
    mask = (times >= start64) & (times <= end64)
    if not np.any(mask):
        raise RuntimeError(f"No OMNI data within {start} -- {end}")
    columns = []
    for field in SOLAR_FIELDS:
        if field not in omni.dtype.names:
            raise KeyError(f"OMNI field {field} not found in {omni.dtype.names}")
        columns.append(np.asarray(omni[field][mask], dtype=np.float32))
    arr = np.column_stack(columns)
    out = np.nanmean(arr, axis=0).astype(np.float32)
    if not np.all(np.isfinite(out)):
        raise ValueError(f"Non-finite interval-mean OMNI vector: {out}")
    return out


def ovation_start_from_path(path: Path) -> datetime:
    token = path.stem.split("_")[-1]
    return datetime.strptime(token, "%Y%m%d")


def ovation_interval_mean(
    ovation: np.ndarray, ovation_start: datetime, start: datetime, end: datetime
) -> Tuple[np.ndarray, int, int]:
    i0 = int(np.floor((start - ovation_start).total_seconds() / 60.0))
    i1 = int(np.ceil((end - ovation_start).total_seconds() / 60.0))
    i0 = max(0, i0)
    i1 = min(len(ovation) - 1, i1)
    if i1 < i0:
        raise ValueError(f"Invalid OVATION interval indices {i0}, {i1}")
    return np.mean(ovation[i0:i1 + 1], axis=0).astype(np.float32), i0, i1


def build_conditional_model(path: Path, device: torch.device) -> DDPM:
    if not path.exists():
        raise FileNotFoundError(f"Conditional checkpoint not found: {path}")
    model = DDPM(ConditionalUNet(1, 1), num_train_steps=1000, schedule="cosine")
    ckpt = torch.load(str(path), map_location="cpu")
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=True)
    model.to(device).eval()
    return model


def build_unconditional_model(path: Path, device: torch.device) -> DDPM_nocond:
    if not path.exists():
        raise FileNotFoundError(
            f"Unconditional checkpoint not found: {path}\n"
            "Expected the checkpoint previously used by test_ddpm_indice.py."
        )
    model = DDPM_nocond(
        UnconditionalUNet(1, 1), num_train_steps=1000, schedule="cosine"
    )
    ckpt = torch.load(str(path), map_location="cpu")
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=True)
    model.to(device).eval()
    return model


def set_seed(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "npu" and hasattr(torch, "npu"):
        try:
            torch.npu.manual_seed_all(seed)
        except Exception:
            pass
    elif device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def run_conditional(
    model: DDPM,
    device: torch.device,
    truth_flux: np.ndarray,
    input_mask: np.ndarray,
    solar_norm: np.ndarray,
    norm: Dict[str, Any],
    seed: int,
) -> np.ndarray:
    input_physical = np.where(input_mask, truth_flux, 0.0).astype(np.float32)
    x = normalize_aurora(input_physical, norm)
    image = torch.from_numpy(x[None, None]).to(device=device, dtype=torch.float32)
    mask = torch.from_numpy(input_mask.astype(np.float32)[None, None]).to(device)
    solar = torch.from_numpy(solar_norm[None]).to(device=device, dtype=torch.float32)
    set_seed(seed, device)
    with torch.no_grad():
        out = model.sample(
            image, mask, solar,
            num_inference_steps=NUM_INFERENCE_STEPS,
            n_sample=N_SAMPLE, j=JUMP_LENGTH, r=JUMP_REPEATS,
        )
    out_n = np.clip(out.detach().cpu().numpy()[0, 0], 0.0, 1.0)
    pred = np.clip(denormalize_aurora(out_n, norm), 0.0, None)
    pred[input_mask] = truth_flux[input_mask]
    return pred.astype(np.float32)


def run_unconditional(
    model: DDPM_nocond,
    device: torch.device,
    truth_flux: np.ndarray,
    input_mask: np.ndarray,
    norm: Dict[str, Any],
    seed: int,
) -> np.ndarray:
    input_physical = np.where(input_mask, truth_flux, 0.0).astype(np.float32)
    x = normalize_aurora(input_physical, norm)
    image = torch.from_numpy(x[None, None]).to(device=device, dtype=torch.float32)
    mask = torch.from_numpy(input_mask.astype(np.float32)[None, None]).to(device)
    set_seed(seed, device)
    with torch.no_grad():
        out = model.sample(
            image, mask,
            num_inference_steps=NUM_INFERENCE_STEPS,
            n_sample=N_SAMPLE, j=JUMP_LENGTH, r=JUMP_REPEATS,
        )
    out_n = np.clip(out.detach().cpu().numpy()[0, 0], 0.0, 1.0)
    pred = np.clip(denormalize_aurora(out_n, norm), 0.0, None)
    pred[input_mask] = truth_flux[input_mask]
    return pred.astype(np.float32)


def interpolation_baseline(
    truth_flux: np.ndarray, input_mask: np.ndarray, target_mask: np.ndarray
) -> np.ndarray:
    rows, cols = np.indices(truth_flux.shape)
    known = input_mask & np.isfinite(truth_flux)
    target = target_mask.astype(bool)
    if not np.any(known):
        raise ValueError("No known pixels for interpolation baseline.")

    kr = rows[known].astype(float)
    kc = cols[known].astype(float)
    kv = truth_flux[known].astype(float)

    w = truth_flux.shape[1]
    points = np.column_stack([
        np.concatenate([kr, kr, kr]),
        np.concatenate([kc - w, kc, kc + w]),
    ])
    values = np.concatenate([kv, kv, kv])
    targets = np.column_stack([rows[target], cols[target]])

    linear = griddata(points, values, targets, method="linear", fill_value=np.nan)
    missing = ~np.isfinite(linear)
    if np.any(missing):
        linear[missing] = griddata(
            points, values, targets[missing], method="nearest"
        )
    result = np.array(truth_flux, copy=True, dtype=np.float32)
    result[target] = np.clip(linear, 0.0, None).astype(np.float32)
    return result


def plot_holdout(
    path: Path,
    case: str,
    width_h: float,
    rank: int,
    center_h: float,
    truth: np.ndarray,
    input_mask: np.ndarray,
    cond: np.ndarray,
    uncond: np.ndarray,
    ovation: np.ndarray,
    interp: np.ndarray,
    vmax: float,
) -> None:
    input_panel = np.where(input_mask, truth, np.nan)
    extent = [0, 24, 50, 90]
    panels = [
        ("(a) SSUSI truth", truth),
        ("(b) Input with held-out sector", input_panel),
        ("(c) Conditional DDPM", cond),
        ("(d) Unconditional DDPM", uncond),
        ("(e) OVATION Prime", ovation),
        ("(f) Interpolation", interp),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), dpi=150, sharex=True, sharey=True)
    ims = []
    for ax, (title, data) in zip(axes.flat, panels):
        im = ax.imshow(
            data, origin="lower", aspect="auto", extent=extent,
            vmin=0, vmax=vmax, cmap="viridis"
        )
        ax.set_title(title)
        ax.set_xlabel("MLT [h]")
        ax.set_ylabel("MLAT [deg]")
        ims.append(im)
    fig.suptitle(
        f"{case}: held-out SSUSI sector, width={width_h:g} h, "
        f"center={center_h:.2f} MLT, rank={rank}",
        fontsize=15,
    )
    cax = fig.add_axes([0.18, 0.04, 0.64, 0.022])
    cb = plt.colorbar(ims[0], cax=cax, orientation="horizontal")
    cb.set_label(r"Auroral Electron Energy Flux (erg cm$^{-2}$ s$^{-1}$)")
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)


def bootstrap_mean_ci(values: Iterable[float], n_boot: int, seed: int):
    x = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if x.size == 0:
        return np.nan, np.nan, np.nan, 0
    mean = float(np.mean(x))
    if x.size == 1 or n_boot <= 0:
        return mean, np.nan, np.nan, int(x.size)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        boot[i] = np.mean(rng.choice(x, size=x.size, replace=True))
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return mean, float(lo), float(hi), int(x.size)


def summarize_metrics(df: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rows = []
    group_cols = ["width_h", "method"]
    for keys, group in df.groupby(group_cols):
        width_h, method = keys
        for metric in METRIC_COLUMNS:
            mean, lo, hi, n = bootstrap_mean_ci(
                group[metric].to_numpy(dtype=float), n_boot=n_boot,
                seed=seed + int(round(width_h * 100)) + sum(map(ord, method))
            )
            finite = group[metric].replace([np.inf, -np.inf], np.nan).dropna()
            rows.append({
                "width_h": width_h,
                "method": method,
                "metric": metric,
                "n": n,
                "mean": mean,
                "std": float(finite.std(ddof=1)) if len(finite) > 1 else np.nan,
                "median": float(finite.median()) if len(finite) else np.nan,
                "ci95_low": lo,
                "ci95_high": hi,
            })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    diag_dir = args.output_root / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    uncond_norm_path, norm_fallback = resolve_uncond_norm(args.uncond_norm, args.cond_norm)
    cond_norm = load_pickle(args.cond_norm)
    uncond_norm = load_pickle(uncond_norm_path)

    cond_model = build_conditional_model(args.cond_checkpoint, device)
    uncond_model = build_unconditional_model(args.uncond_checkpoint, device)

    omni = np.load(args.omni, allow_pickle=True)
    ovation = np.load(args.ovation, allow_pickle=False).astype(np.float32)
    ova_start = ovation_start_from_path(args.ovation)

    print("=" * 88)
    print("HELD-OUT REAL-SSUSI RECONSTRUCTION BENCHMARK")
    print("=" * 88)
    print(f"Device:                  {device}")
    print(f"Conditional checkpoint:  {args.cond_checkpoint}")
    print(f"Unconditional checkpoint:{args.uncond_checkpoint}")
    print(f"Conditional norm:        {args.cond_norm}")
    print(f"Unconditional norm:      {uncond_norm_path}")
    if norm_fallback:
        print("WARNING: ckpt/uncond/norm_params.pkl not found; using conditional "
              "training aurora normalization for the unconditional model.")
    print(f"Widths [h]:              {args.widths}")
    print(f"Windows per width:       {args.windows_per_width}")
    print(f"Auroral band:            {args.band_min:.1f}--{args.band_max:.1f} MLAT")
    print(f"Selection threshold:     flux >= {args.signal_threshold}")
    print()

    records = []
    selections_out: Dict[str, Any] = {}
    run_counter = 0

    for case_idx, case in enumerate(CASES):
        data, meta = load_case(args.prepared_root, case)
        flux = np.asarray(data["flux_grid"], dtype=np.float32)
        obs = np.asarray(data["obs_mask"], dtype=bool)
        start = as_datetime(meta["acquisition_start"])
        end = as_datetime(meta["acquisition_end"])

        solar_raw = interval_mean_solar(omni, start, end)
        solar_norm = normalize_solar(solar_raw, cond_norm)
        ova_mean, ova_i0, ova_i1 = ovation_interval_mean(
            ovation, ova_start, start, end
        )

        print("-" * 88)
        print(case)
        print(f"Acquisition: {start} -> {end}")
        print(f"Observed fraction: {obs.mean():.4f}")
        print(f"Interval-mean solar: {dict(zip(SOLAR_FIELDS, map(float, solar_raw)))}")
        print(f"OVATION frames: {ova_i0}..{ova_i1} ({ova_i1-ova_i0+1})")

        selections_out[case] = {}

        for width_idx, width_h in enumerate(args.widths):
            wins = select_holdout_windows(
                flux, obs, MLAT_1D, MLT_1D,
                width_h=width_h,
                n_windows=args.windows_per_width,
                band=(args.band_min, args.band_max),
                min_coverage=args.min_coverage,
                min_signal_fraction=args.min_signal_fraction,
                signal_threshold=args.signal_threshold,
                candidate_step_h=args.candidate_step,
            )
            if len(wins) < args.windows_per_width:
                print(
                    f"WARNING: {case}, width={width_h:g} h: selected only "
                    f"{len(wins)} of requested {args.windows_per_width} windows."
                )

            selections_out[case][str(width_h)] = []
            for rank, win in enumerate(wins, start=1):
                holdout = np.asarray(win["mask"], dtype=bool)
                input_mask = obs & (~holdout)
                seed = (
                    args.seed
                    + case_idx * 1000
                    + width_idx * 100
                    + rank * 10
                )

                run_counter += 1
                print(
                    f"  [{run_counter:02d}] width={width_h:g} h rank={rank} "
                    f"center={win['center_h']:.2f} MLT "
                    f"pixels={win['n_pixels']} coverage={win['coverage']:.3f} "
                    f"signal={win['signal_fraction']:.3f}"
                )

                t0 = time.perf_counter()
                pred_cond = run_conditional(
                    cond_model, device, flux, input_mask, solar_norm,
                    cond_norm, seed=seed
                )
                t_cond = time.perf_counter() - t0

                t0 = time.perf_counter()
                pred_uncond = run_unconditional(
                    uncond_model, device, flux, input_mask,
                    uncond_norm, seed=seed
                )
                t_uncond = time.perf_counter() - t0

                pred_interp = interpolation_baseline(flux, input_mask, holdout)

                methods = {
                    "conditional_ddpm": pred_cond,
                    "unconditional_ddpm": pred_uncond,
                    "ovation_prime": ova_mean,
                    "interpolation": pred_interp,
                }
                for method, pred in methods.items():
                    metrics = compute_metrics(flux, pred, holdout)
                    row = {
                        "case": case,
                        "width_h": float(width_h),
                        "rank": int(rank),
                        "center_mlt_h": float(win["center_h"]),
                        "holdout_pixels": int(win["n_pixels"]),
                        "coverage": float(win["coverage"]),
                        "signal_fraction": float(win["signal_fraction"]),
                        "selection_score": float(win["score"]),
                        "method": method,
                        **metrics,
                    }
                    if method == "conditional_ddpm":
                        row["runtime_s"] = float(t_cond)
                    elif method == "unconditional_ddpm":
                        row["runtime_s"] = float(t_uncond)
                    else:
                        row["runtime_s"] = 0.0
                    records.append(row)
                    print(
                        f"       {method:20s} RMSE={metrics['rmse']:.4f} "
                        f"MAE={metrics['mae']:.4f} r={metrics['pearson_r']:.3f} "
                        f"peak={metrics['peak_ratio']:.3f} grad={metrics['gradient_ratio']:.3f}"
                    )

                selections_out[case][str(width_h)].append({
                    key: value for key, value in win.items() if key != "mask"
                })

                plot_holdout(
                    diag_dir / f"{case}_width{width_h:g}h_rank{rank}.png",
                    case, width_h, rank, win["center_h"], flux, input_mask,
                    pred_cond, pred_uncond, ova_mean, pred_interp, args.vmax
                )

    if not records:
        raise RuntimeError(
            "No held-out windows met the selection criteria. Lower "
            "--min-coverage or --min-signal-fraction."
        )

    metrics_df = pd.DataFrame(records)
    metrics_path = args.output_root / "metrics_per_holdout.csv"
    metrics_df.to_csv(metrics_path, index=False)

    summary_df = summarize_metrics(metrics_df, args.bootstrap, args.seed)
    summary_path = args.output_root / "metrics_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    with (args.output_root / "selected_holdouts.json").open("w", encoding="utf-8") as f:
        json.dump(selections_out, f, indent=2)

    overview: Dict[str, Any] = {
        "experiment": {
            "cases": list(CASES),
            "widths_h": [float(x) for x in args.widths],
            "windows_per_width": int(args.windows_per_width),
            "band_mlat": [float(args.band_min), float(args.band_max)],
            "selection": {
                "min_coverage": float(args.min_coverage),
                "min_signal_fraction": float(args.min_signal_fraction),
                "signal_threshold": float(args.signal_threshold),
                "candidate_step_h": float(args.candidate_step),
            },
            "conditioning": "OMNI interval mean over each SSUSI EDR acquisition interval",
            "ovation_baseline": "1-min OVATION Prime mean over the same acquisition interval",
            "random_seed": int(args.seed),
        },
        "paths": {
            "conditional_checkpoint": str(args.cond_checkpoint),
            "unconditional_checkpoint": str(args.uncond_checkpoint),
            "conditional_norm": str(args.cond_norm),
            "unconditional_norm": str(uncond_norm_path),
            "omni": str(args.omni),
            "ovation": str(args.ovation),
        },
        "n_holdouts": int(
            metrics_df[["case", "width_h", "rank"]].drop_duplicates().shape[0]
        ),
        "n_method_evaluations": int(len(metrics_df)),
    }
    with (args.output_root / "benchmark_summary.json").open("w", encoding="utf-8") as f:
        json.dump(overview, f, indent=2)

    print("\n" + "=" * 88)
    print("AGGREGATE SUMMARY")
    print("=" * 88)
    for width_h in sorted(metrics_df["width_h"].unique()):
        print(f"\nWidth = {width_h:g} h")
        sub = metrics_df[metrics_df["width_h"] == width_h]
        for method in ("conditional_ddpm", "unconditional_ddpm", "ovation_prime", "interpolation"):
            group = sub[sub["method"] == method]
            if group.empty:
                continue
            print(
                f"  {method:20s} "
                f"RMSE={group['rmse'].mean():.4f}±{group['rmse'].std(ddof=1):.4f}, "
                f"MAE={group['mae'].mean():.4f}, "
                f"r={group['pearson_r'].mean():.3f}, "
                f"peak_ratio={group['peak_ratio'].mean():.3f}, "
                f"gradient_ratio={group['gradient_ratio'].mean():.3f}"
            )

    print("\nOutputs:")
    print(f"  {metrics_path}")
    print(f"  {summary_path}")
    print(f"  {args.output_root / 'selected_holdouts.json'}")
    print(f"  {args.output_root / 'benchmark_summary.json'}")
    print(f"  {diag_dir}")
    print("\nPlease send the complete console output plus metrics_per_holdout.csv "
          "and metrics_summary.csv.")


if __name__ == "__main__":
    main()
