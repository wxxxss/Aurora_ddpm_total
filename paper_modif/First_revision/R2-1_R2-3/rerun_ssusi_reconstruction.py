#!/usr/bin/env python3
"""Re-run the two SSUSI manuscript cases with swath-aware observation masks.

This script is the second stage of the Reviewer-2 Comment-3 correction.
It consumes the direct-binned SSUSI cases produced by
`prepare_ssusi_swath_cases.py`, uses the paper conditional DDPM checkpoint,
and regenerates the Figure-6/Figure-7 reconstructions without treating
interpolated or low-flux pixels as missing observations.

Key corrections relative to the legacy SSUSI pipeline:
  1. Observation support comes from the prepared UT_N > 0 direct-support mask.
  2. Missingness is independent of flux magnitude; low measured flux remains observed.
  3. Solar-wind conditioning uses the five paper variables [Bx, By, Bz, V, P].
  4. Aurora and OMNI values use the training normalization parameters.
  5. After inverse normalization, measured SSUSI pixels are overwritten with their
     exact physical energy-flux values so observations are preserved exactly.
  6. The paper plots retain missing regions in the original SSUSI panel and include
     explicit (a)/(b)/(c) panel labels.

Run from the repository root:
    python paper_modif/R2-1_R2-3/rerun_ssusi_reconstruction.py

Useful overrides:
    --checkpoint /path/to/aurora_diff_best.pth
    --norm-params /path/to/norm_params.pkl
    --device npu:0
    --vmax 5
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import torch

try:
    import torch_npu  # type: ignore  # noqa: F401
except Exception:
    torch_npu = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.image import imread

import cartopy.crs as ccrs
from cartopy.feature.nightshade import Nightshade
import aacgmv2


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.unet import UNet
from models.ddpm import DDPM


# -----------------------------------------------------------------------------
# Defaults matching the revised SSUSI preparation and final paper model.
# -----------------------------------------------------------------------------
DEFAULT_PREPARED_ROOT = REPO_ROOT / "paper_modif/R2-1_R2-3/swath_prepared_cases"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "paper_modif/R2-1_R2-3/swath_reconstruction_results"
DEFAULT_CHECKPOINT = REPO_ROOT / "ckpt/cond/aurora_diff_best.pth"
DEFAULT_NORM_PARAMS = REPO_ROOT / "ckpt/cond/norm_params.pkl"
LEGACY_NORM_PARAMS = Path(
    "/home/docker/code/Aurora_DDPM_final/ckpt/cond/ckptv4_unetv1/norm_params.pkl"
)
DEFAULT_OVATION = Path(
    "/home/docker/data/private/AuroraData/generated_aurora_data/2005_omni_aurora/"
    "aurora_img_20050101.npy"
)
DEFAULT_OMNI = Path(
    "/home/docker/data/private/AuroraData/omni_real_data/omni_5min/2005/"
    "omni_20050101_5min.npy"
)
DEFAULT_BACKGROUND = Path(
    "/home/docker/data/private/AuroraData/background_img/"
    "natural-earth-1_large2048px.png"
)

CASES = ("Figure6", "Figure7")
SOLAR_FIELDS = ("Bx", "By", "Bz", "V", "P")
NUM_INFERENCE_STEPS = 300
N_SAMPLE = 1
JUMP_LENGTH = 10
JUMP_REPEATS = 10
MLAT_1D = np.linspace(50.0, 90.0, 80)
MLT_1D = np.linspace(0.0, 24.0, 96, endpoint=False)
ORTHO_LON = 110.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared-root", type=Path, default=DEFAULT_PREPARED_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--norm-params", type=Path, default=DEFAULT_NORM_PARAMS)
    p.add_argument("--ovation", type=Path, default=DEFAULT_OVATION)
    p.add_argument("--omni", type=Path, default=DEFAULT_OMNI)
    p.add_argument("--background", type=Path, default=DEFAULT_BACKGROUND)
    p.add_argument("--device", default="auto")
    p.add_argument("--vmax", type=float, default=5.0,
                   help="Fixed paper-comparison colorbar maximum. Default: 5.")
    p.add_argument("--seed", type=int, default=2026)
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


def resolve_norm_path(path: Path) -> Path:
    if path.exists():
        return path
    if LEGACY_NORM_PARAMS.exists():
        print(f"Norm-parameter file not found at {path}; using {LEGACY_NORM_PARAMS}")
        return LEGACY_NORM_PARAMS
    raise FileNotFoundError(
        f"Training normalization file not found at {path}. "
        "Pass --norm-params /path/to/the norm_params.pkl created during final training."
    )


def load_norm_params(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        params = pickle.load(f)
    if "aurora" not in params or "omni" not in params:
        raise KeyError(f"Unexpected normalization structure in {path}: {params.keys()}")
    return params


def normalize_aurora_physical(flux: np.ndarray, norm: Dict[str, Any]) -> np.ndarray:
    x = np.clip(np.asarray(flux, dtype=np.float32), 0.0, None)
    log_x = np.log1p(x)
    y = (log_x - float(norm["aurora"]["min"])) / float(norm["aurora"]["range"])
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def denormalize_aurora(y: np.ndarray, norm: Dict[str, Any]) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    log_x = y * float(norm["aurora"]["range"]) + float(norm["aurora"]["min"])
    return np.expm1(log_x).astype(np.float32)


def normalize_solar(raw: np.ndarray, norm: Dict[str, Any]) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float32)
    minimum = np.asarray(norm["omni"]["min"], dtype=np.float32)
    value_range = np.asarray(norm["omni"]["range"], dtype=np.float32)
    if raw.shape[-1] != minimum.shape[-1]:
        raise ValueError(
            f"Solar vector has {raw.shape[-1]} variables but training normalization has "
            f"{minimum.shape[-1]}. Expected the final five-variable model."
        )
    y = (raw - minimum) / value_range
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def as_datetime(value: Any) -> datetime:
    return pd.Timestamp(value).to_pydatetime()


def nearest_time_index(times: np.ndarray, target: datetime) -> Tuple[int, datetime, float]:
    times_ns = pd.to_datetime(times).values.astype("datetime64[ns]")
    target64 = np.datetime64(target, "ns")
    delta_ns = np.abs(times_ns.astype("int64") - target64.astype("int64"))
    idx = int(np.argmin(delta_ns))
    matched = pd.Timestamp(times_ns[idx]).to_pydatetime()
    delta_s = abs((matched - target).total_seconds())
    return idx, matched, float(delta_s)


def extract_solar_row(omni: np.ndarray, idx: int) -> np.ndarray:
    names = omni.dtype.names
    if names is None:
        raise TypeError("OMNI array must be a structured NumPy array with named fields.")
    row = []
    for field in SOLAR_FIELDS:
        if field in names:
            val = np.asarray(omni[field][idx]).reshape(-1)[0]
        elif field == "V" and all(v in names for v in ("Vx", "Vy", "Vz")):
            vx = float(np.asarray(omni["Vx"][idx]).reshape(-1)[0])
            vy = float(np.asarray(omni["Vy"][idx]).reshape(-1)[0])
            vz = float(np.asarray(omni["Vz"][idx]).reshape(-1)[0])
            val = np.sqrt(vx * vx + vy * vy + vz * vz)
        else:
            raise KeyError(f"Required solar-wind field {field!r} not found in OMNI fields {names}")
        row.append(float(val))
    raw = np.asarray(row, dtype=np.float32)
    if not np.all(np.isfinite(raw)):
        raise ValueError(f"Non-finite OMNI conditioning vector at index {idx}: {raw}")
    return raw


def make_aurora_cmap() -> LinearSegmentedColormap:
    colors = [
        (0.0, 0.2, 0.0), (0.0, 0.5, 0.0), (0.0, 0.8, 0.0),
        (0.5, 1.0, 0.0), (1.0, 1.0, 0.0), (1.0, 0.6, 0.0),
        (1.0, 0.3, 0.0), (1.0, 0.0, 0.0),
    ]
    cm = LinearSegmentedColormap.from_list("aurora", colors, N=256)
    cm.set_bad(color="white", alpha=0.0)
    return cm


AURORA_CMAP = make_aurora_cmap()


def build_model(checkpoint_path: Path, device: torch.device) -> DDPM:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. Pass --checkpoint explicitly."
        )
    unet = UNet(1, 1)
    ddpm = DDPM(unet, num_train_steps=1000, schedule="cosine")
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state = checkpoint.get("model_state_dict", checkpoint)
    ddpm.load_state_dict(state, strict=True)
    ddpm.to(device)
    ddpm.eval()
    return ddpm


def load_case(prepared_root: Path, case: str) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    case_dir = prepared_root / case
    npz_path = case_dir / f"{case}_swath_case.npz"
    meta_path = case_dir / f"{case}_metadata.json"
    if not npz_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Prepared {case} inputs not found under {case_dir}. "
            "Run prepare_ssusi_swath_cases.py first."
        )
    data = dict(np.load(npz_path, allow_pickle=False))
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    return data, meta


def reconstruct_case(
    ddpm: DDPM,
    device: torch.device,
    flux_grid: np.ndarray,
    obs_mask: np.ndarray,
    solar_norm: np.ndarray,
    norm: Dict[str, Any],
    seed: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    # Missing cells are set to physical zero before normalization. They are never
    # treated as observations because mask=0 there.
    input_physical = np.where(obs_mask, flux_grid, 0.0).astype(np.float32)
    input_norm = normalize_aurora_physical(input_physical, norm)

    image = torch.from_numpy(input_norm[None, None]).to(device=device, dtype=torch.float32)
    mask = torch.from_numpy(obs_mask.astype(np.float32)[None, None]).to(
        device=device, dtype=torch.float32
    )
    solar = torch.from_numpy(solar_norm[None]).to(device=device, dtype=torch.float32)

    torch.manual_seed(seed)
    if device.type == "npu" and hasattr(torch, "npu"):
        try:
            torch.npu.manual_seed_all(seed)
        except Exception:
            pass
    elif device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    with torch.no_grad():
        out = ddpm.sample(
            image,
            mask,
            solar,
            num_inference_steps=NUM_INFERENCE_STEPS,
            n_sample=N_SAMPLE,
            j=JUMP_LENGTH,
            r=JUMP_REPEATS,
        )

    out_norm_raw = out.detach().cpu().numpy()[0, 0].astype(np.float32)
    diagnostics = {
        "output_norm_min_before_clip": float(np.nanmin(out_norm_raw)),
        "output_norm_max_before_clip": float(np.nanmax(out_norm_raw)),
    }

    # Training targets live in [0, 1]. Clip model extrapolation before inversion.
    out_norm = np.clip(out_norm_raw, 0.0, 1.0)
    repaired = denormalize_aurora(out_norm, norm)
    repaired = np.clip(repaired, 0.0, None)

    # Critical: restore measured SSUSI energy flux exactly in observed cells.
    repaired[obs_mask] = flux_grid[obs_mask]
    return repaired.astype(np.float32), diagnostics


def magnetic_grid_to_geographic(timestamp: datetime) -> Tuple[np.ndarray, np.ndarray]:
    mlt_grid, mlat_grid = np.meshgrid(MLT_1D, MLAT_1D)
    mlon_row = np.asarray(aacgmv2.convert_mlt(MLT_1D, timestamp, m2a=True))
    mlon = np.tile(mlon_row[None, :], (len(MLAT_1D), 1))
    mlat_flat = mlat_grid.reshape(-1)
    mlon_flat = mlon.reshape(-1)
    glat, glon, _ = aacgmv2.convert_latlon_arr(
        mlat_flat, mlon_flat, 100, timestamp, method_code="A2G"
    )
    return np.asarray(glat).reshape(mlat_grid.shape), np.asarray(glon).reshape(mlat_grid.shape)


def draw_background(ax, timestamp: datetime, background_path: Path) -> None:
    if background_path.exists():
        ax.imshow(
            imread(str(background_path)), origin="upper", transform=ccrs.PlateCarree(),
            extent=[-180, 180, -90, 90], zorder=0
        )
    else:
        ax.stock_img()
    ax.gridlines(linestyle="dashed", alpha=0.3, color="white")
    ax.coastlines("50m", color="white", alpha=0.5, linewidth=0.5)
    ax.add_feature(Nightshade(timestamp, alpha=0.4))
    ax.set_facecolor("black")


def plot_flux_on_axis(
    ax,
    flux: np.ndarray,
    glat: np.ndarray,
    glon: np.ndarray,
    vmin: float,
    vmax: float,
):
    masked = np.ma.masked_invalid(flux)
    # pcolormesh preserves the target-grid missing mask; unlike a second griddata
    # interpolation, it does not create new observational support for SSUSI.
    try:
        artist = ax.pcolormesh(
            glon, glat, masked,
            transform=ccrs.PlateCarree(),
            shading="nearest", cmap=AURORA_CMAP,
            vmin=vmin, vmax=vmax, zorder=3, alpha=0.88,
        )
    except Exception as exc:
        print(f"pcolormesh warning ({exc}); falling back to masked scatter.")
        good = np.isfinite(flux) & np.isfinite(glat) & np.isfinite(glon)
        artist = ax.scatter(
            glon[good], glat[good], c=flux[good], s=8, marker="s",
            transform=ccrs.PlateCarree(), cmap=AURORA_CMAP,
            vmin=vmin, vmax=vmax, zorder=3, alpha=0.88, linewidths=0,
        )
    return artist


def robust_auto_vmax(arrays: Iterable[np.ndarray], minimum: float = 5.0) -> float:
    values = []
    for arr in arrays:
        x = np.asarray(arr)
        x = x[np.isfinite(x) & (x >= 0)]
        if x.size:
            values.append(x)
    if not values:
        return minimum
    combined = np.concatenate(values)
    q = float(np.percentile(combined, 99.0))
    return max(minimum, q)


def plot_orthographic_triptych(
    case: str,
    timestamp: datetime,
    observed: np.ndarray,
    reconstructed: np.ndarray,
    ovation: np.ndarray,
    obs_mask: np.ndarray,
    background_path: Path,
    save_path: Path,
    vmax: float,
    orbit: str,
):
    glat, glon = magnetic_grid_to_geographic(timestamp)
    observation_for_plot = observed.copy().astype(np.float32)
    observation_for_plot[~obs_mask] = np.nan

    arrays = (observation_for_plot, reconstructed, ovation)
    titles = ("(a) SSUSI observation", "(b) Reconstruction", "(c) OVATION Prime")

    fig = plt.figure(figsize=(24, 9), dpi=150)
    fig.set_facecolor("black")
    axes = []
    artist = None
    for idx, (arr, title) in enumerate(zip(arrays, titles), start=1):
        ax = fig.add_subplot(1, 3, idx, projection=ccrs.Orthographic(ORTHO_LON, 90.0))
        draw_background(ax, timestamp, background_path)
        artist = plot_flux_on_axis(ax, arr, glat, glon, 0.0, vmax)
        ax.set_title(title, color="white", fontsize=17, fontweight="bold", pad=12)
        axes.append(ax)

    fig.suptitle(
        f"DMSP F16/SSUSI {case} — Orbit {orbit} — "
        f"{timestamp.strftime('%Y-%m-%d %H:%M UT')}",
        color="white", fontsize=20, fontweight="bold", y=0.96,
    )
    cbar_ax = fig.add_axes([0.34, 0.075, 0.32, 0.025])
    cbar = fig.colorbar(artist, cax=cbar_ax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=12, colors="white")
    cbar.set_label(
        r"Auroral Electron Energy Flux (erg cm$^{-2}$ s$^{-1}$)",
        color="white", fontsize=14,
    )
    fig.savefig(save_path, dpi=150, facecolor="black", bbox_inches="tight")
    plt.close(fig)


def plot_mlat_mlt_triptych(
    case: str,
    observed: np.ndarray,
    reconstructed: np.ndarray,
    ovation: np.ndarray,
    obs_mask: np.ndarray,
    save_path: Path,
    vmax: float,
):
    obs = observed.copy().astype(np.float32)
    obs[~obs_mask] = np.nan
    extent = [0.0, 24.0, 50.0, 90.0]
    arrays = (obs, reconstructed, ovation)
    titles = ("(a) SSUSI observation", "(b) Reconstruction", "(c) OVATION Prime")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), dpi=160, sharex=True, sharey=True)
    artist = None
    for ax, arr, title in zip(axes, arrays, titles):
        artist = ax.imshow(
            np.ma.masked_invalid(arr), origin="lower", aspect="auto", extent=extent,
            cmap=AURORA_CMAP, vmin=0.0, vmax=vmax, interpolation="nearest"
        )
        ax.set_title(title)
        ax.set_xlabel("MLT [h]")
        ax.set_ylabel("MLAT [deg]")
    cbar = fig.colorbar(artist, ax=axes.ravel().tolist(), orientation="horizontal",
                        fraction=0.07, pad=0.14, aspect=45)
    cbar.set_label(r"Auroral Electron Energy Flux (erg cm$^{-2}$ s$^{-1}$)")
    fig.suptitle(f"{case}: swath-aware SSUSI reconstruction", fontsize=15)
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def orbit_from_metadata(meta: Dict[str, Any]) -> str:
    nested = meta.get("netcdf_metadata", {})
    orbit = nested.get("STARTING_ORBIT_NUMBER", nested.get("STOPPING_ORBIT_NUMBER", "unknown"))
    return str(orbit)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    norm_path = resolve_norm_path(args.norm_params)
    norm = load_norm_params(norm_path)
    device = choose_device(args.device)

    print("=" * 80)
    print("SWATH-AWARE SSUSI RECONSTRUCTION")
    print("=" * 80)
    print(f"Device:        {device}")
    print(f"Checkpoint:    {args.checkpoint}")
    print(f"Norm params:   {norm_path}")
    print(f"OMNI:          {args.omni}")
    print(f"OVATION:       {args.ovation}")
    print(f"Prepared root: {args.prepared_root}")
    print(f"Output root:   {args.output_root}")

    ddpm = build_model(args.checkpoint, device)
    omni = np.load(args.omni, allow_pickle=True)
    ovation_all = np.load(args.ovation, allow_pickle=True)
    if omni.dtype.names is None or "utc" not in omni.dtype.names:
        raise KeyError("OMNI file must contain a structured 'utc' field.")

    if len(ovation_all) != len(omni):
        print(
            f"WARNING: OVATION length ({len(ovation_all)}) != OMNI length ({len(omni)}). "
            "The nearest OMNI index will still be used if it is within the OVATION array."
        )

    global_summary: Dict[str, Any] = {}

    for case_i, case in enumerate(CASES):
        print("\n" + "-" * 80)
        print(case)
        data, meta = load_case(args.prepared_root, case)
        flux_grid = np.asarray(data["flux_grid"], dtype=np.float32)
        obs_mask = np.asarray(data["obs_mask"], dtype=bool)
        if flux_grid.shape != (80, 96) or obs_mask.shape != (80, 96):
            raise ValueError(f"Unexpected prepared shape for {case}: {flux_grid.shape}, {obs_mask.shape}")

        midpoint = as_datetime(meta["edr_midpoint"])
        omni_idx, omni_time, time_delta_s = nearest_time_index(omni["utc"], midpoint)
        if omni_idx >= len(ovation_all):
            raise IndexError(
                f"Nearest OMNI index {omni_idx} exceeds OVATION length {len(ovation_all)}."
            )
        solar_raw = extract_solar_row(omni, omni_idx)
        solar_norm = normalize_solar(solar_raw, norm)
        ovation = np.asarray(ovation_all[omni_idx], dtype=np.float32)
        if ovation.shape != (80, 96):
            raise ValueError(f"Expected OVATION map (80, 96), got {ovation.shape}")

        repaired, diag = reconstruct_case(
            ddpm=ddpm,
            device=device,
            flux_grid=flux_grid,
            obs_mask=obs_mask,
            solar_norm=solar_norm,
            norm=norm,
            seed=args.seed + case_i,
        )

        observed_exact_error = float(np.nanmax(np.abs(repaired[obs_mask] - flux_grid[obs_mask])))
        missing = ~obs_mask
        generated_stats = {
            "generated_missing_mean": float(np.nanmean(repaired[missing])) if np.any(missing) else None,
            "generated_missing_max": float(np.nanmax(repaired[missing])) if np.any(missing) else None,
            "observed_exact_max_abs_error": observed_exact_error,
        }

        case_dir = args.output_root / case
        case_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            case_dir / f"{case}_reconstruction.npz",
            observed_flux=flux_grid,
            obs_mask=obs_mask.astype(np.uint8),
            reconstructed_flux=repaired,
            ovation_flux=ovation,
            solar_raw=solar_raw,
            solar_normalized=solar_norm,
            mlat_1d=MLAT_1D,
            mlt_1d=MLT_1D,
        )

        orbit = orbit_from_metadata(meta)
        fixed_vmax = float(args.vmax)
        auto_vmax = robust_auto_vmax(
            [np.where(obs_mask, flux_grid, np.nan), repaired, ovation], minimum=fixed_vmax
        )

        plot_orthographic_triptych(
            case, midpoint, flux_grid, repaired, ovation, obs_mask,
            args.background,
            case_dir / f"{case}_comparison_vmax{fixed_vmax:g}.png",
            fixed_vmax, orbit,
        )
        plot_orthographic_triptych(
            case, midpoint, flux_grid, repaired, ovation, obs_mask,
            args.background,
            case_dir / f"{case}_comparison_autoscale.png",
            auto_vmax, orbit,
        )
        plot_mlat_mlt_triptych(
            case, flux_grid, repaired, ovation, obs_mask,
            case_dir / f"{case}_mlat_mlt_comparison.png", fixed_vmax,
        )

        case_summary = {
            "case": case,
            "edr_midpoint": midpoint.isoformat(),
            "acquisition_start": meta.get("acquisition_start"),
            "acquisition_end": meta.get("acquisition_end"),
            "orbit": orbit,
            "observed_fraction": float(obs_mask.mean()),
            "omni_index": int(omni_idx),
            "omni_time": omni_time.isoformat(),
            "omni_time_offset_seconds": time_delta_s,
            "solar_fields": list(SOLAR_FIELDS),
            "solar_raw": {k: float(v) for k, v in zip(SOLAR_FIELDS, solar_raw)},
            "solar_normalized": {k: float(v) for k, v in zip(SOLAR_FIELDS, solar_norm)},
            "fixed_vmax": fixed_vmax,
            "auto_vmax_99th_percentile": auto_vmax,
            **diag,
            **generated_stats,
        }
        with (case_dir / f"{case}_reconstruction_summary.json").open("w", encoding="utf-8") as f:
            json.dump(case_summary, f, indent=2, ensure_ascii=False)
        global_summary[case] = case_summary

        print(f"EDR midpoint:       {midpoint}")
        print(f"Orbit:              {orbit}")
        print(f"Observed fraction:  {obs_mask.mean():.4f}")
        print(f"Nearest OMNI time:  {omni_time} (offset {time_delta_s:.1f} s)")
        print(f"Solar raw:          {dict(zip(SOLAR_FIELDS, solar_raw.tolist()))}")
        print(f"Solar normalized:   {dict(zip(SOLAR_FIELDS, solar_norm.tolist()))}")
        print(f"Output norm range before clip: {diag['output_norm_min_before_clip']:.4f} to "
              f"{diag['output_norm_max_before_clip']:.4f}")
        print(f"Observed-pixel preservation max abs error: {observed_exact_error:.6g}")
        print(f"Generated missing max: {generated_stats['generated_missing_max']:.4f}")
        print(f"Fixed / auto vmax:  {fixed_vmax:.3f} / {auto_vmax:.3f}")
        print(f"Outputs:            {case_dir}")

    with (args.output_root / "reconstruction_summary.json").open("w", encoding="utf-8") as f:
        json.dump(global_summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("DONE")
    print(f"Results saved under: {args.output_root}")
    print("Please send the complete console output, both *_comparison_vmax5.png files,")
    print("both *_mlat_mlt_comparison.png files, and reconstruction_summary.json.")


if __name__ == "__main__":
    main()
