#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Diagnostic SSUSI reconstruction runner for Reviewer-2 Comment 3.

This script performs time-aligned SSUSI reconstruction diagnostics and also
writes the final three-panel manuscript figures.  The manuscript figures use:

(a) swath-aware SSUSI observation,
(b) interval-mean-solar-wind conditional DDPM reconstruction,
(c) interval-mean OVATION Prime.

No single midpoint timestamp is shown in the final figure title because each
SSUSI Auroral EDR map represents an orbital acquisition interval.

Run from repository root:
    python paper_modif/R2-1_R2-3/diagnose_ssusi_reconstruction.py
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch

try:
    import torch_npu  # noqa: F401
except Exception:
    torch_npu = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.image import imread

import scipy.ndimage

import cartopy.crs as ccrs
from cartopy.feature.nightshade import Nightshade
import aacgmv2

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.unet import UNet
from models.ddpm import DDPM


# =============================================================================
# Defaults
# =============================================================================
DEFAULT_PREPARED_ROOT = REPO_ROOT / "paper_modif/R2-1_R2-3/swath_prepared_cases"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "paper_modif/R2-1_R2-3/swath_reconstruction_diagnostics"
DEFAULT_CHECKPOINT = REPO_ROOT / "ckpt/cond/aurora_diff_best.pth"
DEFAULT_NORM_PARAMS = REPO_ROOT / "ckpt/cond/norm_params.pkl"
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


# =============================================================================
# Utilities
# =============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnostic SSUSI reconstruction")
    p.add_argument("--prepared-root", type=Path, default=DEFAULT_PREPARED_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--norm-params", type=Path, default=DEFAULT_NORM_PARAMS)
    p.add_argument("--ovation", type=Path, default=DEFAULT_OVATION)
    p.add_argument("--omni", type=Path, default=DEFAULT_OMNI)
    p.add_argument("--background", type=Path, default=DEFAULT_BACKGROUND)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--display-threshold", type=float, default=0.1)
    p.add_argument("--fixed-vmax", type=float, default=5.0)
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


def as_datetime(value: Any) -> datetime:
    return pd.Timestamp(value).to_pydatetime()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def load_norm_params(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        params = pickle.load(f)
    return params


def normalize_aurora_physical(flux: np.ndarray, norm: Dict[str, Any]) -> np.ndarray:
    x = np.clip(np.asarray(flux, dtype=np.float32), 0.0, None)
    log_x = np.log1p(x)
    y = (log_x - float(norm["aurora"]["min"])) / float(norm["aurora"]["range"])
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def normalize_aurora_raw_no_clip(flux: np.ndarray, norm: Dict[str, Any]) -> np.ndarray:
    x = np.clip(np.asarray(flux, dtype=np.float32), 0.0, None)
    log_x = np.log1p(x)
    y = (log_x - float(norm["aurora"]["min"])) / float(norm["aurora"]["range"])
    return y.astype(np.float32)


def denormalize_aurora(y: np.ndarray, norm: Dict[str, Any]) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    log_x = y * float(norm["aurora"]["range"]) + float(norm["aurora"]["min"])
    return np.expm1(log_x).astype(np.float32)


def normalize_solar(raw: np.ndarray, norm: Dict[str, Any]) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float32)
    minimum = np.asarray(norm["omni"]["min"], dtype=np.float32)
    value_range = np.asarray(norm["omni"]["range"], dtype=np.float32)
    y = (raw - minimum) / value_range
    return np.clip(y, 0.0, 1.0).astype(np.float32)


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


def load_case(prepared_root: Path, case: str):
    case_dir = prepared_root / case
    npz_path = case_dir / f"{case}_swath_case.npz"
    meta_path = case_dir / f"{case}_metadata.json"
    data = dict(np.load(npz_path, allow_pickle=False))
    meta = load_json(meta_path)
    return data, meta


def load_omni(path: Path) -> np.ndarray:
    return np.load(path, allow_pickle=True)


def extract_solar_row(omni: np.ndarray, idx: int) -> np.ndarray:
    names = omni.dtype.names
    row = []
    for field in SOLAR_FIELDS:
        if field in names:
            row.append(float(np.asarray(omni[field][idx]).reshape(-1)[0]))
        else:
            raise KeyError(f"Missing OMNI field {field} in {names}")
    raw = np.asarray(row, dtype=np.float32)
    if not np.all(np.isfinite(raw)):
        raise ValueError(f"Non-finite solar row at idx={idx}: {raw}")
    return raw


def structured_times_to_datetime(omni: np.ndarray) -> np.ndarray:
    names = omni.dtype.names
    if "utc" not in names:
        raise KeyError(f"OMNI structured array must contain 'utc'; got {names}")
    return pd.to_datetime(omni["utc"]).to_pydatetime()


def nearest_time_index(times: np.ndarray, target: datetime) -> Tuple[int, datetime, float]:
    t64 = pd.to_datetime(times).values.astype("datetime64[ns]")
    target64 = np.datetime64(target, "ns")
    delta = np.abs(t64.astype("int64") - target64.astype("int64"))
    idx = int(np.argmin(delta))
    matched = pd.Timestamp(t64[idx]).to_pydatetime()
    offset_s = abs((matched - target).total_seconds())
    return idx, matched, float(offset_s)


def interval_mean_solar(omni: np.ndarray, start: datetime, end: datetime) -> Tuple[np.ndarray, Dict[str, Any]]:
    times = pd.to_datetime(omni["utc"]).to_pydatetime()
    mask = np.array([(t >= start and t <= end) for t in times], dtype=bool)
    if not np.any(mask):
        raise RuntimeError(f"No OMNI rows found inside interval {start} -> {end}")
    row_list = []
    for f in SOLAR_FIELDS:
        row_list.append(np.asarray(omni[f][mask], dtype=np.float32))
    stacked = np.vstack(row_list).T
    mean_vec = np.nanmean(stacked, axis=0).astype(np.float32)
    info = {
        "n_rows": int(mask.sum()),
        "start": start.isoformat(),
        "end": end.isoformat(),
    }
    return mean_vec, info


def load_ovation(path: Path) -> np.ndarray:
    arr = np.load(path, allow_pickle=False)
    if arr.ndim != 3:
        raise ValueError(f"Unexpected OVATION array shape: {arr.shape}")
    return arr.astype(np.float32)


def ovation_month_start_from_path(path: Path) -> datetime:
    stem = path.stem
    token = stem.split("_")[-1]
    if len(token) != 8 or not token.isdigit():
        raise ValueError(f"Cannot parse OVATION month start from {path.name}")
    return datetime.strptime(token, "%Y%m%d")


def ovation_time_to_index(ts: datetime, ovation_start: datetime, ovation_len: int) -> int:
    delta_min = (ts - ovation_start).total_seconds() / 60.0
    idx = int(round(delta_min))
    return max(0, min(ovation_len - 1, idx))


def ovation_interval_indices(start: datetime, end: datetime, ovation_start: datetime, ovation_len: int):
    idx0 = int(np.floor((start - ovation_start).total_seconds() / 60.0))
    idx1 = int(np.ceil((end - ovation_start).total_seconds() / 60.0))
    idx0 = max(0, idx0)
    idx1 = min(ovation_len - 1, idx1)
    if idx1 < idx0:
        idx1 = idx0
    return idx0, idx1


def build_model(checkpoint_path: Path, device: torch.device) -> DDPM:
    unet = UNet(1, 1)
    ddpm = DDPM(unet, num_train_steps=1000, schedule="cosine")
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state = checkpoint.get("model_state_dict", checkpoint)
    ddpm.load_state_dict(state, strict=True)
    ddpm.to(device)
    ddpm.eval()
    return ddpm


def reconstruct_case(
    ddpm: DDPM,
    device: torch.device,
    flux_grid: np.ndarray,
    obs_mask: np.ndarray,
    solar_norm: np.ndarray,
    norm: Dict[str, Any],
    seed: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    input_physical = np.where(obs_mask, flux_grid, 0.0).astype(np.float32)
    input_norm = normalize_aurora_physical(input_physical, norm)

    image = torch.from_numpy(input_norm[None, None]).to(device=device, dtype=torch.float32)
    mask = torch.from_numpy(obs_mask.astype(np.float32)[None, None]).to(device=device, dtype=torch.float32)
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
    out_norm = np.clip(out_norm_raw, 0.0, 1.0)
    repaired = denormalize_aurora(out_norm, norm)
    repaired = np.clip(repaired, 0.0, None)

    repaired[obs_mask.astype(bool)] = flux_grid[obs_mask.astype(bool)]

    diag = {
        "output_norm_min_before_clip": float(np.nanmin(out_norm_raw)),
        "output_norm_max_before_clip": float(np.nanmax(out_norm_raw)),
        "observed_exact_max_abs_error": float(
            np.max(np.abs(repaired[obs_mask.astype(bool)] - flux_grid[obs_mask.astype(bool)]))
        ),
        "generated_missing_mean": float(np.nanmean(repaired[~obs_mask.astype(bool)])),
        "generated_missing_max": float(np.nanmax(repaired[~obs_mask.astype(bool)])),
    }
    return repaired.astype(np.float32), diag


def observed_saturation_stats(flux_grid: np.ndarray, obs_mask: np.ndarray, norm: Dict[str, Any]) -> Dict[str, float]:
    obs = obs_mask.astype(bool)
    raw_norm = normalize_aurora_raw_no_clip(flux_grid[obs], norm)
    if raw_norm.size == 0:
        return {
            "saturation_fraction_gt1": 0.0,
            "fraction_lt0": 0.0,
            "raw_norm_max": 0.0,
            "raw_norm_p99": 0.0,
        }
    return {
        "saturation_fraction_gt1": float(np.mean(raw_norm > 1.0)),
        "fraction_lt0": float(np.mean(raw_norm < 0.0)),
        "raw_norm_max": float(np.max(raw_norm)),
        "raw_norm_p99": float(np.quantile(raw_norm, 0.99)),
    }


def boundary_continuity_metric(reference_obs: np.ndarray, recon: np.ndarray, obs_mask: np.ndarray) -> Dict[str, float]:
    obs = obs_mask.astype(bool)
    missing = ~obs
    dilated_obs = scipy.ndimage.binary_dilation(obs, structure=np.ones((3, 3), dtype=bool))
    missing_ring = missing & dilated_obs

    rows, cols = np.where(missing_ring)
    diffs = []
    for r, c in zip(rows, cols):
        r0 = max(0, r - 1)
        r1 = min(obs.shape[0], r + 2)
        c0 = max(0, c - 1)
        c1 = min(obs.shape[1], c + 2)
        local_obs = obs[r0:r1, c0:c1]
        local_ref = reference_obs[r0:r1, c0:c1]
        if np.any(local_obs):
            ref_mean = float(np.mean(local_ref[local_obs]))
            diffs.append(abs(float(recon[r, c]) - ref_mean))

    if len(diffs) == 0:
        return {
            "ring_count": 0,
            "boundary_absdiff_mean": 0.0,
            "boundary_absdiff_median": 0.0,
            "boundary_absdiff_p90": 0.0,
        }

    diffs = np.asarray(diffs, dtype=np.float32)
    return {
        "ring_count": int(len(diffs)),
        "boundary_absdiff_mean": float(np.mean(diffs)),
        "boundary_absdiff_median": float(np.median(diffs)),
        "boundary_absdiff_p90": float(np.quantile(diffs, 0.9)),
    }


# =============================================================================
# Plotting
# =============================================================================
def magnetic_grid_to_geographic(timestamp: datetime):
    _, mlat_grid = np.meshgrid(MLT_1D, MLAT_1D)
    mlon_row = np.asarray(aacgmv2.convert_mlt(MLT_1D, timestamp, m2a=True))
    mlon = np.tile(mlon_row[None, :], (len(MLAT_1D), 1))
    mlat_flat = mlat_grid.reshape(-1)
    mlon_flat = mlon.reshape(-1)
    glat, glon, _ = aacgmv2.convert_latlon_arr(
        mlat_flat, mlon_flat, 100, timestamp, method_code="A2G"
    )
    return np.asarray(glat).reshape(mlat_grid.shape), np.asarray(glon).reshape(mlat_grid.shape)


def draw_background(ax, timestamp: datetime, background_path: Path):
    if background_path.exists():
        ax.imshow(
            imread(str(background_path)),
            origin="upper",
            transform=ccrs.PlateCarree(),
            extent=[-180, 180, -90, 90],
            zorder=0,
        )
    else:
        ax.stock_img()
    ax.gridlines(linestyle="dashed", alpha=0.3, color="white")
    ax.coastlines("50m", color="white", alpha=0.5, linewidth=0.5)
    ax.add_feature(Nightshade(timestamp, alpha=0.4))
    ax.set_facecolor("black")


def threshold_for_display(arr: np.ndarray, threshold: float, keep_nan=True) -> np.ndarray:
    out = np.array(arr, dtype=np.float32, copy=True)
    if keep_nan:
        mask_nan = np.isnan(out)
    out[out < threshold] = np.nan
    if keep_nan:
        out[mask_nan] = np.nan
    return out


def plot_flux_on_axis(ax, flux: np.ndarray, glat: np.ndarray, glon: np.ndarray, vmin: float, vmax: float):
    masked = np.ma.masked_invalid(flux)
    return ax.pcolormesh(
        glon,
        glat,
        masked,
        transform=ccrs.PlateCarree(),
        shading="nearest",
        cmap=AURORA_CMAP,
        vmin=vmin,
        vmax=vmax,
        zorder=3,
    )


def plot_orthographic_diagnostic(
    out_path: Path,
    case_name: str,
    timestamp: datetime,
    obs_panel: np.ndarray,
    recon_mid: np.ndarray,
    ova_mid: np.ndarray,
    recon_int: np.ndarray,
    ova_int: np.ndarray,
    background_path: Path,
    vmax: float,
    threshold: float,
):
    glat, glon = magnetic_grid_to_geographic(timestamp)

    fig = plt.figure(figsize=(18, 10), dpi=150)
    fig.set_facecolor("black")

    titles = [
        "(a) SSUSI observation",
        "(b) Reconstruction (midpoint solar)",
        "(c) OVATION Prime (midpoint)",
        "(d) SSUSI observation",
        "(e) Reconstruction (interval-mean solar)",
        "(f) OVATION Prime (interval mean)",
    ]
    panels = [obs_panel, recon_mid, ova_mid, obs_panel, recon_int, ova_int]

    artists = []
    for i in range(6):
        ax = fig.add_subplot(2, 3, i + 1, projection=ccrs.Orthographic(ORTHO_LON, 90.0))
        draw_background(ax, timestamp, background_path)
        panel_disp = threshold_for_display(panels[i], threshold)
        artist = plot_flux_on_axis(ax, panel_disp, glat, glon, 0.0, vmax)
        ax.set_title(titles[i], color="white", fontsize=13, weight="bold", pad=8)
        artists.append(artist)

    fig.suptitle(
        f"{case_name}: diagnostic SSUSI reconstruction",
        color="white",
        fontsize=20,
        weight="bold",
        y=0.97,
    )
    cax = fig.add_axes([0.27, 0.05, 0.46, 0.02])
    cbar = plt.colorbar(artists[0], cax=cax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=12, colors="white")
    cbar.set_label(
        r"Auroral Electron Energy Flux (erg cm$^{-2}$ s$^{-1}$)",
        color="white",
        fontsize=14,
    )

    plt.savefig(out_path, dpi=150, facecolor="black", bbox_inches="tight")
    plt.close(fig)


def plot_mlat_mlt_diagnostic(
    out_path: Path,
    case_name: str,
    obs_flux: np.ndarray,
    obs_mask: np.ndarray,
    recon_mid: np.ndarray,
    ova_mid: np.ndarray,
    recon_int: np.ndarray,
    ova_int: np.ndarray,
    vmax: float,
):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=150, sharex=True, sharey=True)
    extent = [0.0, 24.0, 50.0, 90.0]
    obs_panel = np.where(obs_mask.astype(bool), obs_flux, np.nan)

    titles = [
        "(a) SSUSI observation",
        "(b) Reconstruction (midpoint solar)",
        "(c) OVATION Prime (midpoint)",
        "(d) SSUSI observation",
        "(e) Reconstruction (interval-mean solar)",
        "(f) OVATION Prime (interval mean)",
    ]
    panels = [obs_panel, recon_mid, ova_mid, obs_panel, recon_int, ova_int]

    ims = []
    for ax, title, panel in zip(axes.flat, titles, panels):
        im = ax.imshow(
            panel,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap=AURORA_CMAP,
            vmin=0.0,
            vmax=vmax,
        )
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("MLT [h]")
        ax.set_ylabel("MLAT [deg]")
        ims.append(im)

    fig.suptitle(f"{case_name}: diagnostic MLAT-MLT comparison", fontsize=18, y=0.95)
    cax = fig.add_axes([0.15, 0.05, 0.7, 0.025])
    cbar = plt.colorbar(ims[0], cax=cax, orientation="horizontal")
    cbar.set_label(r"Auroral Electron Energy Flux (erg cm$^{-2}$ s$^{-1}$)", fontsize=14)

    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def format_manuscript_title(case_name: str, meta: Dict[str, Any], edr_mid: datetime) -> str:
    """Return a title that does not imply an instantaneous observation."""
    orbit = meta.get("netcdf_metadata", {}).get(
        "STARTING_ORBIT_NUMBER", meta.get("orbit", "unknown")
    )
    orbit_str = str(orbit).strip()
    if orbit_str.isdigit():
        orbit_str = orbit_str.zfill(5)
    date_str = edr_mid.strftime("%-d %B %Y")
    return f"DMSP F16/SSUSI -- Orbit {orbit_str} -- {date_str}"


def plot_manuscript_final(
    out_path: Path,
    timestamp: datetime,
    title: str,
    obs_panel: np.ndarray,
    recon_interval: np.ndarray,
    ovation_interval: np.ndarray,
    background_path: Path,
    vmax: float = 5.0,
    threshold: float = 0.1,
):
    """Write the final three-panel figure used in the revised manuscript."""
    glat, glon = magnetic_grid_to_geographic(timestamp)

    fig = plt.figure(figsize=(18, 6.6), dpi=300)
    fig.set_facecolor("black")

    titles = [
        "(a) SSUSI observation",
        "(b) Reconstruction",
        "(c) OVATION Prime",
    ]
    panels = [obs_panel, recon_interval, ovation_interval]

    artists = []
    for i, (panel_title, panel) in enumerate(zip(titles, panels), start=1):
        ax = fig.add_subplot(1, 3, i, projection=ccrs.Orthographic(ORTHO_LON, 90.0))
        draw_background(ax, timestamp, background_path)
        panel_disp = threshold_for_display(panel, threshold)
        artist = plot_flux_on_axis(ax, panel_disp, glat, glon, 0.0, vmax)
        ax.set_title(panel_title, color="white", fontsize=16, weight="bold", pad=10)
        artists.append(artist)

    fig.suptitle(title, color="white", fontsize=20, weight="bold", y=0.96)

    cax = fig.add_axes([0.29, 0.07, 0.42, 0.025])
    cbar = plt.colorbar(artists[0], cax=cax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=12, colors="white")
    cbar.set_label(
        r"Auroral Electron Energy Flux (erg cm$^{-2}$ s$^{-1}$)",
        color="white",
        fontsize=14,
    )

    plt.subplots_adjust(left=0.015, right=0.985, top=0.86, bottom=0.16, wspace=0.08)
    plt.savefig(out_path, dpi=300, facecolor="black", bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================
def main():
    args = parse_args()
    os.makedirs(args.output_root, exist_ok=True)

    device = choose_device(args.device)
    norm = load_norm_params(args.norm_params)
    ddpm = build_model(args.checkpoint, device=device)

    omni = load_omni(args.omni)
    omni_times = structured_times_to_datetime(omni)

    ovation = load_ovation(args.ovation)
    ovation_start = ovation_month_start_from_path(args.ovation)
    ovation_len = ovation.shape[0]

    print("=" * 80)
    print("DIAGNOSTIC SSUSI RECONSTRUCTION")
    print("=" * 80)
    print(f"Device:          {device}")
    print(f"Checkpoint:      {args.checkpoint}")
    print(f"Norm params:     {args.norm_params}")
    print(f"Prepared root:   {args.prepared_root}")
    print(f"OMNI:            {args.omni}")
    print(f"OVATION:         {args.ovation}")
    print(f"OVATION length:  {ovation_len}")
    print(f"OVATION start:   {ovation_start}")
    print(f"Output root:     {args.output_root}")
    print(f"Display thresh:  {args.display_threshold}")
    print(f"Fixed vmax:      {args.fixed_vmax}")

    all_summary = {}

    for i_case, case in enumerate(CASES):
        print("\n" + "-" * 80)
        print(case)

        data, meta = load_case(args.prepared_root, case)
        flux_grid = np.asarray(data["flux_grid"], dtype=np.float32)
        obs_mask = np.asarray(data["obs_mask"], dtype=np.uint8)

        edr_mid = as_datetime(meta["edr_midpoint"])
        acq_start = as_datetime(meta["acquisition_start"])
        acq_end = as_datetime(meta["acquisition_end"])
        orbit = meta.get("netcdf_metadata", {}).get("STARTING_ORBIT_NUMBER", meta.get("orbit", "unknown"))

        omni_mid_idx, omni_mid_time, omni_mid_offset_s = nearest_time_index(omni_times, edr_mid)
        solar_mid_raw = extract_solar_row(omni, omni_mid_idx)
        solar_mid_norm = normalize_solar(solar_mid_raw, norm)

        solar_int_raw, solar_int_info = interval_mean_solar(omni, acq_start, acq_end)
        solar_int_norm = normalize_solar(solar_int_raw, norm)

        ova_mid_idx = ovation_time_to_index(edr_mid, ovation_start, ovation_len)
        ova_mid = ovation[ova_mid_idx].astype(np.float32)

        ova_i0, ova_i1 = ovation_interval_indices(acq_start, acq_end, ovation_start, ovation_len)
        ova_int = np.mean(ovation[ova_i0:ova_i1 + 1], axis=0).astype(np.float32)

        recon_mid, recon_mid_diag = reconstruct_case(
            ddpm, device, flux_grid, obs_mask, solar_mid_norm, norm, seed=args.seed + i_case * 2
        )
        recon_int, recon_int_diag = reconstruct_case(
            ddpm, device, flux_grid, obs_mask, solar_int_norm, norm, seed=args.seed + i_case * 2 + 1
        )

        sat_stats = observed_saturation_stats(flux_grid, obs_mask, norm)
        boundary_mid = boundary_continuity_metric(flux_grid, recon_mid, obs_mask)
        boundary_int = boundary_continuity_metric(flux_grid, recon_int, obs_mask)

        obs_panel = np.where(obs_mask.astype(bool), flux_grid, np.nan)

        vmax_auto = float(np.quantile(
            np.concatenate([
                recon_mid[np.isfinite(recon_mid)].ravel(),
                recon_int[np.isfinite(recon_int)].ravel(),
                ova_mid[np.isfinite(ova_mid)].ravel(),
                ova_int[np.isfinite(ova_int)].ravel(),
                flux_grid[np.isfinite(flux_grid)].ravel(),
            ]),
            0.99,
        ))
        vmax_use = max(args.fixed_vmax, vmax_auto)

        case_dir = args.output_root / case
        os.makedirs(case_dir, exist_ok=True)

        np.savez_compressed(
            case_dir / f"{case}_diagnostic_arrays.npz",
            flux_grid=flux_grid,
            obs_mask=obs_mask,
            recon_mid=recon_mid,
            recon_int=recon_int,
            ovation_mid=ova_mid,
            ovation_interval_mean=ova_int,
            mlat_1d=MLAT_1D,
            mlt_1d=MLT_1D,
        )

        plot_orthographic_diagnostic(
            case_dir / f"{case}_orthographic_diagnostic.png",
            case,
            edr_mid,
            obs_panel,
            recon_mid,
            ova_mid,
            recon_int,
            ova_int,
            args.background,
            vmax_use,
            args.display_threshold,
        )

        plot_mlat_mlt_diagnostic(
            case_dir / f"{case}_mlat_mlt_diagnostic.png",
            case,
            flux_grid,
            obs_mask,
            recon_mid,
            ova_mid,
            recon_int,
            ova_int,
            vmax_use,
        )

        manuscript_title = format_manuscript_title(case, meta, edr_mid)
        manuscript_path = case_dir / f"{case}_manuscript_final.png"
        plot_manuscript_final(
            manuscript_path,
            edr_mid,
            manuscript_title,
            obs_panel,
            recon_int,
            ova_int,
            args.background,
            vmax=args.fixed_vmax,
            threshold=args.display_threshold,
        )

        summary = {
            "case": case,
            "edr_midpoint": edr_mid.isoformat(),
            "acquisition_start": acq_start.isoformat(),
            "acquisition_end": acq_end.isoformat(),
            "orbit": str(orbit),
            "observed_fraction": float(obs_mask.mean()),
            "omni_midpoint": {
                "index": int(omni_mid_idx),
                "time": omni_mid_time.isoformat(),
                "offset_seconds": float(omni_mid_offset_s),
                "raw": {k: float(v) for k, v in zip(SOLAR_FIELDS, solar_mid_raw)},
                "normalized": {k: float(v) for k, v in zip(SOLAR_FIELDS, solar_mid_norm)},
            },
            "omni_interval_mean": {
                "n_rows": int(solar_int_info["n_rows"]),
                "raw": {k: float(v) for k, v in zip(SOLAR_FIELDS, solar_int_raw)},
                "normalized": {k: float(v) for k, v in zip(SOLAR_FIELDS, solar_int_norm)},
            },
            "ovation_midpoint": {
                "index": int(ova_mid_idx),
                "time_assumed": (ovation_start + timedelta(minutes=int(ova_mid_idx))).isoformat(),
            },
            "ovation_interval_mean": {
                "start_index": int(ova_i0),
                "end_index": int(ova_i1),
                "n_frames": int(ova_i1 - ova_i0 + 1),
                "start_time_assumed": (ovation_start + timedelta(minutes=int(ova_i0))).isoformat(),
                "end_time_assumed": (ovation_start + timedelta(minutes=int(ova_i1))).isoformat(),
            },
            "observed_normalization_saturation": sat_stats,
            "reconstruction_midpoint": recon_mid_diag,
            "reconstruction_interval_mean": recon_int_diag,
            "boundary_midpoint": boundary_mid,
            "boundary_interval_mean": boundary_int,
            "display": {
                "diagnostic_vmax": float(vmax_use),
                "manuscript_vmax": float(args.fixed_vmax),
                "display_threshold": float(args.display_threshold),
            },
            "manuscript_figure": str(manuscript_path),
        }

        save_json(case_dir / f"{case}_diagnostic_summary.json", summary)
        all_summary[case] = summary

        print(f"EDR midpoint:         {edr_mid}")
        print(f"Acquisition interval: {acq_start} -> {acq_end}")
        print(f"Observed fraction:    {obs_mask.mean():.4f}")
        print(f"OMNI midpoint time:   {omni_mid_time} (offset {omni_mid_offset_s:.1f} s)")
        print(f"OMNI midpoint raw:    {dict(zip(SOLAR_FIELDS, [float(v) for v in solar_mid_raw]))}")
        print(f"OMNI interval mean:   {dict(zip(SOLAR_FIELDS, [float(v) for v in solar_int_raw]))}")
        print(f"OVATION midpoint idx: {ova_mid_idx}")
        print(f"OVATION interval idx: {ova_i0} -> {ova_i1} ({ova_i1 - ova_i0 + 1} frames)")
        print(
            "Observed saturation: "
            f"frac>1={sat_stats['saturation_fraction_gt1']:.4f}, "
            f"raw_norm_max={sat_stats['raw_norm_max']:.4f}, "
            f"p99={sat_stats['raw_norm_p99']:.4f}"
        )
        print(
            "Boundary midpoint:    "
            f"mean={boundary_mid['boundary_absdiff_mean']:.4f}, "
            f"median={boundary_mid['boundary_absdiff_median']:.4f}, "
            f"p90={boundary_mid['boundary_absdiff_p90']:.4f}"
        )
        print(
            "Boundary interval:    "
            f"mean={boundary_int['boundary_absdiff_mean']:.4f}, "
            f"median={boundary_int['boundary_absdiff_median']:.4f}, "
            f"p90={boundary_int['boundary_absdiff_p90']:.4f}"
        )
        print(
            "Observed exact err:   "
            f"mid={recon_mid_diag['observed_exact_max_abs_error']:.4e}, "
            f"int={recon_int_diag['observed_exact_max_abs_error']:.4e}"
        )
        print(
            "Generated missing max:"
            f" mid={recon_mid_diag['generated_missing_max']:.4f},"
            f" int={recon_int_diag['generated_missing_max']:.4f}"
        )
        print(f"Diagnostic vmax:      {vmax_use:.4f}")
        print(f"Manuscript vmax:      {args.fixed_vmax:.4f}")
        print(f"Manuscript figure:    {manuscript_path}")
        print(f"Outputs:              {case_dir}")

    save_json(args.output_root / "diagnostic_summary_all_cases.json", all_summary)

    print("\n" + "=" * 80)
    print("DONE")
    print(f"Results saved under: {args.output_root}")
    print("Final manuscript figures:")
    for case in CASES:
        print(f"  {args.output_root / case / (case + '_manuscript_final.png')}")


if __name__ == "__main__":
    main()
