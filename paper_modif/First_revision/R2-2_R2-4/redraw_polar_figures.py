#!/usr/bin/env python3
"""Redraw manuscript Figures 4 and 5 using the saved Polar reconstruction."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.image import imread
from scipy import ndimage

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]

DEFAULT_POLAR_DATA = Path("/home/docker/data/private/AuroraData/real_aurora_data_polar/1996/resampled_5min_1996_0405.npy")
DEFAULT_REPAIRED_DATA = Path("/home/docker/code/Aurora_DDPM/reasult/polar_res/new_res/repaired_polar_unetV3_ckptv2.npy")
DEFAULT_OVATION_DATA = Path("/home/docker/data/private/AuroraData/generated_aurora_data/1996_omni_aurora/aurora_img_19960401.npy")
DEFAULT_OMNI_DATA = Path("/home/docker/data/private/AuroraData/omni_real_data/omni_5min/1996/omni_19960401_5min.npy")
DEFAULT_BACKGROUND = Path("/home/docker/data/private/AuroraData/background_img/natural-earth-1_large2048px.png")
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "paper_modif/R2-2_R2-4/polar_redraw_results"

EVENTS = {"Figure4": datetime(1996, 4, 3, 2, 15, 0), "Figure5": datetime(1996, 4, 1, 8, 40, 0)}
MLAT_1D = np.linspace(50.0, 90.0, 80)
MLT_1D = np.linspace(0.0, 24.0, 96, endpoint=False)
ORTHO_LON = 110.0
OBS_SUPPORT_THRESHOLD = 1.0
DEFAULT_DISPLAY_FLOOR = 0.15
DEFAULT_MIN_COMPONENT_SIZE = 20
DEFAULT_VMAX = 5.0
DEFAULT_MAX_DELTA_SECONDS = 150.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--polar-data", type=Path, default=DEFAULT_POLAR_DATA)
    p.add_argument("--repaired-data", type=Path, default=DEFAULT_REPAIRED_DATA)
    p.add_argument("--ovation-data", type=Path, default=DEFAULT_OVATION_DATA)
    p.add_argument("--omni-data", type=Path, default=DEFAULT_OMNI_DATA)
    p.add_argument("--background", type=Path, default=DEFAULT_BACKGROUND)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--vmax", type=float, default=DEFAULT_VMAX)
    p.add_argument(
        "--display-floor",
        type=float,
        default=DEFAULT_DISPLAY_FLOOR,
        help=(
            "Display-only flux floor for reconstruction and OVATION panels. "
            "Values below this threshold are transparent; source arrays are unchanged."
        ),
    )
    p.add_argument(
        "--min-component-size",
        type=int,
        default=DEFAULT_MIN_COMPONENT_SIZE,
        help=(
            "Display-only minimum connected-component size for reconstruction "
            "and OVATION panels. Smaller regions are hidden."
        ),
    )
    p.add_argument("--max-delta-seconds", type=float, default=DEFAULT_MAX_DELTA_SECONDS)
    return p.parse_args()


def find_nearest_time_index(times: np.ndarray, target: datetime) -> Tuple[int, datetime, float]:
    times_ns = pd.to_datetime(times).values.astype("datetime64[ns]")
    target64 = np.datetime64(target, "ns")
    delta_ns = np.abs(times_ns.astype("int64") - target64.astype("int64"))
    idx = int(np.argmin(delta_ns))
    matched = pd.Timestamp(times_ns[idx]).to_pydatetime()
    return idx, matched, float(abs((matched - target).total_seconds()))


def extract_structured_image(data: np.ndarray, idx: int, field: str) -> np.ndarray:
    names = data.dtype.names
    if names is None or field not in names:
        raise KeyError(f"Required field {field!r} not found in structured array fields {names}")
    image = np.asarray(data[field][idx], dtype=np.float32)
    if image.shape != (80, 96):
        raise ValueError(f"Expected {field} image shape (80, 96), got {image.shape} at row {idx}.")
    return image


def validate_panel_shapes(observation: np.ndarray, reconstruction: np.ndarray, ovation: np.ndarray) -> None:
    shapes = (observation.shape, reconstruction.shape, ovation.shape)
    if len(set(shapes)) != 1:
        raise ValueError(f"Panel shape mismatch: observation={shapes[0]}, reconstruction={shapes[1]}, ovation={shapes[2]}")
    if observation.shape != (80, 96):
        raise ValueError(f"Expected panel shape (80, 96), got {observation.shape}")


def prepare_observation_for_plot(flux: np.ndarray, support_threshold: float = OBS_SUPPORT_THRESHOLD) -> np.ndarray:
    out = np.asarray(flux, dtype=np.float32).copy()
    out[(~np.isfinite(out)) | (out < support_threshold)] = np.nan
    return out


def remove_small_connected_components(
    flux: np.ndarray,
    min_component_size: int,
) -> np.ndarray:
    """Hide small isolated finite regions in a display-only flux array."""
    out = np.asarray(flux, dtype=np.float32).copy()
    if min_component_size <= 1:
        return out
    valid = np.isfinite(out)
    if not np.any(valid):
        return out
    labels, n_labels = ndimage.label(valid, structure=np.ones((3, 3), dtype=np.int8))
    if n_labels == 0:
        return out
    counts = np.bincount(labels.ravel())
    keep = counts >= int(min_component_size)
    keep[0] = False
    out[~keep[labels]] = np.nan
    return out


def prepare_full_field_for_plot(
    flux: np.ndarray,
    display_floor: float = DEFAULT_DISPLAY_FLOOR,
    min_component_size: int = 0,
) -> np.ndarray:
    """Return a display-only copy with negligible flux and speckles transparent."""
    if display_floor < 0:
        raise ValueError("display_floor must be non-negative")
    out = np.asarray(flux, dtype=np.float32).copy()
    out[(~np.isfinite(out)) | (out < display_floor)] = np.nan
    return remove_small_connected_components(out, min_component_size=min_component_size)


def panel_statistics(flux: np.ndarray, display_floor: float) -> Dict[str, float]:
    """Summarize raw panel values and the fraction hidden by the display floor."""
    x = np.asarray(flux, dtype=np.float64)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return {
            "n_finite": 0,
            "min": float("nan"),
            "median": float("nan"),
            "p90": float("nan"),
            "p99": float("nan"),
            "max": float("nan"),
            "fraction_below_display_floor": float("nan"),
        }
    return {
        "n_finite": int(finite.size),
        "min": float(np.min(finite)),
        "median": float(np.median(finite)),
        "p90": float(np.percentile(finite, 90.0)),
        "p99": float(np.percentile(finite, 99.0)),
        "max": float(np.max(finite)),
        "fraction_below_display_floor": float(np.mean(finite < display_floor)),
    }


def match_event(target: datetime, polar_data: np.ndarray, repaired_data: np.ndarray,
                omni_data: np.ndarray, ovation_data: np.ndarray,
                max_delta_seconds: float = DEFAULT_MAX_DELTA_SECONDS) -> Dict[str, Any]:
    for name, arr in (("polar_data", polar_data), ("repaired_data", repaired_data), ("omni_data", omni_data)):
        if arr.dtype.names is None or "utc" not in arr.dtype.names:
            raise KeyError(f"{name} must be a structured array containing a 'utc' field")
    p_idx, p_time, p_delta = find_nearest_time_index(polar_data["utc"], target)
    r_idx, r_time, r_delta = find_nearest_time_index(repaired_data["utc"], target)
    o_idx, o_time, o_delta = find_nearest_time_index(omni_data["utc"], target)
    too_far = {k: v for k, v in {"Polar/UVI": p_delta, "reconstruction": r_delta, "OVATION/OMNI": o_delta}.items() if v > max_delta_seconds}
    if too_far:
        raise ValueError(f"Timestamp match exceeds {max_delta_seconds:.1f} s for {target}: {too_far}")
    if ovation_data.ndim != 3 or ovation_data.shape[1:] != (80, 96):
        raise ValueError(f"OVATION array must have shape (N, 80, 96); got {ovation_data.shape}")
    if o_idx >= len(ovation_data):
        raise IndexError(f"Matched OMNI index {o_idx} exceeds OVATION length {len(ovation_data)}")
    observation = extract_structured_image(polar_data, p_idx, "aurora_image")
    reconstruction = extract_structured_image(repaired_data, r_idx, "image")
    ovation = np.asarray(ovation_data[o_idx], dtype=np.float32)
    validate_panel_shapes(observation, reconstruction, ovation)
    return {"target_time": target, "observation": observation, "reconstruction": reconstruction, "ovation": ovation,
            "polar_index": p_idx, "repaired_index": r_idx, "ovation_index": o_idx,
            "polar_time": p_time, "repaired_time": r_time, "ovation_time": o_time,
            "polar_delta_seconds": p_delta, "repaired_delta_seconds": r_delta, "ovation_delta_seconds": o_delta}


def make_aurora_cmap() -> LinearSegmentedColormap:
    colors = [(0.0, 0.2, 0.0), (0.0, 0.5, 0.0), (0.0, 0.8, 0.0), (0.5, 1.0, 0.0),
              (1.0, 1.0, 0.0), (1.0, 0.6, 0.0), (1.0, 0.3, 0.0), (1.0, 0.0, 0.0)]
    cmap = LinearSegmentedColormap.from_list("aurora", colors, N=256)
    cmap.set_bad(color=(0.0, 0.0, 0.0, 0.0))
    return cmap


AURORA_CMAP = make_aurora_cmap()


def magnetic_grid_to_geographic(timestamp: datetime) -> Tuple[np.ndarray, np.ndarray]:
    import aacgmv2
    _, mlat_grid = np.meshgrid(MLT_1D, MLAT_1D)
    mlon_row = np.asarray(aacgmv2.convert_mlt(MLT_1D, timestamp, m2a=True))
    mlon = np.tile(mlon_row[None, :], (len(MLAT_1D), 1))
    glat, glon, _ = aacgmv2.convert_latlon_arr(mlat_grid.reshape(-1), mlon.reshape(-1), 100, timestamp, method_code="A2G")
    return np.asarray(glat).reshape(mlat_grid.shape), np.asarray(glon).reshape(mlat_grid.shape)


def draw_background(ax, timestamp: datetime, background_path: Path) -> None:
    import cartopy.crs as ccrs
    from cartopy.feature.nightshade import Nightshade
    if background_path.exists():
        ax.imshow(imread(str(background_path)), origin="upper", transform=ccrs.PlateCarree(), extent=[-180, 180, -90, 90], zorder=0)
    else:
        ax.stock_img()
    ax.gridlines(linestyle="dashed", alpha=0.3, color="white")
    ax.coastlines("50m", color="white", alpha=0.5, linewidth=0.5)
    ax.add_feature(Nightshade(timestamp, alpha=0.4))
    ax.set_facecolor("black")


def plot_flux_on_axis(ax, flux: np.ndarray, glat: np.ndarray, glon: np.ndarray, vmax: float):
    import cartopy.crs as ccrs
    masked = np.ma.masked_invalid(flux)
    return ax.pcolormesh(
        glon,
        glat,
        masked,
        transform=ccrs.PlateCarree(),
        shading="nearest",
        cmap=AURORA_CMAP,
        vmin=0.0,
        vmax=vmax,
        zorder=3,
    )


def plot_event(
    case_name: str,
    case: Dict[str, Any],
    output_path: Path,
    background_path: Path,
    vmax: float,
    display_floor: float,
    min_component_size: int,
) -> Dict[str, Dict[str, float]]:
    import cartopy.crs as ccrs
    timestamp = case["target_time"]
    glat, glon = magnetic_grid_to_geographic(timestamp)

    panels = [
        prepare_observation_for_plot(case["observation"]),
        prepare_full_field_for_plot(
            case["reconstruction"],
            display_floor=display_floor,
            min_component_size=min_component_size,
        ),
        prepare_full_field_for_plot(
            case["ovation"],
            display_floor=display_floor,
            min_component_size=min_component_size,
        ),
    ]
    titles = ["(a) Polar/UVI-derived energy flux", "(b) Reconstruction", "(c) OVATION Prime"]

    fig = plt.figure(figsize=(24.0, 9.0), dpi=150)
    fig.patch.set_facecolor("black")
    fig.subplots_adjust(top=0.82, bottom=0.14, left=0.025, right=0.975, wspace=0.08)
    projection = ccrs.Orthographic(ORTHO_LON, 90.0)
    artist = None
    for i, (panel, title) in enumerate(zip(panels, titles), start=1):
        ax = fig.add_subplot(1, 3, i, projection=projection)
        draw_background(ax, timestamp, background_path)
        artist = plot_flux_on_axis(ax, panel, glat, glon, vmax=vmax)
        ax.set_title(title, color="white", fontsize=17, fontweight="bold", pad=10)

    fig.suptitle(
        f"Polar/UVI -- {timestamp.strftime('%-d %B %Y -- %H:%M UT')}",
        color="white",
        fontsize=20,
        fontweight="bold",
        y=0.985,
    )
    cbar_ax = fig.add_axes([0.34, 0.075, 0.32, 0.025])
    cbar = fig.colorbar(artist, cax=cbar_ax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=12, colors="white")
    cbar.set_label(r"Auroral Electron Energy Flux (erg cm$^{-2}$ s$^{-1}$)", color="white", fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, facecolor="black", bbox_inches="tight")
    plt.close(fig)

    return {
        "observation": panel_statistics(case["observation"], OBS_SUPPORT_THRESHOLD),
        "reconstruction": panel_statistics(case["reconstruction"], display_floor),
        "ovation": panel_statistics(case["ovation"], display_floor),
    }


def load_inputs(args: argparse.Namespace):
    for label, path in {"Polar/UVI": args.polar_data, "saved reconstruction": args.repaired_data, "OVATION": args.ovation_data, "OMNI": args.omni_data}.items():
        if not path.exists():
            raise FileNotFoundError(f"{label} file not found: {path}")
    return np.load(args.polar_data, allow_pickle=True), np.load(args.repaired_data, allow_pickle=True), np.load(args.ovation_data, allow_pickle=False), np.load(args.omni_data, allow_pickle=True)


def main() -> None:
    args = parse_args()
    polar_data, repaired_data, ovation_data, omni_data = load_inputs(args)
    args.output_root.mkdir(parents=True, exist_ok=True)
    audit: Dict[str, Any] = {
        "polar_data": str(args.polar_data),
        "repaired_data": str(args.repaired_data),
        "ovation_data": str(args.ovation_data),
        "omni_data": str(args.omni_data),
        "vmax": args.vmax,
        "observation_support_threshold": OBS_SUPPORT_THRESHOLD,
        "display_floor": args.display_floor,
        "min_component_size": args.min_component_size,
        "events": {},
    }
    print("=" * 88)
    print("POLAR/UVI FIGURE 4/5 REDRAW FROM SAVED RECONSTRUCTION")
    print("=" * 88)
    print(f"Display floor (reconstruction/OVATION only): {args.display_floor:g}")
    print(f"Minimum connected-component size: {args.min_component_size}")
    for case_name, target in EVENTS.items():
        case = match_event(target, polar_data, repaired_data, omni_data, ovation_data, args.max_delta_seconds)
        out = args.output_root / f"{case_name}_Polar_UVI_manuscript.png"
        stats = plot_event(
            case_name, case, out, args.background, args.vmax,
            args.display_floor, args.min_component_size,
        )
        audit["events"][case_name] = {
            "target_time": target.isoformat(),
            "polar_time": case["polar_time"].isoformat(),
            "repaired_time": case["repaired_time"].isoformat(),
            "ovation_time": case["ovation_time"].isoformat(),
            "polar_index": int(case["polar_index"]),
            "repaired_index": int(case["repaired_index"]),
            "ovation_index": int(case["ovation_index"]),
            "polar_delta_seconds": float(case["polar_delta_seconds"]),
            "repaired_delta_seconds": float(case["repaired_delta_seconds"]),
            "ovation_delta_seconds": float(case["ovation_delta_seconds"]),
            "statistics": stats,
            "output": str(out),
        }
        print(f"{case_name}: {target} -> {out}")
        print(f"  matched Polar={case['polar_time']} recon={case['repaired_time']} OVATION={case['ovation_time']}")
        for panel_name in ("observation", "reconstruction", "ovation"):
            s = stats[panel_name]
            print(
                f"  {panel_name:14s}: min={s['min']:.3f} median={s['median']:.3f} "
                f"p90={s['p90']:.3f} p99={s['p99']:.3f} max={s['max']:.3f} "
                f"below-floor={s['fraction_below_display_floor']:.3f}"
            )
    audit_path = args.output_root / "polar_redraw_audit.json"
    with audit_path.open("w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    print(f"Audit: {audit_path}")


if __name__ == "__main__":
    main()
