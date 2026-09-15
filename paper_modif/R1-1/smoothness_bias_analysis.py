#!/usr/bin/env python3
"""Quantify OVATION-to-observation smoothness bias for Reviewer-1 Comment 1.

The analysis is deliberately observation-support aware:
- Gradient distributions use only nearest-neighbor edges whose two endpoints
  are directly supported by the SSUSI swath.
- PSD diagnostics use only continuous SSUSI-supported MLT segments and sample
  the time-aligned interval-mean OVATION field at exactly the same cells.
- No zero filling or interpolation across unobserved SSUSI sectors is used.

The script does not run the diffusion model. It complements the existing
real-SSUSI held-out benchmark by quantifying how much smoother the OVATION
training prior is than the real SSUSI morphology.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance, wilcoxon

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
REPO_ROOT = SCRIPT_PATH.parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from smoothness_bias_utils import (
    collect_matched_psd_segments,
    paired_bootstrap_mean_difference,
    summarize_distribution,
    supported_edge_gradients,
)

CASES = ("Figure6", "Figure7")
MLAT_1D = np.linspace(50.0, 90.0, 80)
MLT_1D = np.linspace(0.0, 24.0, 96, endpoint=False)

DEFAULT_PREPARED_ROOT = REPO_ROOT / "paper_modif/R2-1_R2-3/swath_prepared_cases"
DEFAULT_OVATION = Path(
    "/home/docker/data/private/AuroraData/generated_aurora_data/2005_omni_aurora/"
    "aurora_img_20050101.npy"
)
DEFAULT_HELDOUT = (
    REPO_ROOT
    / "paper_modif/R2-1_R2-3/ssusi_heldout_benchmark_results/metrics_per_holdout.csv"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "paper_modif/R1-1/r1_1_results"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared-root", type=Path, default=DEFAULT_PREPARED_ROOT)
    p.add_argument("--ovation", type=Path, default=DEFAULT_OVATION)
    p.add_argument("--heldout-metrics", type=Path, default=DEFAULT_HELDOUT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--band-min", type=float, default=60.0)
    p.add_argument("--band-max", type=float, default=80.0)
    p.add_argument("--min-segment-bins", type=int, default=12)
    p.add_argument("--signal-threshold", type=float, default=0.10)
    p.add_argument("--min-signal-fraction", type=float, default=0.10)
    p.add_argument("--high-frequency-cutoff", type=float, default=0.50,
                   help="PSD high-frequency cutoff in cycles per MLT hour")
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_case(prepared_root: Path, case: str):
    case_dir = prepared_root / case
    npz_path = case_dir / f"{case}_swath_case.npz"
    meta_path = case_dir / f"{case}_metadata.json"
    if not npz_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Prepared {case} inputs not found under {case_dir}. "
            "Run the swath-preparation workflow used for revised Figures 6/7 first."
        )
    data = dict(np.load(npz_path, allow_pickle=False))
    meta = load_json(meta_path)
    return data, meta


def ovation_month_start_from_path(path: Path) -> datetime:
    token = path.stem.split("_")[-1]
    if len(token) != 8 or not token.isdigit():
        raise ValueError(f"Cannot parse YYYYMMDD month start from {path.name}")
    return datetime.strptime(token, "%Y%m%d")


def ovation_interval_mean(
    ovation: np.ndarray, ovation_start: datetime, start: datetime, end: datetime
):
    i0 = int(np.floor((start - ovation_start).total_seconds() / 60.0))
    i1 = int(np.ceil((end - ovation_start).total_seconds() / 60.0))
    i0 = max(0, i0)
    i1 = min(len(ovation) - 1, i1)
    if i1 < i0:
        raise ValueError(f"Invalid OVATION interval {i0}..{i1}")
    return np.mean(ovation[i0:i1 + 1], axis=0).astype(np.float32), i0, i1


def finite_ratio(a: float, b: float) -> float:
    return float(a / b) if np.isfinite(a) and np.isfinite(b) and b != 0 else float("nan")


def safe_wilcoxon(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    valid = np.isfinite(aa) & np.isfinite(bb)
    aa, bb = aa[valid], bb[valid]
    if aa.size == 0 or np.allclose(aa, bb):
        return float("nan")
    try:
        return float(wilcoxon(aa, bb, alternative="two-sided").pvalue)
    except ValueError:
        return float("nan")


def compare_gradient_distributions(
    case: str,
    space: str,
    ssusi: np.ndarray,
    ovation: np.ndarray,
) -> Dict[str, Any]:
    s = summarize_distribution(ssusi)
    o = summarize_distribution(ovation)
    ks = ks_2samp(ssusi, ovation, alternative="two-sided", method="auto")
    return {
        "case": case,
        "space": space,
        "n_ssusi": int(s["n"]),
        "n_ovation": int(o["n"]),
        "mean_ratio_ssusi_over_ovation": finite_ratio(s["mean"], o["mean"]),
        "median_ratio_ssusi_over_ovation": finite_ratio(s["median"], o["median"]),
        "p90_ratio_ssusi_over_ovation": finite_ratio(s["p90"], o["p90"]),
        "p95_ratio_ssusi_over_ovation": finite_ratio(s["p95"], o["p95"]),
        "p99_ratio_ssusi_over_ovation": finite_ratio(s["p99"], o["p99"]),
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "wasserstein_distance": float(wasserstein_distance(ssusi, ovation)),
    }


def ecdf(values: np.ndarray):
    x = np.sort(np.asarray(values, dtype=float))
    x = x[np.isfinite(x)]
    y = np.arange(1, x.size + 1, dtype=float) / max(1, x.size)
    return x, y


def normalized_psd_on_grid(record: Dict[str, Any], prefix: str, grid: np.ndarray) -> np.ndarray:
    f = np.asarray(record[f"{prefix}_frequency"], dtype=float)
    p = np.asarray(record[f"{prefix}_psd"], dtype=float)
    total = float(np.sum(p))
    if total > 0:
        p = p / total
    out = np.full(grid.shape, np.nan, dtype=float)
    valid = (grid >= np.min(f)) & (grid <= np.max(f))
    if np.any(valid):
        out[valid] = np.interp(grid[valid], f, p)
    return out


def plot_summary(
    out_path: Path,
    ssusi_log_grad: np.ndarray,
    ovation_log_grad: np.ndarray,
    psd_records: List[Dict[str, Any]],
    high_freq_cutoff: float,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), dpi=220)

    for values, label in (
        (ssusi_log_grad, "SSUSI"),
        (ovation_log_grad, "OVATION Prime"),
    ):
        x, y = ecdf(values)
        axes[0].plot(x, y, linewidth=1.8, label=label)
    axes[0].set_xlabel(r"Nearest-neighbor $|\Delta\log(1+F)|$")
    axes[0].set_ylabel("Empirical CDF")
    axes[0].set_title("(a) Spatial-gradient distribution")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    common = np.linspace(0.25, 2.0, 36)
    for prefix, label in (("ssusi", "SSUSI"), ("ovation", "OVATION Prime")):
        stack = np.vstack([normalized_psd_on_grid(r, prefix, common) for r in psd_records])
        med = np.nanmedian(stack, axis=0)
        q25 = np.nanquantile(stack, 0.25, axis=0)
        q75 = np.nanquantile(stack, 0.75, axis=0)
        axes[1].plot(common, med, linewidth=1.8, label=label)
        axes[1].fill_between(common, q25, q75, alpha=0.18)
    axes[1].axvline(high_freq_cutoff, linestyle="--", linewidth=1.0,
                    label=f"high-frequency cutoff = {high_freq_cutoff:g} h$^{{-1}}$")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Spatial frequency along MLT [cycles h$^{-1}$]")
    axes[1].set_ylabel("Normalized PSD")
    axes[1].set_title("(b) Matched-segment spatial PSD")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def summarize_heldout(path: Path, output_root: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        print(f"WARNING: existing held-out benchmark not found: {path}")
        return []
    df = pd.read_csv(path)
    required = {"width_h", "method", "rmse", "mae", "pearson_r", "peak_ratio", "gradient_ratio"}
    missing = required - set(df.columns)
    if missing:
        print(f"WARNING: held-out benchmark is missing columns: {sorted(missing)}")
        return []
    cond = df[df["method"] == "conditional_ddpm"].copy()
    if cond.empty:
        print("WARNING: no conditional_ddpm rows in held-out benchmark")
        return []

    rows: List[Dict[str, Any]] = []
    for label, group in [("all", cond)] + [
        (f"{float(w):g}h", cond[cond["width_h"] == w])
        for w in sorted(cond["width_h"].unique())
    ]:
        row = {"group": label, "n": int(len(group))}
        for metric in ("rmse", "mae", "pearson_r", "peak_ratio", "gradient_ratio"):
            vals = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.nanmean(vals))
            row[f"{metric}_std"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else float("nan")
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_root / "heldout_conditional_summary.csv", index=False)
    return rows


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    ovation = np.load(args.ovation, allow_pickle=False).astype(np.float32)
    if ovation.ndim != 3 or ovation.shape[1:] != (80, 96):
        raise ValueError(f"Unexpected OVATION shape {ovation.shape}; expected (time, 80, 96)")
    ovation_start = ovation_month_start_from_path(args.ovation)

    print("=" * 92)
    print("R1.1 OVATION SMOOTHNESS-BIAS ANALYSIS")
    print("=" * 92)
    print(f"Prepared SSUSI root:     {args.prepared_root}")
    print(f"OVATION:                 {args.ovation}")
    print(f"MLAT band:               {args.band_min:g}--{args.band_max:g} deg")
    print(f"PSD min segment:         {args.min_segment_bins} bins "
          f"({args.min_segment_bins * 0.25:g} MLT h)")
    print(f"Signal requirement:      >= {args.signal_threshold:g} in at least "
          f"{100*args.min_signal_fraction:g}% of a PSD segment")
    print(f"High-frequency cutoff:   {args.high_frequency_cutoff:g} cycles/MLT-hour")
    print("No missing SSUSI cell is zero-filled or interpolated for these diagnostics.")

    gradient_summary_rows: List[Dict[str, Any]] = []
    gradient_compare_rows: List[Dict[str, Any]] = []
    psd_records: List[Dict[str, Any]] = []
    case_meta_out: Dict[str, Any] = {}

    combined_grad: Dict[str, List[np.ndarray]] = {
        "physical_ssusi": [], "physical_ovation": [],
        "log1p_ssusi": [], "log1p_ovation": [],
    }

    for case in CASES:
        data, meta = load_case(args.prepared_root, case)
        flux = np.asarray(data["flux_grid"], dtype=np.float32)
        support = np.asarray(data["obs_mask"], dtype=bool)
        start = pd.Timestamp(meta["acquisition_start"]).to_pydatetime()
        end = pd.Timestamp(meta["acquisition_end"]).to_pydatetime()
        ova, i0, i1 = ovation_interval_mean(ovation, ovation_start, start, end)

        print("\n" + "-" * 92)
        print(f"{case}: {start} -- {end}")
        print(f"Observed fraction:       {support.mean():.4f}")
        print(f"OVATION frames:          {i0}..{i1} ({i1-i0+1})")

        physical_s = supported_edge_gradients(
            flux, support, MLAT_1D, args.band_min, args.band_max
        )
        physical_o = supported_edge_gradients(
            ova, support, MLAT_1D, args.band_min, args.band_max
        )
        log_s = supported_edge_gradients(
            np.log1p(np.clip(flux, 0, None)), support,
            MLAT_1D, args.band_min, args.band_max
        )
        log_o = supported_edge_gradients(
            np.log1p(np.clip(ova, 0, None)), support,
            MLAT_1D, args.band_min, args.band_max
        )

        for space, source, values in (
            ("physical_flux", "SSUSI", physical_s),
            ("physical_flux", "OVATION", physical_o),
            ("log1p_flux", "SSUSI", log_s),
            ("log1p_flux", "OVATION", log_o),
        ):
            row = {"case": case, "space": space, "source": source}
            row.update(summarize_distribution(values))
            gradient_summary_rows.append(row)

        gradient_compare_rows.append(
            compare_gradient_distributions(case, "physical_flux", physical_s, physical_o)
        )
        gradient_compare_rows.append(
            compare_gradient_distributions(case, "log1p_flux", log_s, log_o)
        )

        combined_grad["physical_ssusi"].append(physical_s)
        combined_grad["physical_ovation"].append(physical_o)
        combined_grad["log1p_ssusi"].append(log_s)
        combined_grad["log1p_ovation"].append(log_o)

        records = collect_matched_psd_segments(
            flux, ova, support, MLAT_1D, MLT_1D,
            band_min=args.band_min,
            band_max=args.band_max,
            min_segment_bins=args.min_segment_bins,
            signal_threshold=args.signal_threshold,
            min_signal_fraction=args.min_signal_fraction,
            high_freq_cutoff=args.high_frequency_cutoff,
        )
        for record in records:
            record["case"] = case
            psd_records.append(record)
        print(f"Support-safe gradient edges: {len(log_s)}")
        print(f"Matched PSD segments:        {len(records)}")

        case_meta_out[case] = {
            "acquisition_start": start.isoformat(),
            "acquisition_end": end.isoformat(),
            "observed_fraction": float(support.mean()),
            "ovation_frame_start": int(i0),
            "ovation_frame_end": int(i1),
            "n_gradient_edges": int(len(log_s)),
            "n_psd_segments": int(len(records)),
        }

    for space, key_s, key_o in (
        ("physical_flux", "physical_ssusi", "physical_ovation"),
        ("log1p_flux", "log1p_ssusi", "log1p_ovation"),
    ):
        s = np.concatenate(combined_grad[key_s])
        o = np.concatenate(combined_grad[key_o])
        for source, values in (("SSUSI", s), ("OVATION", o)):
            row = {"case": "combined", "space": space, "source": source}
            row.update(summarize_distribution(values))
            gradient_summary_rows.append(row)
        gradient_compare_rows.append(compare_gradient_distributions("combined", space, s, o))

    gradient_summary = pd.DataFrame(gradient_summary_rows)
    gradient_compare = pd.DataFrame(gradient_compare_rows)
    gradient_summary.to_csv(args.output_root / "gradient_summary.csv", index=False)
    gradient_compare.to_csv(args.output_root / "gradient_comparison.csv", index=False)

    if not psd_records:
        raise RuntimeError(
            "No matched PSD segments passed the support/signal criteria. "
            "Consider lowering --min-segment-bins or --min-signal-fraction."
        )

    scalar_rows = []
    for r in psd_records:
        scalar_rows.append({
            "case": r["case"],
            "row_index": r["row_index"],
            "mlat": r["mlat"],
            "n_bins": r["n_bins"],
            "start_mlt": r["start_mlt"],
            "end_mlt": r["end_mlt"],
            "signal_fraction": r["signal_fraction"],
            "ssusi_high_frequency_fraction": r["ssusi_high_frequency_fraction"],
            "ovation_high_frequency_fraction": r["ovation_high_frequency_fraction"],
            "ssusi_spectral_slope": r["ssusi_spectral_slope"],
            "ovation_spectral_slope": r["ovation_spectral_slope"],
        })
    psd_df = pd.DataFrame(scalar_rows)
    psd_df.to_csv(args.output_root / "psd_segment_metrics.csv", index=False)

    psd_summary_rows: List[Dict[str, Any]] = []
    psd_comparisons: List[Dict[str, Any]] = []
    for label, group in [("combined", psd_df)] + [
        (case, psd_df[psd_df["case"] == case]) for case in CASES
    ]:
        if group.empty:
            continue
        for source in ("ssusi", "ovation"):
            hf = group[f"{source}_high_frequency_fraction"].to_numpy(float)
            slope = group[f"{source}_spectral_slope"].to_numpy(float)
            row = {"case": label, "source": source.upper(), "n_segments": int(len(group))}
            row.update({f"hf_{k}": v for k, v in summarize_distribution(hf).items() if k != "n"})
            row.update({f"slope_{k}": v for k, v in summarize_distribution(slope).items() if k != "n"})
            psd_summary_rows.append(row)

        a = group["ssusi_high_frequency_fraction"].to_numpy(float)
        b = group["ovation_high_frequency_fraction"].to_numpy(float)
        boot = paired_bootstrap_mean_difference(a, b, n_boot=args.bootstrap, seed=args.seed)
        sa = group["ssusi_spectral_slope"].to_numpy(float)
        sb = group["ovation_spectral_slope"].to_numpy(float)
        slope_boot = paired_bootstrap_mean_difference(sa, sb, n_boot=args.bootstrap, seed=args.seed + 1)
        psd_comparisons.append({
            "case": label,
            "n_segments": int(len(group)),
            "hf_mean_difference_ssusi_minus_ovation": boot["mean_difference"],
            "hf_difference_ci95_low": boot["ci95_low"],
            "hf_difference_ci95_high": boot["ci95_high"],
            "hf_paired_wilcoxon_p": safe_wilcoxon(a, b),
            "slope_mean_difference_ssusi_minus_ovation": slope_boot["mean_difference"],
            "slope_difference_ci95_low": slope_boot["ci95_low"],
            "slope_difference_ci95_high": slope_boot["ci95_high"],
            "slope_paired_wilcoxon_p": safe_wilcoxon(sa, sb),
        })

    pd.DataFrame(psd_summary_rows).to_csv(args.output_root / "psd_summary.csv", index=False)
    pd.DataFrame(psd_comparisons).to_csv(args.output_root / "psd_comparison.csv", index=False)

    heldout_rows = summarize_heldout(args.heldout_metrics, args.output_root)

    ssusi_log = np.concatenate(combined_grad["log1p_ssusi"])
    ova_log = np.concatenate(combined_grad["log1p_ovation"])
    plot_summary(
        args.output_root / "Figure_R1_1_smoothness_bias.png",
        ssusi_log, ova_log, psd_records, args.high_frequency_cutoff,
    )

    combined_grad_cmp = gradient_compare[
        (gradient_compare["case"] == "combined")
        & (gradient_compare["space"] == "log1p_flux")
    ].iloc[0].to_dict()
    combined_psd_cmp = next(x for x in psd_comparisons if x["case"] == "combined")

    summary = {
        "analysis_scope": {
            "cases": list(CASES),
            "mlat_band_deg": [args.band_min, args.band_max],
            "gradient_definition": "absolute nearest-neighbor difference; both endpoints must be SSUSI-swath supported",
            "headline_gradient_space": "log1p flux (same transform used by the reconstruction pipeline)",
            "psd_definition": "periodogram of standardized continuous MLT profiles sampled at identical SSUSI-supported cells",
            "psd_min_segment_bins": args.min_segment_bins,
            "psd_high_frequency_cutoff_cycles_per_mlt_hour": args.high_frequency_cutoff,
            "signal_threshold": args.signal_threshold,
            "min_signal_fraction": args.min_signal_fraction,
        },
        "case_metadata": case_meta_out,
        "combined_log_gradient_comparison": combined_grad_cmp,
        "combined_psd_comparison": combined_psd_cmp,
        "heldout_conditional_summary": heldout_rows,
    }
    with (args.output_root / "smoothness_bias_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=float)

    print("\n" + "=" * 92)
    print("HEADLINE R1.1 DIAGNOSTICS")
    print("=" * 92)
    print(
        "Combined log-gradient SSUSI/OVATION ratios: "
        f"median={combined_grad_cmp['median_ratio_ssusi_over_ovation']:.3f}, "
        f"p95={combined_grad_cmp['p95_ratio_ssusi_over_ovation']:.3f}, "
        f"p99={combined_grad_cmp['p99_ratio_ssusi_over_ovation']:.3f}"
    )
    print(
        "Combined log-gradient distribution: "
        f"KS={combined_grad_cmp['ks_statistic']:.3f}, "
        f"p={combined_grad_cmp['ks_pvalue']:.3e}, "
        f"Wasserstein={combined_grad_cmp['wasserstein_distance']:.4f}"
    )
    print(
        "Matched-segment high-frequency PSD difference (SSUSI-OVATION): "
        f"{combined_psd_cmp['hf_mean_difference_ssusi_minus_ovation']:+.4f} "
        f"[{combined_psd_cmp['hf_difference_ci95_low']:+.4f}, "
        f"{combined_psd_cmp['hf_difference_ci95_high']:+.4f}], "
        f"p={combined_psd_cmp['hf_paired_wilcoxon_p']:.3e}"
    )
    print(
        "Matched-segment spectral-slope difference (SSUSI-OVATION): "
        f"{combined_psd_cmp['slope_mean_difference_ssusi_minus_ovation']:+.4f} "
        f"[{combined_psd_cmp['slope_difference_ci95_low']:+.4f}, "
        f"{combined_psd_cmp['slope_difference_ci95_high']:+.4f}], "
        f"p={combined_psd_cmp['slope_paired_wilcoxon_p']:.3e}"
    )

    if heldout_rows:
        print("\nExisting conditional-DDPM real-SSUSI held-out benchmark:")
        for row in heldout_rows:
            print(
                f"  {row['group']:>4s}: n={row['n']:2d}, "
                f"RMSE={row['rmse_mean']:.4f}, "
                f"peak_ratio={row['peak_ratio_mean']:.3f}, "
                f"gradient_ratio={row['gradient_ratio_mean']:.3f}"
            )

    print("\nOutputs:")
    for name in (
        "gradient_summary.csv",
        "gradient_comparison.csv",
        "psd_segment_metrics.csv",
        "psd_summary.csv",
        "psd_comparison.csv",
        "heldout_conditional_summary.csv",
        "smoothness_bias_summary.json",
        "Figure_R1_1_smoothness_bias.png",
    ):
        path = args.output_root / name
        if path.exists():
            print(f"  {path}")

    print("\nPlease send the complete console output, smoothness_bias_summary.json, "
          "gradient_comparison.csv, psd_comparison.csv, and Figure_R1_1_smoothness_bias.png.")


if __name__ == "__main__":
    main()
