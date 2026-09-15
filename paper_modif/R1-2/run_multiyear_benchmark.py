#!/usr/bin/env python3
"""Reviewer 1 Comment 2: multi-year robustness benchmark.

The final conditional checkpoint is evaluated without retraining on the
prepared 2001/2005/2009 OVATION test set. The same controlled missing region,
diffusion seed, inference schedule, and metrics are used for the conditional
and unconditional DDPMs. Interpolation follows the manuscript baseline.

Primary analyses:
1. solar-cycle phase: 2001 solar maximum, 2005 declining phase, 2009 minimum;
2. geomagnetic activity: balanced pooled Kp<=3 and Kp>=4 samples;
3. paired conditional-vs-baseline improvements with bootstrap 95% CIs and
   two-sided Wilcoxon signed-rank tests.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

try:
    import torch_npu  # noqa: F401
except Exception:
    torch_npu = None

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
REPO_ROOT = SCRIPT_PATH.parents[2]
for p in (SCRIPT_DIR, REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from benchmark_utils import (
    compute_masked_metrics,
    create_paper_mask,
    interpolate_inpainting,
    paired_comparison,
    summarize_methods,
)
from models.ddpm import DDPM
from models.ddpm_nocond import DDPM_nocond
from models.simplenet import UNet as UNetNoCond
from models.unet import UNet

DEFAULT_TESTSET = SCRIPT_DIR / "r1_2_prepared/multiyear_testset.npz"
DEFAULT_COND_CKPT = REPO_ROOT / "ckpt/cond/aurora_diff_best.pth"
DEFAULT_UNCOND_CKPT = REPO_ROOT / "ckpt/uncond/aurora_diff_best.pth"
DEFAULT_COND_NORM = REPO_ROOT / "ckpt/cond/norm_params.pkl"
DEFAULT_UNCOND_NORM = REPO_ROOT / "ckpt/uncond/norm_params.pkl"
DEFAULT_OUTPUT = SCRIPT_DIR / "r1_2_results"
METHODS = ("conditional", "unconditional", "interpolation")
SOLAR_FIELDS = ("Bx", "By", "Bz", "V", "P")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--testset", type=Path, default=DEFAULT_TESTSET)
    p.add_argument("--cond-checkpoint", type=Path, default=DEFAULT_COND_CKPT)
    p.add_argument("--uncond-checkpoint", type=Path, default=DEFAULT_UNCOND_CKPT)
    p.add_argument("--cond-norm", type=Path, default=DEFAULT_COND_NORM)
    p.add_argument("--uncond-norm", type=Path, default=DEFAULT_UNCOND_NORM)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--device", default="auto")
    p.add_argument("--limit", type=int, default=0, help="Run only first N maps for smoke testing; 0=all")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--inference-steps", type=int, default=300)
    p.add_argument("--jump-length", type=int, default=10)
    p.add_argument("--jump-repeats", type=int, default=10)
    p.add_argument("--fresh", action="store_true", help="Ignore any existing per-sample CSV and recompute")
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


def load_norm(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Normalization file not found: {path}")
    with path.open("rb") as f:
        obj = pickle.load(f)
    if "aurora" not in obj:
        raise KeyError(f"Normalization file missing aurora parameters: {path}")
    return obj


def normalize_aurora(flux: np.ndarray, norm: Dict[str, Any]) -> np.ndarray:
    x = np.clip(np.asarray(flux, dtype=np.float32), 0.0, None)
    log_x = np.log1p(x)
    y = (log_x - float(norm["aurora"]["min"])) / float(norm["aurora"]["range"])
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def denormalize_aurora(y: np.ndarray, norm: Dict[str, Any]) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    log_x = y * float(norm["aurora"]["range"]) + float(norm["aurora"]["min"])
    return np.clip(np.expm1(log_x), 0.0, None).astype(np.float32)


def normalize_conditions(raw: np.ndarray, norm: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    if "omni" not in norm:
        raise KeyError("Conditional normalization file has no OMNI parameters")
    raw = np.asarray(raw, dtype=np.float32)
    minimum = np.asarray(norm["omni"]["min"], dtype=np.float32).reshape(-1)
    value_range = np.asarray(norm["omni"]["range"], dtype=np.float32).reshape(-1)
    if minimum.size != 5 or value_range.size != 5:
        raise ValueError(f"Expected five OMNI normalization dimensions, got {minimum.size}")
    unbounded = (raw - minimum) / value_range
    clipped = (unbounded < 0.0) | (unbounded > 1.0)
    return np.clip(unbounded, 0.0, 1.0).astype(np.float32), clipped


def load_conditional(checkpoint_path: Path, device: torch.device) -> DDPM:
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    model = DDPM(UNet(1, 1), num_train_steps=1000, schedule="cosine")
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    inc = model.load_state_dict(state, strict=False)
    if getattr(inc, "missing_keys", None) or getattr(inc, "unexpected_keys", None):
        print(f"Conditional checkpoint note: missing={len(inc.missing_keys)}, unexpected={len(inc.unexpected_keys)}")
    return model.eval().to(device)


def load_unconditional(checkpoint_path: Path, device: torch.device) -> DDPM_nocond:
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    model = DDPM_nocond(UNetNoCond(1, 1), num_train_steps=1000, schedule="cosine")
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    inc = model.load_state_dict(state, strict=False)
    if getattr(inc, "missing_keys", None) or getattr(inc, "unexpected_keys", None):
        print(f"Unconditional checkpoint note: missing={len(inc.missing_keys)}, unexpected={len(inc.unexpected_keys)}")
    return model.eval().to(device)


def reconstruct_conditional(
    model: DDPM,
    truth: np.ndarray,
    mask: np.ndarray,
    cond_norm: np.ndarray,
    norm: Dict[str, Any],
    device: torch.device,
    *,
    seed: int,
    inference_steps: int,
    jump_length: int,
    jump_repeats: int,
) -> np.ndarray:
    corrupted = np.asarray(truth, dtype=np.float32).copy()
    corrupted[mask == 0] = 0.0
    image_t = torch.from_numpy(normalize_aurora(corrupted, norm)[None, None]).float().to(device)
    mask_t = torch.from_numpy(mask[None, None].astype(np.float32)).to(device)
    cond_t = torch.from_numpy(np.asarray(cond_norm, dtype=np.float32)[None]).to(device)
    set_all_seeds(seed, device)
    synchronize(device)
    with torch.no_grad():
        out = model.sample(
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


def reconstruct_unconditional(
    model: DDPM_nocond,
    truth: np.ndarray,
    mask: np.ndarray,
    norm: Dict[str, Any],
    device: torch.device,
    *,
    seed: int,
    inference_steps: int,
    jump_length: int,
    jump_repeats: int,
) -> np.ndarray:
    corrupted = np.asarray(truth, dtype=np.float32).copy()
    corrupted[mask == 0] = 0.0
    image_t = torch.from_numpy(normalize_aurora(corrupted, norm)[None, None]).float().to(device)
    mask_t = torch.from_numpy(mask[None, None].astype(np.float32)).to(device)
    set_all_seeds(seed, device)
    synchronize(device)
    with torch.no_grad():
        out = model.sample(
            image_t,
            mask_t,
            num_inference_steps=int(inference_steps),
            n_sample=1,
            j=int(jump_length),
            r=int(jump_repeats),
        )
    synchronize(device)
    return denormalize_aurora(out.detach().cpu().numpy().squeeze(), norm)


def json_safe(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    if isinstance(x, np.ndarray):
        return [json_safe(v) for v in x.tolist()]
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        y = float(x)
        return y if np.isfinite(y) else None
    if isinstance(x, Path):
        return str(x)
    return x


def write_latex_table(path: Path, summary: pd.DataFrame, caption: str, label: str) -> None:
    order = ["conditional", "unconditional", "interpolation"]
    display = {
        "conditional": "Conditional DDPM",
        "unconditional": "Unconditional DDPM",
        "interpolation": "Interpolation",
    }
    endrow = r"\\"
    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\begin{tabular}{llrrrr}",
        r"\hline",
        "Group & Method & $n$ & SSIM & RMSE (95\\% CI) & $R^2$ " + endrow,
        r"\hline",
    ]
    for group in summary["group"].drop_duplicates():
        part = summary[summary["group"] == group].set_index("method")
        for method in order:
            if method not in part.index:
                continue
            row = part.loc[method]
            lines.append(
                f"{group} & {display[method]} & {int(row['n'])} & {float(row['mean_ssim']):.3f} & "
                f"{float(row['mean_rmse']):.3f} [{float(row['ci_low_rmse']):.3f}, {float(row['ci_high_rmse']):.3f}] & "
                f"{float(row['mean_r2']):.3f} " + endrow
            )
        lines.append(r"\hline")
    lines.extend([r"\end{tabular}", r"\end{table*}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_summaries(metrics_df: pd.DataFrame, bootstrap_n: int, seed: int):
    phase_df = metrics_df.loc[metrics_df["phase_sample"] == True].copy()  # noqa: E712
    phase_rows: List[Dict[str, Any]] = []
    pair_rows: List[Dict[str, Any]] = []
    phase_order = ["solar_maximum", "declining_phase", "solar_minimum"]
    for gi, group in enumerate(phase_order):
        phase_rows.extend(
            summarize_methods(
                phase_df,
                group_col="phase",
                group_value=group,
                bootstrap_n=bootstrap_n,
                seed=seed + gi * 1000,
            )
        )
        for bi, baseline in enumerate(("unconditional", "interpolation")):
            rec = paired_comparison(
                phase_df,
                group_col="phase",
                group_value=group,
                baseline=baseline,
                bootstrap_n=bootstrap_n,
                seed=seed + gi * 1000 + bi * 100,
            )
            rec["analysis"] = "solar_phase"
            pair_rows.append(rec)

    low = metrics_df.loc[metrics_df["activity_low"] == True].copy()  # noqa: E712
    low["activity_group"] = "Kp<=3"
    high = metrics_df.loc[metrics_df["activity_high"] == True].copy()  # noqa: E712
    high["activity_group"] = "Kp>=4"
    activity_df = pd.concat([low, high], ignore_index=True)
    activity_rows: List[Dict[str, Any]] = []
    for gi, group in enumerate(("Kp<=3", "Kp>=4")):
        activity_rows.extend(
            summarize_methods(
                activity_df,
                group_col="activity_group",
                group_value=group,
                bootstrap_n=bootstrap_n,
                seed=seed + 5000 + gi * 1000,
            )
        )
        for bi, baseline in enumerate(("unconditional", "interpolation")):
            rec = paired_comparison(
                activity_df,
                group_col="activity_group",
                group_value=group,
                baseline=baseline,
                bootstrap_n=bootstrap_n,
                seed=seed + 5000 + gi * 1000 + bi * 100,
            )
            rec["analysis"] = "kp_activity"
            pair_rows.append(rec)
    return pd.DataFrame(phase_rows), pd.DataFrame(activity_rows), pd.DataFrame(pair_rows)


def condition_clip_audit(data: Any, clipped: np.ndarray) -> Dict[str, Any]:
    def summarize(sel: np.ndarray) -> Dict[str, Any]:
        sel = np.asarray(sel, dtype=bool)
        if not np.any(sel):
            return {"n": 0}
        part = clipped[sel]
        return {
            "n": int(np.sum(sel)),
            "any_dimension_fraction": float(np.mean(np.any(part, axis=1))),
            "by_field": {name: float(np.mean(part[:, j])) for j, name in enumerate(SOLAR_FIELDS)},
        }

    audit: Dict[str, Any] = {
        "overall": summarize(np.ones(len(clipped), dtype=bool)),
        "by_phase": {},
        "by_activity": {},
    }
    for phase in ("solar_maximum", "declining_phase", "solar_minimum"):
        sel = (data["phase"] == phase) & data["phase_sample"]
        audit["by_phase"][phase] = summarize(sel)
    audit["by_activity"]["Kp<=3"] = summarize(data["activity_low"])
    audit["by_activity"]["Kp>=4"] = summarize(data["activity_high"])
    return audit


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)

    print("=" * 96)
    print("R1.2 MULTI-YEAR SOLAR-CYCLE / GEOMAGNETIC-ACTIVITY ROBUSTNESS BENCHMARK")
    print("=" * 96)
    print(f"Device:            {device}")
    print(f"Test set:          {args.testset}")
    print(f"Inference steps:   {args.inference_steps}")
    print(f"Jump length/reps:  {args.jump_length}/{args.jump_repeats}")

    if not args.testset.exists():
        raise FileNotFoundError(args.testset)
    data = np.load(args.testset, allow_pickle=False)
    required = (
        "aurora", "solar", "kp", "utc", "year", "phase",
        "phase_sample", "activity_low", "activity_high",
    )
    missing = [key for key in required if key not in data.files]
    if missing:
        raise KeyError(f"Test set missing arrays: {missing}")

    n_total = len(data["aurora"])
    n_run = min(n_total, args.limit) if args.limit > 0 else n_total
    sample_ids = np.arange(n_run, dtype=int)
    print(f"Samples this run:  {n_run}/{n_total}")
    print(f"Aurora shape:      {data['aurora'].shape}")

    cond_norm = load_norm(args.cond_norm)
    if args.uncond_norm.exists():
        uncond_norm = load_norm(args.uncond_norm)
        uncond_norm_source = str(args.uncond_norm)
    else:
        uncond_norm = cond_norm
        uncond_norm_source = f"fallback_to_conditional:{args.cond_norm}"
        print(
            "NOTE: unconditional norm file not found; using conditional aurora "
            f"normalization: {args.cond_norm}"
        )

    conditions_norm, clipped = normalize_conditions(data["solar"], cond_norm)
    clip_audit = condition_clip_audit(data, clipped)
    print("Overall condition clipping:", clip_audit["overall"])

    print("Loading paper checkpoints...")
    cond_model = load_conditional(args.cond_checkpoint, device)
    uncond_model = load_unconditional(args.uncond_checkpoint, device)
    mask = create_paper_mask(data["aurora"].shape[1:], mlat_range=(60.0, 80.0))
    print(f"Masked pixels:     {int(np.sum(mask == 0))}/{mask.size} ({np.mean(mask == 0):.3f})")

    metrics_path = args.output_dir / "metrics_per_sample.csv"
    existing = pd.DataFrame()
    if metrics_path.exists() and not args.fresh:
        existing = pd.read_csv(metrics_path)
        print(f"Resume file found: {len(existing)} method rows")

    done = set()
    if not existing.empty and {"sample_id", "method"}.issubset(existing.columns):
        counts = existing.groupby("sample_id")["method"].nunique()
        done = set(int(i) for i in counts[counts >= len(METHODS)].index)
        # Keep only complete sample triplets. A partially written sample is recomputed
        # from scratch so an interrupted run cannot create duplicate method rows.
        existing = existing[existing["sample_id"].astype(int).isin(done)].copy()
        print(f"Complete samples already present: {len(done)}")

    rows: List[Dict[str, Any]] = existing.to_dict(orient="records") if not args.fresh else []
    start = time.time()
    completed_now = 0

    for pos, sid in enumerate(sample_ids, start=1):
        if int(sid) in done:
            continue

        truth = np.asarray(data["aurora"][sid], dtype=np.float32)
        seed_i = int(args.seed + int(sid) * 1009)
        corrupted = truth.copy()
        corrupted[mask == 0] = 0.0

        t0 = time.time()
        pred_cond = reconstruct_conditional(
            cond_model,
            truth,
            mask,
            conditions_norm[sid],
            cond_norm,
            device,
            seed=seed_i,
            inference_steps=args.inference_steps,
            jump_length=args.jump_length,
            jump_repeats=args.jump_repeats,
        )
        t1 = time.time()
        pred_uncond = reconstruct_unconditional(
            uncond_model,
            truth,
            mask,
            uncond_norm,
            device,
            seed=seed_i,
            inference_steps=args.inference_steps,
            jump_length=args.jump_length,
            jump_repeats=args.jump_repeats,
        )
        t2 = time.time()
        pred_interp = interpolate_inpainting(corrupted, mask)
        t3 = time.time()

        base = {
            "sample_id": int(sid),
            "utc": str(data["utc"][sid].astype("datetime64[s]")),
            "year": int(data["year"][sid]),
            "phase": str(data["phase"][sid]),
            "kp": float(data["kp"][sid]),
            "phase_sample": bool(data["phase_sample"][sid]),
            "activity_low": bool(data["activity_low"][sid]),
            "activity_high": bool(data["activity_high"][sid]),
            "condition_clipped_dims": int(np.sum(clipped[sid])),
        }
        for method, pred, runtime in (
            ("conditional", pred_cond, t1 - t0),
            ("unconditional", pred_uncond, t2 - t1),
            ("interpolation", pred_interp, t3 - t2),
        ):
            met = compute_masked_metrics(truth, pred, mask)
            rows.append({**base, "method": method, **met, "runtime_s": float(runtime)})

        completed_now += 1
        pd.DataFrame(rows).to_csv(metrics_path, index=False)
        if pos == 1 or pos % 5 == 0 or pos == len(sample_ids):
            mc = compute_masked_metrics(truth, pred_cond, mask)
            print(
                f"[{pos:03d}/{len(sample_ids):03d}] {base['utc']} year={base['year']} "
                f"Kp={base['kp']:.1f} cond RMSE={mc['rmse']:.4f} R2={mc['r2']:.4f} "
                f"runtime={t2-t0:.1f}s clipped={base['condition_clipped_dims']}"
            )

    metrics_df = pd.DataFrame(rows)
    valid_ids = set(int(x) for x in sample_ids)
    metrics_df = metrics_df[metrics_df["sample_id"].astype(int).isin(valid_ids)].copy()
    metrics_df.to_csv(metrics_path, index=False)

    phase_summary, kp_summary, paired = build_summaries(metrics_df, args.bootstrap, args.seed)
    phase_path = args.output_dir / "solar_phase_summary.csv"
    kp_path = args.output_dir / "kp_group_summary.csv"
    pair_path = args.output_dir / "paired_significance.csv"
    phase_summary.to_csv(phase_path, index=False)
    kp_summary.to_csv(kp_path, index=False)
    paired.to_csv(pair_path, index=False)

    write_latex_table(
        args.output_dir / "solar_phase_table.tex",
        phase_summary,
        "Reconstruction performance across solar-cycle phases. Metrics are evaluated only in the controlled missing region; RMSE intervals are bootstrap 95\\% confidence intervals for the method mean.",
        "tab:solar_phase_robustness",
    )
    write_latex_table(
        args.output_dir / "kp_activity_table.tex",
        kp_summary,
        "Reconstruction performance under low and elevated geomagnetic activity. The two activity groups contain balanced multi-year samples with $Kp\\leq3$ and $Kp\\geq4$, respectively.",
        "tab:kp_robustness",
    )

    summary_json = {
        "testset": str(args.testset),
        "samples_requested": int(n_run),
        "method_rows": int(len(metrics_df)),
        "inference_steps": int(args.inference_steps),
        "jump_length": int(args.jump_length),
        "jump_repeats": int(args.jump_repeats),
        "seed": int(args.seed),
        "mask_missing_fraction": float(np.mean(mask == 0)),
        "condition_clipping": clip_audit,
        "conditional_norm": str(args.cond_norm),
        "unconditional_norm": uncond_norm_source,
        "conditional_checkpoint": str(args.cond_checkpoint),
        "unconditional_checkpoint": str(args.uncond_checkpoint),
        "solar_phase_summary": phase_summary.to_dict(orient="records"),
        "kp_group_summary": kp_summary.to_dict(orient="records"),
        "paired_significance": paired.to_dict(orient="records"),
        "wall_time_min": float((time.time() - start) / 60.0),
        "completed_now": int(completed_now),
    }
    with (args.output_dir / "robustness_summary.json").open("w", encoding="utf-8") as f:
        json.dump(json_safe(summary_json), f, indent=2)

    print("\n" + "=" * 96)
    print("R1.2 HEADLINE SUMMARY")
    print("=" * 96)
    for _, row in phase_summary[phase_summary.method == "conditional"].iterrows():
        print(
            f"{row['group']:16s} n={int(row['n']):2d} | "
            f"RMSE={row['mean_rmse']:.4f} [{row['ci_low_rmse']:.4f}, {row['ci_high_rmse']:.4f}] | "
            f"SSIM={row['mean_ssim']:.4f} | R2={row['mean_r2']:.4f}"
        )
    for _, row in kp_summary[kp_summary.method == "conditional"].iterrows():
        print(
            f"{row['group']:16s} n={int(row['n']):2d} | "
            f"RMSE={row['mean_rmse']:.4f} [{row['ci_low_rmse']:.4f}, {row['ci_high_rmse']:.4f}] | "
            f"SSIM={row['mean_ssim']:.4f} | R2={row['mean_r2']:.4f}"
        )

    print("\nPaired RMSE improvement (positive = conditional better):")
    for _, row in paired.iterrows():
        print(
            f"{row['analysis']:12s} {row['group']:16s} vs {row['baseline']:13s}: "
            f"{row['mean_improvement_rmse']:+.4f} "
            f"[{row['ci_low_improvement_rmse']:+.4f}, {row['ci_high_improvement_rmse']:+.4f}], "
            f"p={row['p_rmse']:.3e}"
        )

    print("\nOutputs:")
    for path in (
        metrics_path,
        phase_path,
        kp_path,
        pair_path,
        args.output_dir / "robustness_summary.json",
        args.output_dir / "solar_phase_table.tex",
        args.output_dir / "kp_activity_table.tex",
    ):
        print(f"  {path}")
    print(f"Wall time: {(time.time() - start)/60.0:.1f} min")


if __name__ == "__main__":
    main()
