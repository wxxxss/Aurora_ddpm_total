#!/usr/bin/env python3
"""Collect the implementation metrics requested in Reviewer 1, Comment 4.

This script reports three groups of quantities for the final paper model:
1. quantitative training/validation loss statistics from a checkpoint history;
2. total/trainable parameter counts;
3. end-to-end reconstruction runtime for the paper's sampling configuration.

It also reports the actual number of reverse denoising-network evaluations and
forward re-noising transitions implied by the RePaint-style sampling schedule.

Run from the repository root, e.g.:
    python paper_modif/R1-4/collect_r1_4_metrics.py

If the checkpoint cannot be found automatically, provide it explicitly:
    python paper_modif/R1-4/collect_r1_4_metrics.py \
        --checkpoint /path/to/aurora_diff_best.pth \
        --history-checkpoint /path/to/aurora_diff_final.pth
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

# -----------------------------------------------------------------------------
# Make repository-root imports work when this file is executed by path.
# paper_modif/R1-4/collect_r1_4_metrics.py -> repo root is parents[2].
# -----------------------------------------------------------------------------
SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Register Ascend NPU device support when torch_npu is installed.
try:
    import torch_npu  # type: ignore  # noqa: F401
except Exception:
    torch_npu = None

from models.unet import UNet
from models.ddpm import DDPM


DEFAULT_LEGACY_CHECKPOINT = Path(
    "/home/docker/code/Aurora_DDPM_final/ckpt/cond/ckptv4_unetv1/aurora_diff_best.pth"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect R1.4 loss, parameter-count, sampling-complexity, and runtime metrics."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Best-model checkpoint used for paper reconstruction experiments."
    )
    parser.add_argument(
        "--history-checkpoint",
        type=str,
        default=None,
        help=(
            "Checkpoint containing the complete epoch_train_losses/val_losses history. "
            "If omitted, the script first looks for aurora_diff_final.pth or "
            "checkpoint_epoch_100.pth beside --checkpoint."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Runtime benchmark device: auto, npu:0, cuda:0, cpu, etc. Default: auto.",
    )
    parser.add_argument("--warmup", type=int, default=1, help="Number of warm-up reconstructions.")
    parser.add_argument("--repeats", type=int, default=5, help="Number of timed reconstructions.")
    parser.add_argument("--batch-size", type=int, default=1, help="Benchmark batch size. Default: 1.")
    parser.add_argument("--height", type=int, default=80, help="Auroral map height. Default: 80.")
    parser.add_argument("--width", type=int, default=96, help="Auroral map width. Default: 96.")
    parser.add_argument(
        "--inference-steps", type=int, default=300,
        help="Nominal reverse-diffusion horizon used in the paper. Default: 300."
    )
    parser.add_argument(
        "--jump-length", type=int, default=10,
        help="RePaint-style jump length. Default: 10."
    )
    parser.add_argument(
        "--jump-repeats", type=int, default=10,
        help="Number of repeated jumps. Default: 10."
    )
    parser.add_argument(
        "--n-sample", type=int, default=1,
        help="n_sample argument passed to DDPM.sample. Default: 1."
    )
    parser.add_argument(
        "--expected-epochs", type=int, default=100,
        help="Expected total training epochs, used only for history-completeness warnings."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(SCRIPT_PATH.with_name("r1_4_metrics.json")),
        help="JSON output path. Default: paper_modif/R1-4/r1_4_metrics.json",
    )
    return parser.parse_args()


def resolve_model_checkpoint(user_path: Optional[str]) -> Path:
    """Resolve the model checkpoint with conservative, transparent fallbacks."""
    if user_path:
        path = Path(user_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"--checkpoint does not exist: {path}")
        return path

    env_path = os.environ.get("AURORA_DDPM_CHECKPOINT")
    if env_path:
        path = Path(env_path).expanduser().resolve()
        if path.exists():
            return path

    if DEFAULT_LEGACY_CHECKPOINT.exists():
        return DEFAULT_LEGACY_CHECKPOINT.resolve()

    # Local repository fallback. Only auto-select when exactly one candidate exists.
    candidates = sorted(REPO_ROOT.glob("ckpt/**/aurora_diff_best.pth"))
    if len(candidates) == 1:
        return candidates[0].resolve()
    if len(candidates) > 1:
        candidate_text = "\n  ".join(str(p) for p in candidates)
        raise RuntimeError(
            "Multiple aurora_diff_best.pth files were found. Please pass the final paper checkpoint "
            f"explicitly with --checkpoint. Candidates:\n  {candidate_text}"
        )

    raise FileNotFoundError(
        "Could not locate the paper checkpoint automatically. Run again with:\n"
        "  --checkpoint /path/to/aurora_diff_best.pth\n"
        "Optionally also provide --history-checkpoint /path/to/aurora_diff_final.pth"
    )


def resolve_history_checkpoint(
    model_checkpoint: Path, user_history_path: Optional[str]
) -> Tuple[Path, str]:
    """Prefer a final/epoch-100 checkpoint so loss history spans all epochs."""
    if user_history_path:
        path = Path(user_history_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"--history-checkpoint does not exist: {path}")
        return path, "explicit --history-checkpoint"

    sibling_candidates = [
        model_checkpoint.parent / "aurora_diff_final.pth",
        model_checkpoint.parent / "checkpoint_epoch_100.pth",
    ]
    for path in sibling_candidates:
        if path.exists():
            return path.resolve(), f"auto-selected sibling {path.name}"

    return model_checkpoint, "fallback to model checkpoint"


def load_checkpoint(path: Path) -> Dict[str, Any]:
    obj = torch.load(str(path), map_location="cpu")
    if not isinstance(obj, dict):
        raise TypeError(f"Checkpoint must be a dict, got {type(obj)} from {path}")
    return obj


def collect_loss_statistics(
    checkpoint: Dict[str, Any], history_path: Path, expected_epochs: int
) -> Dict[str, Any]:
    train = checkpoint.get("epoch_train_losses")
    val = checkpoint.get("val_losses")

    if train is None or val is None:
        missing = []
        if train is None:
            missing.append("epoch_train_losses")
        if val is None:
            missing.append("val_losses")
        raise KeyError(
            f"History checkpoint {history_path} is missing: {', '.join(missing)}. "
            "Use a final or epoch-100 checkpoint produced by train.py."
        )

    train = np.asarray(train, dtype=np.float64)
    val = np.asarray(val, dtype=np.float64)

    if train.size == 0 or val.size == 0:
        raise ValueError("Loss-history arrays are empty.")

    best_idx = int(np.argmin(val))
    stats: Dict[str, Any] = {
        "history_checkpoint": str(history_path),
        "num_train_epochs_recorded": int(train.size),
        "num_validation_epochs_recorded": int(val.size),
        "initial_train_loss": float(train[0]),
        "final_train_loss": float(train[-1]),
        "initial_validation_loss": float(val[0]),
        "final_validation_loss": float(val[-1]),
        "best_validation_loss": float(val[best_idx]),
        "best_validation_epoch": int(best_idx + 1),
    }

    if train.size >= 10:
        stats["mean_train_loss_last_10_epochs"] = float(np.mean(train[-10:]))
    if val.size >= 10:
        stats["mean_validation_loss_last_10_epochs"] = float(np.mean(val[-10:]))

    stats["history_complete_for_expected_epochs"] = bool(
        train.size >= expected_epochs and val.size >= expected_epochs
    )
    return stats


def build_model(model_checkpoint: Path) -> Tuple[DDPM, Dict[str, Any]]:
    unet = UNet(1, 1)
    ddpm = DDPM(unet, num_train_steps=1000, schedule="cosine")

    ckpt = load_checkpoint(model_checkpoint)
    state_dict = ckpt.get("model_state_dict", ckpt)
    incompatible = ddpm.load_state_dict(state_dict, strict=False)

    load_info = {
        "model_checkpoint": str(model_checkpoint),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }
    return ddpm, load_info


def collect_parameter_count(model: torch.nn.Module) -> Dict[str, Any]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "total_parameters_million": float(total / 1e6),
        "trainable_parameters_million": float(trainable / 1e6),
    }


def collect_sampling_schedule(
    ddpm: DDPM,
    inference_steps: int,
    n_sample: int,
    jump_length: int,
    jump_repeats: int,
) -> Dict[str, Any]:
    times = ddpm.get_inference_schedule(
        t_T=inference_steps,
        n_sample=n_sample,
        jump_len=jump_length,
        jump_n_sample=jump_repeats,
    )

    transitions = list(zip(times[:-1], times[1:]))
    reverse_transitions = sum(1 for t_last, t_cur in transitions if t_cur < t_last)
    forward_transitions = sum(1 for t_last, t_cur in transitions if t_cur > t_last)

    # DDPM.step() returns immediately at t=0 without calling the U-Net.
    denoising_network_evaluations = sum(
        1 for t_last, t_cur in transitions if t_cur < t_last and t_last > 0
    )
    terminal_t0_merge = sum(
        1 for t_last, t_cur in transitions if t_cur < t_last and t_last == 0
    )

    return {
        "nominal_inference_steps": int(inference_steps),
        "n_sample": int(n_sample),
        "jump_length": int(jump_length),
        "jump_repeats": int(jump_repeats),
        "schedule_points": int(len(times)),
        "total_schedule_transitions": int(len(transitions)),
        "reverse_transitions": int(reverse_transitions),
        "forward_renoising_transitions": int(forward_transitions),
        "denoising_network_evaluations": int(denoising_network_evaluations),
        "terminal_t0_merge_without_unet": int(terminal_t0_merge),
    }


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


def hardware_name(device: torch.device) -> str:
    idx = 0 if device.index is None else device.index
    try:
        if device.type == "npu" and hasattr(torch, "npu"):
            return str(torch.npu.get_device_name(idx))
        if device.type == "cuda":
            return str(torch.cuda.get_device_name(idx))
    except Exception:
        pass
    if device.type == "cpu":
        return platform.processor() or platform.machine() or "CPU"
    return str(device)


def collect_runtime(
    ddpm: DDPM,
    device: torch.device,
    batch_size: int,
    height: int,
    width: int,
    inference_steps: int,
    n_sample: int,
    jump_length: int,
    jump_repeats: int,
    warmup: int,
    repeats: int,
) -> Dict[str, Any]:
    if device.type == "cpu":
        print(
            "WARNING: runtime benchmark is running on CPU. This is valid for debugging but should not "
            "be reported in the paper if the paper experiments were performed on an NPU."
        )

    ddpm = ddpm.to(device)
    ddpm.eval()

    # Deterministic representative tensors. Runtime depends on tensor shapes and the
    # sampling schedule, not on the physical values themselves.
    torch.manual_seed(2026)
    image = torch.rand(batch_size, 1, height, width, device=device, dtype=torch.float32)
    mask = torch.ones_like(image)

    # Representative missing sector, matching the spatial scale used in the paper tests.
    row0 = max(0, height // 2 - 10)
    row1 = min(height, row0 + 20)
    col0 = max(0, width // 24)       # roughly 1 MLT on a 96-column grid
    col1 = min(width, col0 + 16)     # roughly a 4-hour MLT sector
    mask[:, :, row0:row1, col0:col1] = 0.0

    # The paper uses five normalized conditioning variables [Bx, By, Bz, V, Pdyn].
    solar = torch.full((batch_size, 5), 0.5, device=device, dtype=torch.float32)

    def one_reconstruction() -> torch.Tensor:
        return ddpm.sample(
            image,
            mask,
            solar,
            num_inference_steps=inference_steps,
            n_sample=n_sample,
            j=jump_length,
            r=jump_repeats,
        )

    with torch.no_grad():
        for _ in range(max(0, warmup)):
            _ = one_reconstruction()
        synchronize(device)

        timings = []
        for i in range(repeats):
            synchronize(device)
            t0 = time.perf_counter()
            _ = one_reconstruction()
            synchronize(device)
            elapsed = time.perf_counter() - t0
            timings.append(elapsed)
            print(f"  timed reconstruction {i + 1}/{repeats}: {elapsed:.6f} s")

    arr = np.asarray(timings, dtype=np.float64)
    return {
        "device": str(device),
        "hardware_name": hardware_name(device),
        "batch_size": int(batch_size),
        "image_shape": [int(height), int(width)],
        "warmup_runs": int(warmup),
        "timed_runs": int(repeats),
        "runtime_seconds_each": [float(x) for x in arr],
        "runtime_mean_seconds_per_batch": float(arr.mean()),
        "runtime_std_seconds_per_batch": float(arr.std(ddof=1) if arr.size > 1 else 0.0),
        "runtime_median_seconds_per_batch": float(np.median(arr)),
        "runtime_min_seconds_per_batch": float(arr.min()),
        "runtime_max_seconds_per_batch": float(arr.max()),
        "runtime_mean_seconds_per_image": float(arr.mean() / batch_size),
        "runtime_std_seconds_per_image": float(
            (arr.std(ddof=1) if arr.size > 1 else 0.0) / batch_size
        ),
    }


def print_section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main() -> None:
    args = parse_args()

    model_checkpoint = resolve_model_checkpoint(args.checkpoint)
    history_checkpoint, history_resolution = resolve_history_checkpoint(
        model_checkpoint, args.history_checkpoint
    )

    print_section("R1.4 INPUTS")
    print(f"Repository root:       {REPO_ROOT}")
    print(f"Model checkpoint:      {model_checkpoint}")
    print(f"History checkpoint:    {history_checkpoint}")
    print(f"History resolution:    {history_resolution}")

    # ------------------------------------------------------------------
    # 1. Loss statistics
    # ------------------------------------------------------------------
    history_ckpt = load_checkpoint(history_checkpoint)
    loss_stats = collect_loss_statistics(
        history_ckpt, history_checkpoint, args.expected_epochs
    )

    print_section("1) LOSS STATISTICS")
    print(f"Recorded training epochs:  {loss_stats['num_train_epochs_recorded']}")
    print(f"Recorded validation epochs:{loss_stats['num_validation_epochs_recorded']}")
    print(f"Initial train loss:         {loss_stats['initial_train_loss']:.9f}")
    print(f"Final train loss:           {loss_stats['final_train_loss']:.9f}")
    print(f"Initial validation loss:    {loss_stats['initial_validation_loss']:.9f}")
    print(f"Final validation loss:      {loss_stats['final_validation_loss']:.9f}")
    print(f"Best validation loss:       {loss_stats['best_validation_loss']:.9f}")
    print(f"Best validation epoch:      {loss_stats['best_validation_epoch']}")
    print(
        f"History complete for {args.expected_epochs} epochs: "
        f"{loss_stats['history_complete_for_expected_epochs']}"
    )
    if not loss_stats["history_complete_for_expected_epochs"]:
        print(
            "WARNING: the selected history checkpoint does not contain the full expected training history. "
            "Point --history-checkpoint to aurora_diff_final.pth or checkpoint_epoch_100.pth."
        )

    # ------------------------------------------------------------------
    # 2. Parameter count + sampling schedule complexity
    # ------------------------------------------------------------------
    ddpm, load_info = build_model(model_checkpoint)
    param_stats = collect_parameter_count(ddpm)
    schedule_stats = collect_sampling_schedule(
        ddpm,
        args.inference_steps,
        args.n_sample,
        args.jump_length,
        args.jump_repeats,
    )

    print_section("2) PARAMETER COUNT AND SAMPLING COMPLEXITY")
    print(f"Total parameters:           {param_stats['total_parameters']:,}")
    print(f"Trainable parameters:       {param_stats['trainable_parameters']:,}")
    print(f"Trainable parameters (M):   {param_stats['trainable_parameters_million']:.6f}")
    print(f"Missing checkpoint keys:    {len(load_info['missing_keys'])}")
    print(f"Unexpected checkpoint keys: {len(load_info['unexpected_keys'])}")
    if load_info["missing_keys"]:
        print("  Missing keys:")
        for key in load_info["missing_keys"]:
            print(f"    - {key}")
    if load_info["unexpected_keys"]:
        print("  Unexpected keys:")
        for key in load_info["unexpected_keys"]:
            print(f"    - {key}")

    print(f"Nominal inference steps:        {schedule_stats['nominal_inference_steps']}")
    print(f"Jump length:                    {schedule_stats['jump_length']}")
    print(f"Jump repeats:                   {schedule_stats['jump_repeats']}")
    print(f"Total schedule transitions:     {schedule_stats['total_schedule_transitions']}")
    print(f"Reverse transitions:            {schedule_stats['reverse_transitions']}")
    print(f"Forward re-noising transitions: {schedule_stats['forward_renoising_transitions']}")
    print(f"Denoising-network evaluations:  {schedule_stats['denoising_network_evaluations']}")

    # ------------------------------------------------------------------
    # 3. Runtime benchmark
    # ------------------------------------------------------------------
    device = choose_device(args.device)
    print_section("3) END-TO-END INFERENCE RUNTIME")
    print(f"Benchmark device: {device}")
    print(f"Hardware:         {hardware_name(device)}")
    print(f"Warm-up runs:     {args.warmup}")
    print(f"Timed runs:       {args.repeats}")

    runtime_stats = collect_runtime(
        ddpm=ddpm,
        device=device,
        batch_size=args.batch_size,
        height=args.height,
        width=args.width,
        inference_steps=args.inference_steps,
        n_sample=args.n_sample,
        jump_length=args.jump_length,
        jump_repeats=args.jump_repeats,
        warmup=args.warmup,
        repeats=args.repeats,
    )

    print(f"Mean runtime:   {runtime_stats['runtime_mean_seconds_per_image']:.6f} s/image")
    print(f"Std runtime:    {runtime_stats['runtime_std_seconds_per_image']:.6f} s/image")
    print(f"Median runtime: {runtime_stats['runtime_median_seconds_per_batch'] / args.batch_size:.6f} s/image")
    print(f"Range:          {runtime_stats['runtime_min_seconds_per_batch'] / args.batch_size:.6f} -- "
          f"{runtime_stats['runtime_max_seconds_per_batch'] / args.batch_size:.6f} s/image")

    results: Dict[str, Any] = {
        "repository_root": str(REPO_ROOT),
        "model_checkpoint": str(model_checkpoint),
        "history_checkpoint": str(history_checkpoint),
        "history_resolution": history_resolution,
        "loss_statistics": loss_stats,
        "checkpoint_load": load_info,
        "parameter_count": param_stats,
        "sampling_complexity": schedule_stats,
        "runtime": runtime_stats,
        "software": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_npu": getattr(torch_npu, "__version__", None) if torch_npu is not None else None,
            "numpy": np.__version__,
        },
    }

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print_section("DONE")
    print(f"JSON results saved to: {output_path}")
    print("Please send me the complete console output (or the JSON file contents).")


if __name__ == "__main__":
    main()
