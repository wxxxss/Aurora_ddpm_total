# Second-round first batch: R2-5, R3-2, R3-11 (with R3-7 audit)

## Scope
Review the implementation of observation-constrained reverse diffusion, check the sampler mask convention and missing-value leakage, and record the provenance of Polar/UVI and SSUSI masks. R3-7's single-key attention issue is checked alongside, but is not counted as empirically resolved until the conditioning ablation is performed.

## Key findings from current code
- `models/ddpm.py` reinjects forward-noised known observations during each reverse step; the inference-time mask is `1=observed, 0=missing`.
- `models/ddpm.py` now clears missing input values before initialization, avoiding contamination of held-out tests.
- The training denoiser still receives image, timestep and solar wind only; masks/observations are introduced by the sampling loop, not by a mask-conditioned training loss.
- `models/unet.py` forms `Q=[B,H*W,32]` and a *single* `K,V=[B,1,32]`. Hence attention reduces to a broadcast global transformation, with no spatial key selection.
- Historical Polar and SSUSI scripts use different mask criteria. Do not assume that a script's presence proves it generated the manuscript figure.

## Commands
Run from repository root, using a Python environment with PyTorch and NumPy:

```bash
python paper_modif/Second_revision/R2-5_R3-2_R3-11/audit_sampling.py
python paper_modif/Second_revision/R2-5_R3-2_R3-11/audit_mask_provenance.py
```

Generated outputs: `audit_results.json` and `mask_provenance.json` in this folder. These are diagnostic reports, not statistical proof of satellite reconstruction accuracy.

## Before final reviewer response
1. Identify the checkpoint and exact script/data product that generated each manuscript real-data figure.
2. Run `audit_sampling.py` on the current branch and review all assertions.
3. Verify the physical meaning and threshold of the Polar/UVI mask from the actual figure-generation pipeline; do not infer this from an unrelated legacy script.
4. Re-run any relevant reconstruction experiments after the sampler's missing-value sanitization change, then compare against manuscript figures and metrics.
5. Evaluate new observations, true withheld cells and OVATION baselines in the next experiment group.

## Reported audit execution (20 November 2026)
The author ran `audit_sampling.py` on their local `torch` environment and provided:
- Observed pixels preserved: `true`.
- Change in missing output when observed pixels were changed: maximum absolute difference `0.06910479068756104`.
- Change in missing output when *only missing input values* were altered: `0.0`.
- Attention Q shape: `[1,20,32]`; key/value sequence length: `1`; single-key softmax = `true`.

These are code-path observations with an untrained local denoiser, **not** a satellite-accuracy benchmark and not evidence of an improvement over OVATION.

The author also reported a Python 3.9 `re.error` in `audit_mask_provenance.py`. The failing regex used doubled backslashes in a raw regex. The pattern was corrected on `main` in commit `0f4094788794218bf98840927a355639e3fb201f`; the author should pull and rerun the provenance audit.
