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

## Mask-provenance audit received
The author supplied `mask_provenance.json` from the corrected `audit_mask_provenance.py` run. The result confirms distinct historical criteria:
- `repaired_polar.py`: `mask=0` when `polar < 1`. No OVATION criterion appears in that mask assignment.
- `repaired_ssusi.py`: `mask=0` only for `(ssusi < 1) & (ovation > 0)`.
- First-round `rerun_ssusi_reconstruction.py`: `mask=1` for native swath-supported grid cells (`UT_N > 0` propagated by direct binning), irrespective of measured flux magnitude.
- First-round `redraw_polar_figures.py`: `OBS_SUPPORT_THRESHOLD=1.0` controls observational display and diagnostics, not the saved model's inference-time mask.

**Do not merge these criteria into one claimed method.** The current manuscript describes an OVATION-dependent Polar mask, yet the legacy Polar reconstruction script does not implement that dependency. To trace Figure 4/5 provenance, run `check_polar_mask_cases.py` against the actual saved Polar reconstruction. Confirm its checkpoint and program version before changing the manuscript's operational criterion or completing the R3-11 response.

Run from repository root:

```bash
python paper_modif/Second_revision/R2-5_R3-2_R3-11/check_polar_mask_cases.py
```

Outputs: `polar_mask_case_diagnostic.json`. This compares candidate mask categories and saved-image agreement only; it does not establish the causal mask rule if the production script/checkpoint is unknown.

## Figure 4/5 saved-output audit: additional gate check
The author uploaded `polar_mask_case_diagnostic.json` after running the script. A compact, path-scrubbed copy of the measured results is stored in `polar_mask_case_initial_findings.json` (2026-11-20).

- Figure 4: 448 / 7680 cells had Polar flux >= 1; 7232 below threshold. Above-threshold saved-output maximum absolute difference: 3.52e-7. Of the low-flux cells, 7060 corresponded to OVATION>0 and 172 to OVATION<=0.
- Figure 5: 1317 / 7680 cells had Polar flux >= 1; 6363 below threshold. Above-threshold maximum absolute difference: 5.43e-6. Of the low-flux cells, 6188 corresponded to OVATION>0 and 175 to OVATION<=0.
- These values show very close preservation of above-threshold measurements but **do not** prove whether the actual inference mask was gated by OVATION. The old `repaired_polar.py` and exploratory `test.diff_polar.ipynb` contain different mask rules; figure-generating checkpoint provenance is not yet established.

The `check_polar_mask_cases.py` script now also reports the saved reconstruction difference **within low-flux, OVATION-nonpositive cells**. This is the discriminating diagnostic: a strictly OVATION-gated mask would preserve those cells, while material changes would contradict that hypothesis. An unchanged cell cannot establish the mask by itself.

After pulling the newest main branch, rerun:

```bash
python paper_modif/Second_revision/R2-5_R3-2_R3-11/check_polar_mask_cases.py
```

Inspect `low_flux_output_change_OVATION_nonpositive` in both Figure 4 and Figure 5. Before finalizing the manuscript, also verify the exact preprocessing code and checkpoint associated with the saved figures. Importantly, the very large flux-threshold exclusions are *not* proof that the corresponding satellite pixels were truly unobserved.

## Discriminating gate result (20 November 2026)
The second author-run `polar_mask_case_diagnostic(1).json` demonstrated that, among Polar cells with flux < 1 erg cm^-2 s^-1 **and OVATION <= 0**, saved reconstructions changed beyond 1e-4 erg cm^-2 s^-1 in **all 172/172 cells** (Figure 4) and **all 175/175 cells** (Figure 5). Mean absolute changes were 0.0518037 and 0.0531314 erg cm^-2 s^-1. By contrast, the max saved-output differences for flux >= 1 cells were only 3.52e-7 and 5.43e-6. See `polar_mask_case_discriminating_findings.json` for the source-grounded numeric summary.

This evidence **contradicts a strict OVATION-positive gate on the Polar reconstruction mask**. It is consistent with the legacy `repaired_polar.py` flux-threshold rule. The manuscript's previous claim that Polar cells were reconstructed only where OVATION indicated possible emission should be corrected. Importantly, the below-threshold cells include unknown observational support and may include valid weak precipitation; their abundance (94.17% and 82.85% of the grid) does not mean the satellite lacked coverage over all those cells. A scientific test of an observation-support-based Polar mask requires more than this saved-output comparison.

The Word manuscript was updated separately (data selection, Section 3.3, Polar results, and discussion) with the mask rule and limitation. This archive does not include that manuscript file because manuscript delivery is managed separately from experimental scripts. The first-batch sampling mechanism and single-key attention audits are unchanged; R3-7 still needs a conditioning ablation for an empirical claim.
