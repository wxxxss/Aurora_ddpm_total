# Second revision, batch 2 — R2-7 / R3-10 / R3-12 / R3-13

## Purpose
Audit the physical meaning of Polar/UVI and SSUSI measurements, grid and mask construction, LBHL-to-flux scaling, and the statistical unit of independent replication. Avoid treating data-quality thresholds as physical coverage and spatially correlated pixels as independent experimental realizations.

## Verified from first-revision source and archived outputs
- Northern Hemisphere target image: 80 MLAT centers spanning 50°–90° inclusive, spacing **40/79 ≈ 0.50633°**, and 96 MLT centers at 0,0.25,...,23.75 h (24-h cyclic, no duplicated 24 h center).
- First-revision SSUSI `prepare_ssusi_swath_cases.py`: northern `ENERGY_FLUX_NORTH_MAP`; native support requires finite `UT_N > 0`, flux, MLAT and MLT; direct target-grid binning averages multiple native measurements per cell using `sum/count`, no interpolation into unsupported cells. The original EDR quality flag and output-support provenance should remain under review.
- Archived swath summary reports 56.5755% direct grid support for Figure 6 and 55.7161% for Figure 7.
- Polar/UVI is converted using an **approximate** empirical 110 R per erg cm⁻² s⁻¹ LBHL scaling, *not* a direct measurement of electron flux. The 40 erg cm⁻² s⁻¹ artifact cutoff is a processing heuristic and cannot presently be defended as a physically universal threshold; without uncensored raw calibrated data, the cutoff's sensitivity cannot be quantified.
- Polar Figure 4/5 preserved fractions under the operational threshold `flux>=1` are 448/7680 (5.83%) and 1317/7680 (17.15%). They are **not** physical instrument coverage fractions, unlike the SSUSI direct-support fractions.
- The previously reported 8202 gradient edges and 79 PSD segments come from only two EDR events. The 79 PSD segments comprise 45 for Figure6 and 34 for Figure7; adjacent latitude rows and segments are correlated.
- Event-level descriptive PSD differences (SSUSI − OVATION) in high-frequency fraction are **+0.0702** for Figure6 and **−0.0322** for Figure7. Spectral-slope differences are **−0.0209** and **−1.1237**, respectively. The pooled differences +0.0261 and −0.4955 are *descriptive segment-weighted* values, not event-level generalizations.
- There is no evidence that the reported 79-segment Wilcoxon p values or bootstrap intervals have proper event-level coverage. The manuscript must not present them as independent-event inferential statistics.

## Run on the user's data environment
```bash
git pull --ff-only origin main
python paper_modif/Second_revision/R2-7_R3-10_R3-12_R3-13/audit_statistical_units.py
python paper_modif/Second_revision/R2-7_R3-10_R3-12_R3-13/audit_grid_calibration.py
```

Outputs: `statistical_units_audit.json`, `psd_event_level_descriptive.csv` and `grid_calibration_audit.json`. The last script reports missing private inputs rather than inventing measurements.

Sanity tests (no satellite data required):

```bash
python -m unittest discover -s paper_modif/Second_revision/R2-7_R3-10_R3-12_R3-13 -p 'test_*.py'
```

The `audit_grid_calibration.py` script checks both the archived `First_revision` prepared NPZ path and the historical `paper_modif/R2-1_R2-3/swath_prepared_cases` path for local, untracked case arrays. It emits `missing_inputs` if neither is available. Do not interpret missing outputs as zero or fully checked coverage.

Optional, **only if calibrated uncensored physical Polar flux is available**:

```bash
python paper_modif/Second_revision/R2-7_R3-10_R3-12_R3-13/audit_uncensored_polar_threshold.py --uncensored-flux /path/to/pre_QC_calibrated_LBHL_flux.npy --thresholds 20 30 40 50 60
```

Its output is a count-based cutoff sensitivity, not a calibrated uncertainty interval or a proof that bright values are invalid. Never supply post-QC gridded products for this pre-QC test; otherwise the conclusion would be circular. Do not commit private observation arrays.


## Explicit remaining requirements
1. Recover the **native-to-grid Polar/UVI conversion code** for the preprocessed flux products, including how duplicate native pixels and missing/quality flags were handled. That mapping cannot be established from the current committed code alone; do not claim it has been independently audited.
2. Verify the **40 erg cm⁻² s⁻¹ cutoff** using the uncensored, calibrated Polar brightness/flux values and available quality flags. Perform threshold sensitivity (e.g. 20, 40, 60) before asserting a justified optimum.
3. Broaden the independent-event SSUSI sample beyond two EDR files if inferential claims about spectral differences are essential. No event-level 95% CI/p value should be asserted using just these two.
4. R2-7 visual differences are plausibly affected by the very different definitions and retained fractions of Polar threshold-constraint pixels and SSUSI swath support, plus frame-vs-orbit timing and instrument products; relative accuracy cannot be inferred from visual appearance alone.

**Revision status**: descriptive/statistical audit and manuscript clarification can be completed from repository outputs; calibration threshold validation and full Polar native-grid reproduction remain open.

## Author-run audit confirmation (2026-11-20)
The author uploaded `grid_calibration_audit.json` and `statistical_units_audit.json`. Their derived, path-scrubbed results are stored in `verified_server_audit_summary.json`. The former reported `missing_inputs=[]` and read both private SSUSI prepared `.npz` files and the two gridded Polar cases.

- Current prepared SSUSI cases have **4355/7680 (56.7057%)** and **4290/7680 (55.8594%)** supported target cells for Figure6 and Figure7, respectively. **4110/4355 (94.37%)** and **4034/4290 (94.03%)** supported cells contain two or more original samples, with **42268** and **41805** total native hits in the supported grid cells.
- The earlier first-round `ssusi_swath_summary_v2.csv` reported **4345/7680 (56.5755%)** and **4279/7680 (55.7161%)** as direct support. The small **+10/+11-cell discrepancy** is real and the cause has **not** been established. The updated `audit_grid_calibration.py` compares the archived CSV with current prepared input counts under `prior_swath_audit_comparison`. Use the prepared-`.npz` figures for current input descriptions, but verify the precise map/checkpoint provenance before attributing them to all previously saved panels.
- Postprocessed Polar maps contain no flux > 40; maxima were 4.0439 and 32.3327 erg cm^-2 s^-1. This is **not** a sensitivity test of the pre-screening 40 cutoff, because the surviving product is already processed. Figure5 contains 75 cells > 20, illustrating why an uncensored sensitivity test cannot be inferred from a censored input.
- Event-unit audit confirmed **2 EDR cases**, **8202 spatial gradient edges**, **79 correlated PSD segments** (45+34). High-frequency fraction differences (SSUSI minus OVATION) are +0.07020 / -0.03223 for the two cases; slope differences are -0.02091 / -1.12370. The unweighted mean of two case averages is **+0.01898** / **-0.57230**, distinct from the 79-segment weighted pooled differences **+0.02611** / **-0.49552**. No event-level inferential p value or CI is justified from only two cases.

### Re-run updated support audit (optional follow-up)
```bash
python paper_modif/Second_revision/R2-7_R3-10_R3-12_R3-13/audit_grid_calibration.py
```
Inspect the output `prior_swath_audit_comparison`. Reconciliation of the 10/11-cell difference and recovery of the original Polar native preprocessing/uncensored LBHL flux remain open. Synthetic unit tests were committed, but the user has not supplied their test-run output.

## Newly identified R3-10 alignment inconsistency (must resolve before finalizing)
Code comparison establishes two distinct MLT mappings:

- `First_revision/R2-1_R2-3/prepare_ssusi_swath_cases.py` uses `np.digitize(mlt, edges)`, where MLT edges are **0, 0.25, 0.50, ... 24 h**. With nominal grid centers `0,0.25,...,23.75`, this creates bins *offset from the nominal centers* (for example 0.20 h maps to index 0).
- `First_revision/R2-1_R2-3/audit_ssusi_swath_v2.py` uses `np.rint(mlt/0.25) % 96`, associating samples with the **nearest** nominal grid center (0.20 h maps to index 1). The boundaries of these bins are shifted by half an MLT interval (0.125 h = 7.5 min) relative to the edge-zero convention.
- Their archived direct-support percentages differ from the prepared NPZ by 10 and 11 cells, respectively, but support-count differences alone **cannot quantify how many samples/cells have changed assignments**. Different native filtering may also matter. Do not assert the half-bin shift is the sole explanation until the raw-EDR comparison is run.

New comparison script, run from the repository root with `netCDF4` available and the original EDR files on disk:

```bash
python paper_modif/Second_revision/R2-7_R3-10_R3-12_R3-13/audit_mlt_binning_alignment.py
```

Produces `mlt_binning_alignment_audit.json`: direct support, symmetric difference of support maps, original-sample assignment differences, flux difference on shared supported cells, and whether the locally prepared NPZ agrees with either convention. **Do not silently switch the production pipeline to a new binning rule or claim unchanged performance.** If this changes input products, the affected SSUSI figures, baseline comparisons and held-out evaluation must be recomputed on the corrected grid. The current manuscript is therefore a cumulative *working draft* pending this specific reproducibility check.

## Unit-test fix: latitude midpoint precision (2026-11-20)
The initial `test_center_alignment.py` used a synthetic input at MLAT=70.0°, which lies exactly between two of the 80 grid latitude centers. The implementation computed `LAT_STEP` from `np.linspace`, while the test independently computed the ideal `40/79` step; floating-point rounding put the same midpoint into different adjacent latitude rows. This was a **test setup mismatch**, not evidence that the cyclic MLT mapping is incorrect. The test now supplies the exact declared latitude-grid center `LAT_CENTERS[40]` and checks MLT assignment/wrap at that row. No production preprocessing code was changed in this fix.

Rerun `python -m unittest discover -s paper_modif/Second_revision/R2-7_R3-10_R3-12_R3-13 -p 'test_*.py'` before generating new inputs. The assistant validated the expected numpy index mapping locally; the complete suite still requires the user's environment to confirm.

## Native-EDR binning comparison: confirmed by author (2026-11-20)
The author uploaded `mlt_binning_alignment_audit.json`. A path-scrubbed, numeric copy is in `mlt_binning_alignment_verified.json`. The source mapping is now confirmed:

| Event | Edge-zero supported | Centered supported | Native samples assigned to different target cells | Symmetric difference (support mask) | MAE on jointly supported flux cells |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fig. 6 | 4355 | 4345 | 21005/42268 (49.69%) | 224 | 0.09289 |
| Fig. 7 | 4290 | 4279 | 20721/41805 (49.57%) | 231 | 0.15371 |

The old prepared `.npz` masks match **edge-zero** exactly. The older swath-v2 support summaries match **nearest-center** exactly. The 10/11-cell count discrepancy hid a more consequential ~50% reassignment of native samples and flux changes within shared cells. Differences are not merely rounding of coverage percentages. The mean absolute differences shown are measured in the energy-flux units of the SSUSI EDR.

**Do not describe the old Figure6/7 reconstructions as center-aligned results.** The manuscript is a working version until the affected reconstructions and diagnostics are regenerated, and the SSUSI figures, Tables, response claims and spatial metrics are synchronized.

### New center-aligned pipeline (without overwriting First_revision)
Run from repository root. Files are generated only under `Second_revision/R2-7_R3-10_R3-12_R3-13/`.

1. Synthetic tests (numpy and netCDF4):

```bash
python -m unittest discover -s paper_modif/Second_revision/R2-7_R3-10_R3-12_R3-13 -p 'test_*.py'
```

2. Generate the center-aligned case inputs with `UT_N>0` and direct native-cell averaging. The script asserts the expected centered support totals (4345 and 4279) and will refuse to overwrite an existing dataset unless `--overwrite` is specified deliberately.

```bash
python paper_modif/Second_revision/R2-7_R3-10_R3-12_R3-13/prepare_center_aligned_ssusi.py
```

Check `center_aligned_inputs/center_alignment_manifest.json`, with **4345** and **4279** directly supported cells; first-revision inputs and results are untouched.

3. Re-run the paper-model reconstructions (requires the author's trained weights, normalization, OMNI/OVATION and plotting dependencies). The earlier script is re-used, with only inputs/outputs redirected:

```bash
python paper_modif/First_revision/R2-1_R2-3/rerun_ssusi_reconstruction.py \
  --prepared-root paper_modif/Second_revision/R2-7_R3-10_R3-12_R3-13/center_aligned_inputs \
  --output-root paper_modif/Second_revision/R2-7_R3-10_R3-12_R3-13/center_aligned_reconstruction
```

If weights are elsewhere, supply `--checkpoint /path/to/weights.pth --norm-params /path/to/norm_params.pkl --device npu:0`. The old historical comparison figures should never be silently overwritten.

4. Re-run the direct-OVATION / interpolation / conditional / unconditional held-out sector benchmark against the **regridded real SSUSI** support (requires both sets of weights):

```bash
python paper_modif/First_revision/R2-1_R2-3/ssusi_heldout_benchmark.py \
  --prepared-root paper_modif/Second_revision/R2-7_R3-10_R3-12_R3-13/center_aligned_inputs \
  --output-root paper_modif/Second_revision/R2-7_R3-10_R3-12_R3-13/center_aligned_heldout
```

5. Recompute the Figure 8 local gradients and PSD segments on the same regridded support and use the **new** held-out metrics rather than mixing the old sector scores with the new input:

```bash
python paper_modif/First_revision/R1-1/smoothness_bias_analysis.py \
  --prepared-root paper_modif/Second_revision/R2-7_R3-10_R3-12_R3-13/center_aligned_inputs \
  --heldout-metrics paper_modif/Second_revision/R2-7_R3-10_R3-12_R3-13/center_aligned_heldout/metrics_per_holdout.csv \
  --output-root paper_modif/Second_revision/R2-7_R3-10_R3-12_R3-13/center_aligned_smoothness
```

This fifth stage may output pixel/segment bootstrap confidence intervals as part of the historical script; **do not restore those as event-level statistical claims in the manuscript**. Report descriptive event-level statistics and holdout metrics with independent sampling units clearly identified.

### Required output provenance for manuscript revision
Return the following generated text artifacts (not private raw EDR arrays): `center_alignment_manifest.json`, the two `Figure*_reconstruction_summary.json`, `center_aligned_heldout/benchmark_summary.json`, and `center_aligned_smoothness/smoothness_bias_summary.json`. Recheck the figures and the remaining first-revision statistics on the corrected grid before updating the paper's numeric results.

**Status:** preprocessing correction *implemented*, but no center-aligned reconstruction, benchmark or PSD rerun has yet been reported by the author. No inference-side improvement or unchanged scientific metric is claimed.

## Corrected-grid author rerun and hidden temporal mismatch (2026-11-22)
The author supplied corrected-grid `reconstruction_summary.json`, `benchmark_summary.json`, `metrics_per_holdout.csv`, `metrics_summary.csv`, `smoothness_bias_summary.json`, `gradient_summary.csv`, `gradient_comparison.csv`, Figure6/7 triptychs, and the structure-diagnostic plot. Verified summary: `center_aligned_rerun_findings.json`.

**Reconstruction and evaluation are not yet fully synchronized:** the first-revision figure-maker reconstructed Fig6/7 with the *nearest single 5-min OMNI record* (at 15:35/03:05 UT), and used the same OMNI *array position* (187/901) to index a 1-min OVATION series. Those indices do not refer to the corresponding solar-wind time in the 1-min OVATION series. In contrast, the separate real-observation heldout benchmark and spatial PSD audit use acquisition-interval means. The uploaded Figure6/7 triptychs are therefore **not final time-matched figure panels**. Retain the previous run as a diagnostic; do not describe it as a time-consistent OVATION interval mean.

The new `rerun_ssusi_reconstruction_timecorrected.py` clones the first-round figure-making procedure without overwriting its historical source and uses **UTC acquisition-interval means for OMNI conditioning and the OVATION baseline**. It explicitly rejects incomplete 1-min OVATION orbit coverage and saves the indices used in the JSON result.

```bash
git pull --ff-only origin main
python paper_modif/Second_revision/R2-7_R3-10_R3-12_R3-13/rerun_ssusi_reconstruction_timecorrected.py
```

After the new reconstruction finishes, verify its interval-frame alignment and observation-preservation invariants:

```bash
python paper_modif/Second_revision/R2-7_R3-10_R3-12_R3-13/verify_timecorrected_reconstruction.py
```

Required results: `center_aligned_reconstruction_timecorrected/reconstruction_summary.json` plus the fresh `Figure6/Figure6_comparison_vmax5.png` and `Figure7/Figure7_comparison_vmax5.png`. Model weights and normalization must match the existing tested checkpoint. This **does not** require rerunning the separate heldout benchmark or the spatial PSD analysis, since both already use orbit-mean time alignment.

### Corrected grid: completed real-observation benchmark
The delivered benchmark contains **12 MLT-sector holdouts** from two SSUSI EDR acquisitions, evaluated across **four methods** (48 method evaluations). Descriptive means are:

| Method | RMSE | MAE | Pearson r | Peak ratio | Gradient ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Conditional DDPM | 3.1924 | 1.3316 | 0.4518 | 0.1140 | 0.0981 |
| Unconditional DDPM | 3.2714 | 1.2836 | 0.5429 | 0.0972 | 0.1227 |
| Direct interval-mean OVATION | 3.1470 | 1.6128 | 0.1911 | 0.2486 | 0.1851 |
| Spatial interpolation | 3.0143 | 1.1980 | 0.3827 | 0.4929 | 0.3593 |

The conditional model does *not* uniformly outperform direct OVATION or interpolation on real measured flux; its peak and gradient retention remain poor. The 12 sectors are not independent events and were selected for coverage and auroral signal. Do not use a naive bootstrap across sectors for independent-event significance.

The updated structure analysis has **8194** support-safe gradient edges (**4103+4091**) and **79** PSD segments (**44+35**) from the same two EDRs. Log-gradient KS distance is **0.6842**, Wasserstein distance **0.1086**, and pooled SSUSI-to-OVATION gradient ratios **4.304** and **4.549** at the 95th and 99th percentiles, respectively. Pooled segment-weighted high-frequency PSD-fraction difference is **+0.00565** and spectral-slope difference is **−0.7300**. Earlier event-specific PSD values (45/34 segments) are invalid for these new inputs. To report revised per-event values, also provide **`center_aligned_smoothness/psd_comparison.csv` and `psd_segment_metrics.csv`**; these files were not included in the upload.

The cumulative manuscript now reports the corrected-grid descriptive results and a direct-OVATION/SSUSI heldout baseline table, but the triptych figures remain pending the interval-time correction described above. **No model superiority or updated image quality is claimed on the basis of reruns not yet performed.**
