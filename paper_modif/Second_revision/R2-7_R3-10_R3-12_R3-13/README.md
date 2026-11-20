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
