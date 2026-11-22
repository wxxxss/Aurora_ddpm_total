# Third batch — independent satellite validation and direct-OVATION controls

Reviewer comments: **R2-1, R2-2, R2-3, R3-3, R3-4, R3-5**.

**Response-letter rule:** No reviewer-specific English Response is drafted or delivered until the corresponding issue is supported by completed analysis and the manuscript has been checked for consistency. Scripts and planned experiments are NOT evidence of completed reviewer resolution.

## Why a new experiment is necessary

The corrected-grid, orbit-mean second-round audit used **two real DMSP F16/SSUSI EDR acquisitions**, with 12 deliberately aurorally active held-out sectors nested within those acquisitions. The conditional DDPM's mean RMSE was 3.1924, versus 3.1470 for direct OVATION Prime and 3.0143 for interpolation; peak and gradient ratios were ~0.114 and ~0.098. This does NOT show a consistent advantage over those baselines. The synthetic OVATION validation is not independent of the training target generator.

The third batch adds (1) previously unused real EDR acquisitions selected by acquisition time, (2) flux-blind random sector holdouts as the *primary* benchmark, (3) an exploratory active-region holdout protocol separately labeled, (4) common exact holdout masks for conditional DDPM, unconditional DDPM, time-matched direct OVATION, observation-fused OVATION, and spatial interpolation, and (5) case-clustered inference rather than misleading pixel/sector bootstrap intervals.

Time-selected EDR acquisitions are distinct **acquisition clusters**; they are NOT guaranteed statistically independent geomagnetic storms. Broader storm/solar-cycle extrapolation still requires different conditions and years.

## Steps on the author's data server

From repository root, with NumPy, Pandas, SciPy, PyTorch and `netCDF4`, plus original SSUSI EDR inputs, the trained checkpoints and properly aligned OMNI/OVATION arrays.

**1. Preview the independent acquisition selection — no model or raw files are modified.**

```bash
git pull --ff-only origin main
python -m unittest discover -s paper_modif/Second_revision/R2-1_R2-2_R2-3_R3-3_R3-4_R3-5 -p 'test_*.py'
python paper_modif/Second_revision/R2-1_R2-2_R2-3_R3-3_R3-4_R3-5/prepare_multievent_ssusi.py --dry-run
```

Preview output: `independent_cases/catalog_preview.json`. Inspect `n_eligible_month_paths`, `n_selected`, and the dates. The first-round development orbits **06228** and **06263** are excluded by default.

**2. Prepare the center-aligned 80×96 real-observation arrays only if at least six appropriately separated EDRs are available.**

```bash
python paper_modif/Second_revision/R2-1_R2-2_R2-3_R3-3_R3-4_R3-5/prepare_multievent_ssusi.py --max-events 12 --min-events 6 --min-gap-hours 24
```

Produces case-specific `*_swath_case.npz`, `*_metadata.json` and `independent_cases/manifest.json`. Selection does not use flux amplitude or model quality. It refuses to overwrite saved inputs.

**3. Run the primary, brightness-blind real-observation holdout comparison.**

```bash
python paper_modif/Second_revision/R2-1_R2-2_R2-3_R3-3_R3-4_R3-5/multievent_heldout_benchmark.py --selection-mode random --windows-per-width 1
```

Outputs under `benchmarks/random_primary/`: `metrics_per_holdout.csv`, `metrics_summary.csv` (sector-level descriptive means; deliberately no sector-bootstrap intervals), `selected_holdouts.json`, `benchmark_summary.json`, `protocol_verification.json`, and diagnostic plots. The script reuses the first-revision DDPM implementation and checkpoint defaults; custom weights may be supplied with `--cond-checkpoint`, `--uncond-checkpoint`, `--cond-norm`, and `--uncond-norm`. The January 2005 default OVATION path is required to match the catalog month.

**4. Compute correctly paired acquisition-level effect sizes.**

```bash
python paper_modif/Second_revision/R2-1_R2-2_R2-3_R3-3_R3-4_R3-5/event_level_statistics.py
```

Outputs `metrics_by_acquisition.csv`, `paired_acquisition_differences.csv`, and `event_level_summary.json`. If fewer than eight finite, separately acquired events are available, interval estimates are suppressed. With eight or more, case-level cluster-bootstrap intervals are marked conditional on the selected acquisitions; they still do not prove independent geomagnetic storms.

**5. OPTIONAL: rerun an explicitly labeled active-aurora stress test for R3-5.** This SELECTS high-signal sectors using the held-out target field and thus is NOT an unbiased primary validation set.

```bash
python paper_modif/Second_revision/R2-1_R2-2_R2-3_R3-3_R3-4_R3-5/multievent_heldout_benchmark.py --selection-mode active --windows-per-width 1 --output-root paper_modif/Second_revision/R2-1_R2-2_R2-3_R3-3_R3-4_R3-5/benchmarks/active_stress
python paper_modif/Second_revision/R2-1_R2-2_R2-3_R3-3_R3-4_R3-5/event_level_statistics.py --metrics paper_modif/Second_revision/R2-1_R2-2_R2-3_R3-3_R3-4_R3-5/benchmarks/active_stress/metrics_per_holdout.csv
```

## Baseline definitions and leakage checks

- **Direct OVATION Prime:** time-aligned *1-minute maps averaged over the same EDR acquisition interval*, scored on the withheld real SSUSI pixels only.
- **Observed-fused OVATION:** preserves known SSUSI input pixels elsewhere, using OVATION only in missing regions. Its values and metrics inside the hidden region must equal direct OVATION; it is NOT a second independently predictive baseline.
- **Interpolation:** spatial interpolation from available observations only, respecting periodic MLT.
- **Conditional/unconditional DDPM:** receive an input with withheld targets set to zero and a strict `1=observed, 0=missing` mask. The sampler sanitizes missing cells internally. Neither receives the hidden target through inference inputs.
- **Random primary protocol:** random MLT sector selection depends on EDR-native `UT_N` support, latitude and predefined RNG, not on withheld flux amplitude.
- **Active stress protocol:** deliberately selects energetic sectors; compare separately and report the selection rule.
- **Statistical unit:** an EDR acquisition / time cluster; multiple sector widths, neighboring pixels, and contiguous MLT segments from the same orbit are correlated.

## Required outputs to return to this chat

First return `catalog_preview.json` or its console summary so we can establish whether sufficiently many eligible 2005-01 EDRs exist. After confirmed successful runs, send `manifest.json`, `benchmark_summary.json`, `protocol_verification.json`, `metrics_per_holdout.csv`, `event_level_summary.json`, and a small sample of diagnostic images for interpretation. **Do not send trained weights, private raw EDR arrays or confidential local paths if not necessary.**

## Scope and open questions

No third-batch experiments were executed on the private server at code-creation time. No statistical superiority is claimed. Real SSUSI electron energy flux is an EDR product and may itself have calibration and temporal-integration biases. New acquisitions in one month do not guarantee independence across magnetic storms or generalization across years or Kp regimes. Full paper/Response edits follow *only after* the tests have been completed, metrics verified, and contradictions with the main text resolved.

## Filename rollover fix (2026-11-22)

The first catalog dry-run was blocked by an SSUSI EDR filename with identical start/end year-day tokens but a crossing-midnight clock interval:
`2005244T235954-2005244T014123-REV09652`. The actual acquisition spans **2005-09-01 23:59:54 UTC through 2005-09-02 01:41:23 UTC**. The catalog parser now adds one calendar day to the end only when the date tokens are the same and the clocks imply a plausible late-night/early-morning rollover (start hour >=22, end hour <=4). The resulting duration must also be between 1 minute and 4 hours. Correctly dated cross-year intervals remain unchanged. Implausible filename times are listed in `n_rejected_invalid_timing` / `rejected_invalid_timing_examples` without stopping the catalog scan. The number of normalized cross-midnight names is recorded in `n_corrected_midnight_rollovers`. Four regression tests were added.

**Rerun only the catalog checks first**, not the model:

```bash
git pull --ff-only origin main
python -m unittest discover -s paper_modif/Second_revision/R2-1_R2-2_R2-3_R3-3_R3-4_R3-5 -p 'test_*.py'
python paper_modif/Second_revision/R2-1_R2-2_R2-3_R3-3_R3-4_R3-5/prepare_multievent_ssusi.py --dry-run
```

Only after the dry-run lists enough eligible, temporally separated January acquisitions should the user execute the actual preparation/benchmark. No reviewer Response is issued from this parser correction.
