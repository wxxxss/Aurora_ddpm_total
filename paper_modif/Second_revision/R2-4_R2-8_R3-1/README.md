# Fourth revision batch — R2-4 / R2-8 / R3-1

**Status: experimental setup + source-code audit only.** No new Kp>5 real-satellite results have been produced; no reviewer Response will be given until findings and manuscript changes are fully verified.

## Questions

- **R2-4**: What happens under genuinely strong geomagnetic activity (**strictly Kp>5**) rather than combining moderate activity Kp>=4 with storms? Do peaks and sharp gradients degrade, and does conditional DDPM beat direct OVATION, unconditional DDPM, and interpolation on the SAME truly observed withheld SSUSI cells?
- **R2-8**: Is a model nominally trained with only April–May 1996 OVATION-derived targets robust to later periods, and what evidence would justify or require longer training? Merely evaluating an OVATION-targeted model in more synthetic years cannot demonstrate real-satellite generalization.
- **R3-1**: Is training disjoint in calendar time from real-satellite evaluation, and are the 1996 training **aurora frames actually UTC-matched to the corresponding OMNI conditions**? The source loader currently concatenates raw arrays and truncates their lengths; this MUST be audited before interpreting conditioning and generalization.

## Confirmed code and historical results

1. Public \`train.py\` config: \`years=[1996]\`, \`months=range(4,6)\` (April–May only), train/val ratio 0.9. The \`data/dataset_diff2.py\` loader slices chronologically *after* concatenating aurora and solar-wind arrays independently, without reading or joining timestamp indices. This is a **potential serious temporal misalignment** if the original 1996 OVATION cube is minute-cadence but OMNI is five-minute. The true 1996 file shapes and checkpoint lineage have NOT YET been inspected by the assistant.
2. First-revision \`R1-2/r1_2_results/metrics_per_sample.csv\` has 240 distinct OVATION-generated target timestamps (80 per 2001, 2005, 2009) and 720 evaluation rows. Only **17** distinct tested timestamps satisfy **strict Kp>5** (2001:10, 2005:5, 2009:2). Earlier Kp>=4 pooled grouping is NOT the reviewer-requested strong-activity class. These 17 synthetic test instances do not establish real aurora accuracy.
3. First-revision synthetic-domain conditional mean RMSE is **0.03423** for Kp<=3 (174 timepoints), **0.09271** for 3<Kp<=5 (49), **0.26777** for Kp>5 (17), as recalculated from tracked CSV. These are different target distributions; do NOT interpret the absolute RMSE increase as causal storm degradation or as new satellite evidence. Archived facts: \`prior_multiyear_archived_facts.json\`.
4. The third-batch independent real-SSUSI validation used 12 January 2005 F16 EDR acquisitions (36 selected measured-pixel masks per random and active protocol) and showed substantial peak/gradient attenuation and no consistent improvement over interpolation or unconditional DDPM. Those orbit metrics already exist on the author's server; we first attach strict real-time Kp labels rather than rerunning the same predictions.

## Required first stage on author data server — **NO model inference yet**

From the repository root:

\`\`\`bash
cd /home/docker/code/Aurora_DDPM_total
git pull --ff-only origin main

python -m unittest discover \
  -s paper_modif/Second_revision/R2-4_R2-8_R3-1 -p 'test_*.py'

# PRIORITY: verify 1996 aurora-versus-OMNI training timestamp alignment
python paper_modif/Second_revision/R2-4_R2-8_R3-1/audit_training_alignment.py

# Recalculate the archived 2001/2005/2009 OVATION-targeted Kp>5 results
python paper_modif/Second_revision/R2-4_R2-8_R3-1/audit_prior_multiyear.py

# Label the existing 12 REAL SSUSI EDRs using contemporaneous Kp (no DDPM)
python paper_modif/Second_revision/R2-4_R2-8_R3-1/audit_existing_real_kp.py

# Count previously unused Kp>5 EDR acquisitions and DISTINCT UTC days in January
python paper_modif/Second_revision/R2-4_R2-8_R3-1/prepare_high_kp_real_ssusi.py --dry-run
\`\`\`

Send back these four JSON reports (they contain numerical results, not private raw inputs):
- \`training_alignment_audit.json\`
- \`prior_multiyear_strict_kp_audit.json\`
- \`existing_real_kp/existing_real_kp_summary.json\`
- \`kp_stratified_cases/kp_catalog_preview.json\`

**If \`training_alignment_audit.json\` reports time misalignment**, STOP new checkpoint inference and DO NOT claim 1996 solar-wind-conditioned training was correctly matched. The source loader needs fixing, and the *actual checkpoint's training lineage* (training script/version, original arrays, weight path, and run logs) must be checked. A source-code defect does NOT automatically prove the published checkpoint was trained with that exact code. If source temporal cadence is not inferable, report the data gap rather than guessing. Re-training may be necessary, but must not be triggered until this audit and checkpoint lineage are established.

**If \`kp_catalog_preview.json\` reports fewer than three distinct Kp>5 calendar dates**, January-only storm evaluation is insufficient; expand to suitable additional months/years with fully timestamped OMNI and matching OVATION before using any population-level storm robustness claim.

## Second stage — only after training alignment and checkpoint provenance are confirmed

If the 2005-01 catalogue has at least 3 distinct high-Kp days and at least matching quiet days:

\`\`\`bash
python paper_modif/Second_revision/R2-4_R2-8_R3-1/prepare_high_kp_real_ssusi.py --prepare

# Verify selected SSUSI grids, fixed holdouts and full UTC support, NO model
python paper_modif/Second_revision/R2-1_R2-2_R2-3_R3-3_R3-4_R3-5/preflight_multievent_benchmark.py \
  --prepared-root paper_modif/Second_revision/R2-4_R2-8_R3-1/kp_stratified_cases \
  --output paper_modif/Second_revision/R2-4_R2-8_R3-1/kp_preflight_report.json

# Reuse tested third-batch benchmark with new activity-selected EDR cohort,
# but flux-BLIND RANDOM spatial holdouts; never overwrite original outputs.
python paper_modif/Second_revision/R2-1_R2-2_R2-3_R3-3_R3-4_R3-5/multievent_heldout_benchmark.py \
  --prepared-root paper_modif/Second_revision/R2-4_R2-8_R3-1/kp_stratified_cases \
  --output-root paper_modif/Second_revision/R2-4_R2-8_R3-1/kp_benchmark \
  --selection-mode random --min-events 6

# Analyze separate Kp>5 / Kp<=3 orbit means and paired baselines
python paper_modif/Second_revision/R2-4_R2-8_R3-1/analyze_high_kp_benchmark.py
\`\`\`

If preflight fails, do NOT force reconstruction by silently changing observation-coverage thresholds. Report the reasons. The Kp catalogue excludes **development orbits 06228/06263 and all 12 third-batch validation orbits**.

## Inferential safeguards

- \`Kp>5\` means strictly greater than 5 at the acquisition midpoint; \`Kp=5\` is NOT a storm-class measurement under this test. Each EDR's within-orbit Kp variation is recorded.
- The source first-round OMNI hourly product may fill short missing three-hour Kp blocks; the output source and time offsets are logged. Data provenance still matters.
- At most ONE selected storm EDR per UTC date, but different dates still can belong to the same disturbance. No nominal n=event p value will be presented without verifying independent storm episodes.
- Real withheld SSUSI pixels, not OVATION maps, are the ground truth for R2-4; condition/input/mask must be matched exactly across methods.
- R2-8 and R3-1 may remain open after observational stress tests if 1996 training alignment, checkpoint lineage, or expanded-training comparison is unresolved.
- Existing six third-batch Responses should not be silently rewritten; if a proven training defect undermines the checkpoints used there, the manuscript and all affected responses require a new methodological re-evaluation.

## No reviewer Response until resolved

In line with the user's instruction, do not deliver R2-4, R2-8, or R3-1 English reviewer reply text until original input alignment, checkpoint provenance, Kp>5 satellite results, and cumulative manuscript consistency are verified.


## Author-audited stage-one results — 2026-11-23 (new)

The author supplied all four JSON reports requested above. A sanitized synthesis
is committed in \`author_audits_2026_11_23_verified.json\`.

**1996 cadence concern: provisionally ruled out.** April 1996 has 8,640
OVATION frames and 8,640 five-minute OMNI samples; May has 8,928 and 8,928,
respectively. The OMNI UTC axis is regular at 5-minute intervals and begins
exactly on the month boundary; under the complete-month 5-minute OVATION frame
assumption, all **17,568** historical array pairs are aligned, with zero
discarded frames and maximum inferred time offset zero minutes. This rules out
the hypothesized 1-minute-vs-5-minute pairing bug for these particular files,
**but it does not directly verify per-image stored timestamps or checkpoint
training lineage**. Do not announce retraining is necessary solely because the
loader lacks an explicit timestamp join.

**Original 12 independent-EDR satellite tests contained zero strict Kp>5
acquisitions**: 7 Kp<=3 and 5 with 3<Kp<=5. Thus third-batch real-satellite
results did *not* demonstrate storm-class robustness. Their Kp-stratified
random-sector RMSE for conditional DDPM was 1.2839 (7 quiet acquisitions) and
1.8100 (5 moderate acquisitions); for interpolation 1.3025 and 1.5225,
respectively. These are observational descriptors, not causal activity effects.

**New 2005-01 Kp-only discovery succeeded.** After excluding 14 previously
used orbits, the source inventory contained 47 additional F16 EDR files with
midpoint Kp>5 spanning 9 UTC dates, and 223 quiet files spanning 27 dates.
The prospectively selected validation cohort contains **6 high (Kp>5)**
and **6 quiet (Kp<=3)** new orbits, at most one per calendar date, none
identical to the old EDR orbits. Some selected high dates coincide with
earlier validation dates, so do not claim independence from all previous
space-weather events. Do not imply that any of the six high orbits has passed
the native SSUSI holdout coverage audit yet.

**First-round multi-year synthetic test:** 240 complete OVATION-target
timestamps in 2001/2005/2009; only 17 had Kp>5 (10/5/2 per year).
Mean conditional RMSE on synthetic targets was 0.0342 for Kp<=3,
0.0927 for 3<Kp<=5, and 0.2678 for strict Kp>5.
This is neither a real satellite storm result nor proof of monotonic causality.

### Next commands: original data preparation, no DDPM inference

\`\`\`bash
cd /home/docker/code/Aurora_DDPM_total
git pull --ff-only origin main

python -m unittest discover \
  -s paper_modif/Second_revision/R2-4_R2-8_R3-1 -p 'test_*.py'

# Hash checkpoint and training scripts. No model unpickling or private weight upload.
python paper_modif/Second_revision/R2-4_R2-8_R3-1/audit_checkpoint_provenance.py

# Materialize native, center-aligned 80x96 support masks for preselected
# 6 strong and 6 quiet previously unused F16 EDRs.
python paper_modif/Second_revision/R2-4_R2-8_R3-1/prepare_high_kp_real_ssusi.py --prepare

# Mandatory: ensure ALL selected EDRs have meaningful true observed
# holdout targets and matching OMNI/OVATION UTC data before model inference.
python paper_modif/Second_revision/R2-1_R2-2_R2-3_R3-3_R3-4_R3-5/preflight_multievent_benchmark.py \
  --prepared-root paper_modif/Second_revision/R2-4_R2-8_R3-1/kp_stratified_cases \
  --output paper_modif/Second_revision/R2-4_R2-8_R3-1/kp_preflight_report.json
\`\`\`

Send:
- \`checkpoint_provenance_audit.json\` (just a small file hash/manifest; do **not** share the raw model weights)
- \`kp_stratified_cases/manifest.json\` (it contains private source paths; remove paths if sharing publicly)
- \`kp_preflight_report.json\` (with \`ready_for_primary_inference\`)

**Only after the preflight passes** should the third-batch
\`multievent_heldout_benchmark.py\` be rerun on the NEW Kp-stratified manifest,
using a distinct \`kp_benchmark\` output directory. Use the run commands in
the "Second stage" section above. If checkpoint provenance remains
unverified, preserve that caveat regardless of the empirical performance
outcome; if evidence of incorrect training provenance arises, stop, correct
it and reassess affected third-batch claims.

The April–May 1996 train/validation split is the last chronological ~10%
for validation, but neighboring 5-minute maps are serially correlated.
External 2001/2005/2009 model-input years are calendar-disjoint from 1996,
yet OVATION-derived target agreement alone is insufficient to close R2-8
or R3-1 as assertions of real-satellite solar-cycle generalization.

### Interpretation guards

- The prior source \`train.py\` imports \`models.unet_v1\`, which is absent
  from the tracked GitHub source tree checked at batch initiation. The
  published checkpoint used by the benchmark may originate from a different
  local run. The provenance audit records this discrepancy; no assumption
  about actual weights is justified.
- R2-4 is not yet resolved by candidate availability or Kp matching.
- R2-8/R3-1 still require an honest boundary on 1996-only training and
  independent-year *measured* targets; extensive retraining should be
  decided after the Kp and provenance audits, not implied by synthetic
  reconstruction scores.
- Do NOT issue English reviewer Responses or claim new manuscript results
  until the real-Kp measurements and full-text consistency checks are complete.


## 2026-11-23 — completed 6+6 REAL SSUSI Kp holdout run, manuscript still interim

The author provided:
- \`kp_benchmark/benchmark_summary.json\`: 12 new January 2005 F16 EDR acquisitions, 3 **flux-blind random** holdout-sector widths (1, 2, 4 MLT h) per acquisition, 36 selected sectors, 180 evaluation rows, and the previously used conditional/unconditional checkpoints and normalization files;
- \`kp_benchmark/metrics_per_holdout.csv\`: actual measured withheld SSUSI target metrics for conditional DDPM, unconditional DDPM, direct OVATION, observed-fused OVATION and interpolation;
- \`kp_benchmark/kp_group_real_holdout_summary.json\`: case-equal mean statistics split by **Kp>5 (6 EDRs, Kp 5.3–6.0)** and **Kp<=3 (6 EDRs, Kp 0.7–3.0)**.

All 36 sector masks contain five matched method results and common target values. The fusion display and direct OVATION have the same withheld-pixel predictions. A source-data-independent numerical record has been committed as \`kp_6plus6_real_holdout_verified.json\`. The author's CSV was also independently recomputed and checked against the JSON summary. The **updated** \`analyze_high_kp_benchmark.py\` adds automatic verification of consistent withheld truth/coordinates, strict Kp groups and flux-blind selection, plus aggregate peak/gradient diagnostics and absolute peak error; the associated synthetic unit tests are in \`test_fourth_batch.py\`. Re-running this summary script does **not** require new model inference.

### Real observed-target results (erg cm^-2 s^-1)

| Model | 6 EDRs, Kp>5 RMSE | 6 EDRs, Kp<=3 RMSE | Kp>5 MAE | Kp>5 sector peak ratio | Kp>5 aggregate-gradient ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Conditional DDPM | **2.821** | **0.358** | 1.425 | 0.194 | 0.103 |
| Unconditional DDPM | 2.751 | 0.321 | 1.319 | 0.184 | 0.116 |
| Direct OVATION Prime | 3.172 | 0.510 | 1.843 | 0.366 | 0.241 |
| Spatial interpolation | 2.687 | 0.396 | 1.420 | 0.880 | 0.619 |

In the strong-activity group, conditional-DDPM minus baseline paired acquisition-level mean RMSE is **-0.351** vs direct OVATION (6/6 acquisition wins), **+0.069** vs unconditional DDPM (2/6 wins), and **+0.134** vs interpolation (3/6 wins). In quiet conditions, conditional diffusion has lower RMSE than direct OVATION in 6/6 acquisitions and lower RMSE than unconditional diffusion in **0/6**.

**Crucial real-data limitation:** mean true withheld flux is ~**1.535** erg cm^-2 s^-1 in Kp>5 acquisitions versus **0.117** in Kp<=3. Their gross RMSE difference is therefore **not a controlled causal effect of Kp**. A larger dynamic range and more intense precipitation lead to higher absolute error for all methods. The strong-group conditional model's ratio of mean predicted to mean measured peaks is **0.129** (interpolation: 0.868); its ratio of mean gradients is **0.103** (interpolation: 0.619), and its mean absolute peak error is **12.954** versus interpolation's **3.048**. This is not evidence of robust small-scale peak reconstruction. Seven quiet withheld sectors had reference gradient below 0.01, inflating the arithmetic mean of their gradient ratios (conditional 4.867) compared with the ratio of aggregate mean gradients (0.311).

The 12 test EDRs occupy separate dates **within January 2005**, but multiple acquisitions and dates may belong to the same space-weather episode. No p value, independent-storm replication count or cross-solar-cycle confidence interval is inferred.

### Cumulative manuscript and status

The cumulative Word working manuscript has been updated separately (31 pages, 11 tables, 20 retained original math objects), including a new Table 11 for the strict real SSUSI Kp groups, Section 2.5's cohort selection, Section 4.5's strict Kp synthetic-versus-measured distinction, and Abstract/Key Points/Discussion. Older Table 4 remains a **different**, balanced synthetic experiment with Kp>=4. The archived broader synthetic audit had 240 samples (80/year) with only 17 strict Kp>5 targets. These must not be substituted for the 12 real SSUSI acquisition results.

The experiment materially addresses the *measured-real-data* aspect of R2-4, but **R2-8 and R3-1 remain open**: the 1996-only training data and calendar-disjoint OVATION-based synthetic tests do not establish cross-year REAL satellite generalization, and hashes cannot establish the precise inference checkpoint's original training code/model provenance. The public train.py refers to a currently absent \`models/unet_v1.py\` and a historical output directory unlike the current checkpoint locations. This does not itself prove the checkpoint invalid; investigate original run logs, archived source and training commands. Do not issue premature Responses for open comments.

Re-audit the newly reported Kp metrics (no DDPM):
\`\`\`bash
git pull --ff-only origin main
python -m unittest discover -s paper_modif/Second_revision/R2-4_R2-8_R3-1 -p 'test_*.py'
python paper_modif/Second_revision/R2-4_R2-8_R3-1/analyze_high_kp_benchmark.py
\`\`\`

Next scientifically necessary decision: recover training lineage (original run log/source commit and the \`unet_v1\` version if actually used); then assess **independent-year real satellite test availability**. Avoid simply adding more synthetic OVATION target periods as a substitute for real observational generalization. If usable real data from another year are available and time-matched OMNI/OVATION arrays can be constructed, run a prospectively selected heldout comparison on genuinely separate-year acquisitions before closing R2-8/R3-1.


### Next low-cost task: determine real cross-year data availability (R2-8 / R3-1)

The audited six-high/six-quiet acquisition comparison has finished, but does not
establish **real measured SSUSI reconstruction transfer to another calendar
year**. A new inventory script \`survey_crossyear_real_inputs.py\` checks
2004/2006/2007/2008/2009 F16 EDR availability and the presence/shape of
appropriately matched OMNI and OVATION monthly inputs without loading any
model or processing full satellite flux maps.

\`\`\`bash
git pull --ff-only origin main
python paper_modif/Second_revision/R2-4_R2-8_R3-1/survey_crossyear_real_inputs.py
\`\`\`

The output is \`paper_modif/Second_revision/R2-4_R2-8_R3-1/crossyear_real_input_survey.json\`.
If \`n_months_with_all_potential_inputs\` is zero, provide the summary anyway:
generation of date-matched OVATION/OMNI arrays and an advance-held-out
satellite catalog would be needed for independent-year evaluation.
This survey alone is NOT evidence of cross-year prediction skill.

Source/provenance follow-up: look for the *original* training run script,
log, and the missing \`models/unet_v1.py\` (possibly in an older local checkout
such as \`Aurora_DDPM_final\`). File hashes recorded previously are not proof
of the code that produced model weights; the original run metadata is needed.
Do not upload hundreds of megabytes of checkpoint bytes.


## Cross-year inventory crash fix and recovered local UNet sources (2026-11-23)

The original \`survey_crossyear_real_inputs.py\` aborted on the first unreadable
OVATION \`.npy\` monthly product, with \`EOFError: No data left in file\`,
so the \`crossyear_real_input_survey.json\` output did **not** exist. The
error may mean a zero-byte placeholder or an otherwise malformed/incomplete
NPY; its exact identity is not known until the inventory is rerun.

The corrected survey:
- Records each OMNI/OVATION file's **existence, byte size, read status,
  NumPy shape and exact error**.
- Continues across all requested months if a file is missing, empty,
  unreadable, truncated or has an unexpected grid/cadence.
- Writes \`crossyear_real_input_survey.json\` even when **no months are usable**;
  \`ssusi_month_input_issues\` flags months with real EDR files but broken
  or missing OMNI/OVATION inputs.
- Does **not** infer valid model data from a mere filename. A zero-byte
  OVATION file should not be deleted or regenerated automatically before its
  source workflow is identified.
- Has regression tests for zero-byte, malformed, truncated, and full-survey
  JSON-output behavior in \`test_crossyear_survey.py\`.

The author also located two old architecture sources:
\`/home/docker/code/Aurora_DDPM_final/models/unet_v1.py\` and
\`/home/docker/code/Aurora_DDPM/models/unet_v1.py\`.
This means the historical UNet source is not entirely missing locally,
although it was not tracked in the **current** \`Aurora_DDPM_total\` repo.
A lightweight \`inspect_historical_training_sources.py\` program records
both versions' hashes and source markers and checks historical \`train.py\`
files without loading checkpoint weights. These files may help trace training
lineage; their mere existence is **not** proof of which one trained the actual
published checkpoint.

Run after pulling the fix:

\`\`\`bash
git pull --ff-only origin main
python -m unittest discover -s paper_modif/Second_revision/R2-4_R2-8_R3-1 -p 'test_*.py'
python paper_modif/Second_revision/R2-4_R2-8_R3-1/survey_crossyear_real_inputs.py
python paper_modif/Second_revision/R2-4_R2-8_R3-1/inspect_historical_training_sources.py
\`\`\`

Return the small JSON files
\`crossyear_real_input_survey.json\` and
\`historical_training_sources_audit.json\`, or the summary sections of their
printed output. Do not upload checkpoint weights or raw EDR files.
**R2-8 and R3-1 remain open; no final English Response is released yet.**


## 2026-11-23: independent-year inventory and historical UNet-v1 audit received

Author-uploaded \`crossyear_real_input_survey.json\` and
\`historical_training_sources_audit.json\` have been examined; key results
are archived in \`crossyear_2006_and_unet_sources_verified.json\`.

**Cross-year actual inventory.** The survey checked 60 months in
2004/2006/2007/2008/2009. It counted 4,927 original F16/SSUSI EDRs
in 2006, and 11 full, valid, one-minute 80×96 OVATION monthly arrays:
**January, February and April–December 2006**. March 2006's purported
OVATION monthly NPY is an empty **zero-byte** file. These files make
2006 a sensible first independent-year real-observation cohort.

However **no 2006 OMNI NPYs were found in the specific expected folder**
\`omni_real_data/omni_5min/2006/\`. The overall candidate count remains
zero, but this does **not** establish that all 2006 OMNI data are absent:
the author may have raw CDFs at a different location. Do not regenerate
all 11 months of OVATION or claim cross-year skill without checking raw
OMNI coverage and source fields.

**Historical UNet lineage.** Two local \`unet_v1.py\` copies were found,
one 454-line version in \`Aurora_DDPM_final\` (SHA-256
\`dc376e090fbab2e3...\`), another 536-line version in \`Aurora_DDPM\`
(\`75cfcd5b79ab069a...\`). They are different source versions.
The published inference checkpoint's original training command, source
version and data lineage remain unverified. Historical
\`inspect_historical_training_sources.py\` now compares the actual
\`UNet\` class AST signatures, constructor arguments and methods
without executing code or loading weights. An AST difference is not
proof of which architecture produced checkpoint weights.

### Run this precise next-stage inventory, no inference or retraining

\`\`\`bash
cd /home/docker/code/Aurora_DDPM_total
git pull --ff-only origin main

python -m unittest discover \
  -s paper_modif/Second_revision/R2-4_R2-8_R3-1 \
  -p 'test_*.py'

# Find the potentially reusable 2006 5-minute solar-wind source CDF/NPY
python paper_modif/Second_revision/R2-4_R2-8_R3-1/find_crossyear_omni_sources.py

# Compare both historical UNet-v1 definitions using AST structural signatures
python paper_modif/Second_revision/R2-4_R2-8_R3-1/inspect_historical_training_sources.py
\`\`\`

Return \`crossyear_omni_sources.json\` and the regenerated
\`historical_training_sources_audit.json\`. They are small summary reports,
NOT raw satellite data or 410MB model weights.

If a 2006 CDF/NPY candidate is found, the next stage must verify that it
matches 2006 **UTC**, exact [Bx,By,Bz,V,P] conditioning schema, five-minute
cadence and reasonable gap handling before preparing a 2006 real-SSUSI
held-out cohort. January 2006 already has 418 EDRs and a full OVATION
monthly NPY, so it is a practical first target. Do not use 2005
OMNI as a stand-in for 2006.

**Open reviewer comments:** R2-8 and R3-1 remain open; no reviewer
Response is generated from this file inventory.


## 2026-11-23 update — 2006 five-minute CDF FOUND; strict source audit before test

New author uploads: `crossyear_omni_sources.json` and a fresh
`historical_training_sources_audit.json`. The summary is archived as
`crossyear_2006_omni_and_unet_audit_verified.json`.

**Availability.** Twelve 2006 HRO2 five-minute CDF files and twelve 2006 HRO
five-minute CDF files are present under
`/home/docker/data/ro-share/omni/omni_cdaweb/{hro2_5min,hro_5min}/2006/`.
Two additional **2020** CDF files were inadvertently included in the old
scanner because the text `2006` appears within `20200601`. The
`find_crossyear_omni_sources.py` candidate-year filter and regression
tests have now been corrected. There are **26 actual 2006 candidates** in
these known folders (24 five-minute monthly CDFs plus 2 hourly CDFs).
This proves **source file presence**, not real Epoch coverage or usable
conditioning values. No matching 2006 five-minute model-ready NPYs were
found yet.

**Historical architectures.** The old
`Aurora_DDPM_final/models/unet_v1.py` has a default
`solar_wind_dim=5` and a `num_fourier_freq` parameter; the separate
`Aurora_DDPM/models/unet_v1.py` defaults to
`solar_wind_dim=7` and has a structurally distinct cross-modal design.
They must not be silently interchanged. Neither hash nor source AST
proves which generated the current pretrained checkpoint. The current
inference checkpoint's known strict compatibility with
`models/unet.py` is likewise NOT its historical training log.

### Phase A — validate the January 2006 HRO2 CDF (no inference, no NPY writing)

```bash
cd /home/docker/code/Aurora_DDPM_total
git pull --ff-only origin main

python -m unittest discover \
  -s paper_modif/Second_revision/R2-4_R2-8_R3-1 \
  -p 'test_*.py'

python paper_modif/Second_revision/R2-4_R2-8_R3-1/audit_and_prepare_2006_omni.py \
  --year 2006 --month 1
```

The JSON `omni_200601_cdf_audit.json` MUST have
`ready_for_conversion=true` and
`calendar_time_grid.exact_regular_complete_5min_utc=true`.
The audit uses the prior tested OMNI CDF decoder and variable selection
`Epoch,BX_GSE,BY_GSM,BZ_GSM,flow_speed,Pressure` (allowing documented
alternative speed source), same physical range checks, and a strict
fraction/gap guard (default ≤5% invalid values per channel, maximum
60 consecutive invalid minutes). It checks all source timestamps against
the **exact** 2006 January 5-minute calendar. The prior 2005
generator's example-event audit is NOT used for 2006.

### Phase B — only if Phase A passed

```bash
# Create VERIFIED January 2006 solar-wind input, do not overwrite existing files.
python paper_modif/Second_revision/R2-4_R2-8_R3-1/audit_and_prepare_2006_omni.py \
  --year 2006 --month 1 --convert

# Select separate 2006 acquisition dates without inspecting measured flux.
python paper_modif/Second_revision/R2-1_R2-2_R2-3_R3-3_R3-4_R3-5/prepare_multievent_ssusi.py \
  --edr-root /home/docker/data/private/AuroraData/real_aurora_data_ssusi/f16/2006 \
  --month 2006-01 --max-events 12 --min-gap-hours 24 \
  --output-root paper_modif/Second_revision/R2-4_R2-8_R3-1/real_2006_01_inputs \
  --dry-run

# If the preview selects >=6 nonoverlapping times, prepare real native SSUSI masks.
python paper_modif/Second_revision/R2-1_R2-2_R2-3_R3-3_R3-4_R3-5/prepare_multievent_ssusi.py \
  --edr-root /home/docker/data/private/AuroraData/real_aurora_data_ssusi/f16/2006 \
  --month 2006-01 --max-events 12 --min-gap-hours 24 \
  --output-root paper_modif/Second_revision/R2-4_R2-8_R3-1/real_2006_01_inputs
```

```bash
# Check exact UTC coverage and flux-blind masks for January 2006, NOT 2005.
python paper_modif/Second_revision/R2-1_R2-2_R2-3_R3-3_R3-4_R3-5/preflight_multievent_benchmark.py \
  --prepared-root paper_modif/Second_revision/R2-4_R2-8_R3-1/real_2006_01_inputs \
  --omni /home/docker/data/private/AuroraData/omni_real_data/omni_5min/2006/omni_20060101_5min.npy \
  --ovation /home/docker/data/private/AuroraData/generated_aurora_data/2006_omni_aurora/aurora_img_20060101.npy \
  --output paper_modif/Second_revision/R2-4_R2-8_R3-1/real_2006_01_preflight.json
```

No model inference until `real_2006_01_preflight.json` confirms
`ready_for_primary_inference=true`, with full 2006 OMNI+OVATION UTC
coverage, clean native SSUSI support and matched flux-blind masked targets.

### Phase C — only after the independent-year preflight passes

```bash
python paper_modif/Second_revision/R2-1_R2-2_R2-3_R3-3_R3-4_R3-5/multievent_heldout_benchmark.py \
  --prepared-root paper_modif/Second_revision/R2-4_R2-8_R3-1/real_2006_01_inputs \
  --output-root paper_modif/Second_revision/R2-4_R2-8_R3-1/real_2006_01_benchmark \
  --selection-mode random --windows-per-width 1 --min-events 6 \
  --omni /home/docker/data/private/AuroraData/omni_real_data/omni_5min/2006/omni_20060101_5min.npy \
  --ovation /home/docker/data/private/AuroraData/generated_aurora_data/2006_omni_aurora/aurora_img_20060101.npy
```

Keep the original 2005 checkpoints and baseline definitions unchanged.
Evaluate only genuinely measured, withheld SSUSI cells, with event-equal
summary and suitable baselines; do not mix 2006 and 2005 selected case
counts in a single event-pseudoreplicated CI.

**Reviewer status:** R2-8 and R3-1 remain OPEN pending genuine independent
2006 observed-target inference and an honest accounting of original
checkpoint training lineage. No English Response or unsupported manuscript
performance claims are delivered at this stage.

## January 2006 OMNI 825-minute outage: safe conditional path (2026-11-23)

The author returned `omni_200601_cdf_audit.json` and the SSUSI EDR
`--dry-run` catalog. Both were checked and summarized in
`author_2006_01_omni_quality_and_edr_preview_verified.json`.

**Observed source facts:** the HRO2 CDF contains all **8,928** 5-minute
timestamps, from January 1 00:00 through January 31 23:55 UTC, with
no clock gaps/duplicates. There are nevertheless **308 invalid Bx/By/Bz
records (3.45%)**, **402 invalid V/P records (4.50%)**, and each channel
has a longest invalid streak of **825 minutes (13 h 45 min)**. The
declared full-month missingness cap of 5% is satisfied but the
independent **60-minute maximum gap cap** is violated. Therefore the
original `audit_and_prepare_2006_omni.py --convert` correctly refused
to write the NPY. **Do not pass --max-invalid-gap-minutes 825 or
unconditionally fill/bfill these observations.** The conversion failed
by design, not due to a code crash.

The author also identified 418 January EDR candidates and selected
12 flux-blind, temporally spaced F16 acquisitions (January 2–30). This
establishes ONLY timestamp availability; it does not tell whether the
selected acquisition intervals overlap the long OMNI outages or have
sufficient native SSUSI support.

### Stage 1: require original five-field solar-wind completeness for EVERY selected EDR

```bash
cd /home/docker/code/Aurora_DDPM_total
git pull --ff-only origin main

python -m unittest discover \
  -s paper_modif/Second_revision/R2-4_R2-8_R3-1 -p 'test_*.py'

# Audit original raw CDF values inside the 12 preregistered SSUSI intervals.
# No NPY write, no DDPM; outputs a JSON even if one EDR is affected.
python paper_modif/Second_revision/R2-4_R2-8_R3-1/audit_2006_omni_event_coverage.py
```

Output `omni_200601_event_coverage_audit.json` includes each orbit's
`n_invalid_by_field`, `first_invalid_utc`,
`raw_fully_valid_acquisition`, and the total number of valid orbits.
Report the actual coverage rather than choosing a looser threshold
after inspecting predicted model errors.

If **all_preselected_acquisitions_raw_valid is true**, it is
scientifically acceptable to run a source-qualified analysis without
replacing the rest of January's missing OMNI measurements by fake
values. For that constrained use, the following second program is
available. It does **not** relax or overwrite the original whole-month
conversion rule.

### Stage 2: only if all 12 EDRs pass Stage 1

```bash
# No-write dry audit: fill only short, bracketed <=60min missing runs;
# preserve longer outages as NaN. Requires ALL selected EDRs raw valid.
python paper_modif/Second_revision/R2-4_R2-8_R3-1/prepare_2006_gap_preserving_omni.py

# Only when ready_for_event_qualified_write=true:
python paper_modif/Second_revision/R2-4_R2-8_R3-1/prepare_2006_gap_preserving_omni.py --write

# Prepare the same 12 time-selected real SSUSI swaths, after quality audit.
python paper_modif/Second_revision/R2-1_R2-2_R2-3_R3-3_R3-4_R3-5/prepare_multievent_ssusi.py \
  --edr-root /home/docker/data/private/AuroraData/real_aurora_data_ssusi/f16/2006 \
  --month 2006-01 --max-events 12 --min-gap-hours 24 \
  --output-root paper_modif/Second_revision/R2-4_R2-8_R3-1/real_2006_01_inputs
```

The resulting *qualified* array is deliberately named
`omni_20060101_5min_gap_preserving.npy` in the 2006 OMNI folder.
The unresolved long gaps remain explicitly **NaN** and are forbidden
as model inputs. The independent CDF audit proves that the selected
EDR intervals do not intersect those gaps.

### Stage 3: require BOTH OMNI/OVATION source + native-SSUSI holdout preflight

```bash
python paper_modif/Second_revision/R2-1_R2-2_R2-3_R3-3_R3-4_R3-5/preflight_multievent_benchmark.py \
  --prepared-root paper_modif/Second_revision/R2-4_R2-8_R3-1/real_2006_01_inputs \
  --omni /home/docker/data/private/AuroraData/omni_real_data/omni_5min/2006/omni_20060101_5min_gap_preserving.npy \
  --ovation /home/docker/data/private/AuroraData/generated_aurora_data/2006_omni_aurora/aurora_img_20060101.npy \
  --output paper_modif/Second_revision/R2-4_R2-8_R3-1/real_2006_01_preflight.json
```

This existing preflight is sufficient ONLY in combination with the
source-CDF `all_preselected_acquisitions_raw_valid` gate above;
its per-acquisition finite means alone would otherwise allow partially
missing OMNI to pass. **No DDPM inference before both gates pass.**

If Stage 1 identifies even one invalid EDR, do **not** run Stage 2;
the cohort must be selected again using prespecified time and valid
source-OMNI availability, without consulting withheld SSUSI flux or
model reconstruction error. Alternatively use a different month with
sufficient CDF quality and fully valid 1-min OVATION reference maps.

After clean input preflight, the 2006 independent-year comparison can
use the existing `multievent_heldout_benchmark.py` with explicit
`--omni ...gap_preserving.npy` and 2006 `--ovation`, keeping
the existing 2005 checkpoints, masks and method baselines constant.

**R2-8 and R3-1 remain OPEN**: a catalog with valid timestamps or
preserved missing regions is not yet a measured real-observation
cross-year performance result. The original checkpoint training
lineage is still not verified. No reviewer Response is issued.

## 2026-11-23 — 8/12 pass raw OMNI; 4 replacements REQUIRED before 2006 inference

The author ran the original 12-orbit CDF quality gate and returned
`omni_200601_event_coverage_audit.json`.

**Result: eight valid / four invalid**. The month contains 8,522 of
8,928 five-minute ticks with all five fields raw valid; 406 ticks have
at least one invalid field. The four invalid preselected EDRs are:

| Original F16 orbit | Bad 5-min samples inside EDR | Affected fields |
| --- | ---: | --- |
| REV11430, Jan 4 | 3 | V, P |
| REV11576, Jan 15 | 21 | Bx, By, Bz, V, P |
| REV11612, Jan 17 | 1 | V, P |
| REV11795, Jan 30 | 1 | V, P |

The **825-minute uninterrupted source outage** affects the Jan 15
candidate especially. The old `real_2006_01_inputs` catalog is now
identified as **not suitable for inference**. The gap-preserving OMNI
generator correctly refused both audit and `--write`; no new model
input file was produced and no checkpoint inference should be run.

### New selection protocol — no target flux or predictions involved

New program `reselect_2006_omni_qualified_ssusi.py` reads the
time-only January 2006 catalog and all 418 January EDR filenames
and checks each candidate's acquisition UTC interval against **raw**
2006 HRO2 source CDF [Bx, By, Bz, V, P]. It retains the eight
original preselected cases that pass every raw sample, replacing the
four failed cases with the closest-in-time **source-qualified** orbit
that satisfies the same **>=24 h gap BETWEEN EDR intervals**.
A deterministic stable timestamp/path tie-break resolves equal-distance
cases. Selection is **before model inference** and does not read
SSUSI flux, grid support, OVATION brightness, or model predictions.

The old selection is NOT overwritten. The new output directory is
`real_2006_01_omni_qualified_inputs`. The program saves all
exclusions, candidate counts, retained/replaced case identities,
distance from original time slots, source CDF SHA-256, and protocol
limitations in its new `catalog_preview.json` and `manifest.json`.

A source-quality screen is **not random sampling** of all solar-wind
conditions and may change the cohort's activity distribution. This
limitation must appear in the paper; the retained eight are not
representative proof of 2006 or an independent storm set.

### Phase A: check what the new selection actually yields — no model

```bash
cd /home/docker/code/Aurora_DDPM_total
git pull --ff-only origin main

python -m unittest discover \
  -s paper_modif/Second_revision/R2-4_R2-8_R3-1 -p 'test_*.py'

python paper_modif/Second_revision/R2-4_R2-8_R3-1/reselect_2006_omni_qualified_ssusi.py --dry-run
```

Check `ready_for_preparation`, `n_original_valid_preserved`,
`n_output_cases`, and `replacements` in:

`paper_modif/Second_revision/R2-4_R2-8_R3-1/real_2006_01_omni_qualified_inputs/catalog_preview.json`

If `ready_for_preparation=false` or fewer than twelve new qualified
EDRs are available, report the preview. DO NOT loosen the 24h separation,
allow invalid raw solar wind, or claim all 12 EDRs survived the filter.

### Phase B: only after complete preview + unit tests pass

```bash
# Generates new 80 x 96 native SSUSI support grids only after selection
python paper_modif/Second_revision/R2-4_R2-8_R3-1/reselect_2006_omni_qualified_ssusi.py --prepare

# Audits SAME NEW 12-case manifest, not original failed catalog
python paper_modif/Second_revision/R2-4_R2-8_R3-1/audit_2006_omni_event_coverage.py \
  --catalog paper_modif/Second_revision/R2-4_R2-8_R3-1/real_2006_01_omni_qualified_inputs/manifest.json

# Audit eligible, gap-preserving OMNI without writing files
python paper_modif/Second_revision/R2-4_R2-8_R3-1/prepare_2006_gap_preserving_omni.py \
  --catalog paper_modif/Second_revision/R2-4_R2-8_R3-1/real_2006_01_omni_qualified_inputs/manifest.json

# Write only when ready_for_event_qualified_write=true and no previous file exists
python paper_modif/Second_revision/R2-4_R2-8_R3-1/prepare_2006_gap_preserving_omni.py \
  --catalog paper_modif/Second_revision/R2-4_R2-8_R3-1/real_2006_01_omni_qualified_inputs/manifest.json \
  --write
```

The output `omni_20060101_5min_gap_preserving.npy` leaves ALL long
source gaps as NaN (short bracketed gaps ≤60 min may be filled). The
selected acquisition intervals remain raw-source complete; they do not
rely on any interpolation. The original full-month `--convert`
refusal remains intact.

### Phase C: model-free observed-pixel/OMNI/OVATION preflight

```bash
python paper_modif/Second_revision/R2-1_R2-2_R2-3_R3-3_R3-4_R3-5/preflight_multievent_benchmark.py \
  --prepared-root paper_modif/Second_revision/R2-4_R2-8_R3-1/real_2006_01_omni_qualified_inputs \
  --omni /home/docker/data/private/AuroraData/omni_real_data/omni_5min/2006/omni_20060101_5min_gap_preserving.npy \
  --ovation /home/docker/data/private/AuroraData/generated_aurora_data/2006_omni_aurora/aurora_img_20060101.npy \
  --output paper_modif/Second_revision/R2-4_R2-8_R3-1/real_2006_01_preflight.json
```

Only if both raw-source audit and this preflight pass (not merely
finite `nanmean`) should the existing model benchmark run, with
`--prepared-root real_2006_01_omni_qualified_inputs`, explicit 2006
`--omni` and `--ovation`, and a separate
`real_2006_01_benchmark` output folder.

**Current R2-8 and R3-1 status: OPEN.** No independent-year model
result exists yet, and precise checkpoint training lineage remains
unverified. No final reviewer Responses are issued.
