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

## 2006 duplicate EDR case identities — fixed at source inventory (2026-11-23)

The author ran the new `reselect_2006_omni_qualified_ssusi.py
--dry-run` and the file inventory raised:
`ValueError: Duplicate EDR case identity in source directory:
F16_2006012T222703_REV11544`. This occurs BEFORE any raw OMNI
reselection, native flux read or model inference. There may be
multiple processing versions of the same orbit/start time or even
disagreeing acquisition end times. This is an **inventory ambiguity**;
do not interpret the duplicate filenames as independent acquisitions.

The scanner now groups source files by (start-UTC, orbit) *as encoded
in the deterministic case ID*. It then follows an explicit source
version rule:

1. For a case in the original 12-EDR flux-blind catalog,
   **preserve the exact original `source_file` path**.
   Raise an error if the originally chosen file is missing, even
   when a different processed version is available.
2. For a previously unused case with the **same start/end times**
   across multiple files, prefer the highest explicit filename
   processing revision (e.g., `vA8.2.0r001` over
   `vA8.2.0r000`). Equal versions use lexical path order.
   This rule NEVER evaluates the flux values or predictions.
3. If a new case has **conflicting end times** among files sharing
   its orbit/start identity, exclude that ambiguous acquisition
   from the candidate population and record the conflict.
4. All duplicates, source versions, chosen paths, excluded candidates
   and reasons are recorded under
   `source_scan.duplicate_identity_resolution` in the output
   `catalog_preview.json`. The saved preview, if used for
   `--prepare`, is checked against BOTH orbit identity and
   source file path, plus OMNI source CDF SHA.

The results previously uploaded (8 clean, 4 invalid old EDRs) and
the source-quality gate are **unchanged** by the filename fix.
The `--dry-run` must still verify the ability to select a
12-case cohort with >=24-hour separation.

Re-run after pulling:

```bash
cd /home/docker/code/Aurora_DDPM_total
git pull --ff-only origin main
python -m unittest discover \
  -s paper_modif/Second_revision/R2-4_R2-8_R3-1 -p 'test_*.py'
python paper_modif/Second_revision/R2-4_R2-8_R3-1/reselect_2006_omni_qualified_ssusi.py --dry-run
```

Please inspect `n_duplicate_identity_groups`,
`n_conflicting_end_time_groups_excluded`,
`retained`, `replaced`, `n_selected` and
`ready_for_preparation`. If the last is false, stop and report
the preview; do not relax source quality or timing thresholds.

No model inference has been performed by this fix; R2-8 and R3-1
remain open.


## 2026-11-23: qualified 2006 cohort READY for source write and mask preflight

The author provided complete dry-run, native swath preparation, raw HRO2
source-CDF quality and gap-preserving-OMNI **no-write** logs. An exact
fact summary is committed in
`author_2006_qualified_cohort_preflight_stage_verified.json`.

The original catalog's eight fully clean EDRs were preserved, four
source-unqualified EDRs replaced, and the resulting 12-case independent
**January 2006** cohort is prepared. Raw original five-field OMNI
measurements are valid for **all 12 acquisition windows**. The source
inventory reports 412 unique usable EDR cases, 3 ambiguous duplicate
identity groups excluded and 329 OMNI-qualified EDR candidates. The
dedicated gap-preserving monthly input will retain **281 NaN timepoints**
across the month; no source-invalid OMNI values appear within the selected
EDRs. **This is not a fully complete monthly dataset.**

**Critical native-coverage gate:** 11 of 12 prepared orbit grids have
3,134–4,341 observed native-supported cells, but replacement
`F16_2006017T143452_REV11610` has **only 849/7680 native-supported
cells (11.05%)**. It might lack one or more flux-blind observed target
windows. The *preflight* enforces the unchanged 1/2/4 h MLT sectors,
60–80 MLAT, 50% target coverage, and >=20 measured target pixels.
Do **not** assume that the high-level EDR source audit certifies
eligible holdout masks or loosen those thresholds simply to keep 12 EDRs.

### Authorized next two commands — no DDPM inference

```bash
cd /home/docker/code/Aurora_DDPM_total
git pull --ff-only origin main

# Now write the validated source-qualified array using the NEW manifest,
# never the original 12-case file in real_2006_01_inputs/.
python paper_modif/Second_revision/R2-4_R2-8_R3-1/prepare_2006_gap_preserving_omni.py \
  --catalog paper_modif/Second_revision/R2-4_R2-8_R3-1/real_2006_01_omni_qualified_inputs/manifest.json \
  --write

# Native SSUSI geometry / flux-blind targets / exact UTC source preflight.
python paper_modif/Second_revision/R2-1_R2-2_R2-3_R3-3_R3-4_R3-5/preflight_multievent_benchmark.py \
  --prepared-root paper_modif/Second_revision/R2-4_R2-8_R3-1/real_2006_01_omni_qualified_inputs \
  --omni /home/docker/data/private/AuroraData/omni_real_data/omni_5min/2006/omni_20060101_5min_gap_preserving.npy \
  --ovation /home/docker/data/private/AuroraData/generated_aurora_data/2006_omni_aurora/aurora_img_20060101.npy \
  --output paper_modif/Second_revision/R2-4_R2-8_R3-1/real_2006_01_preflight.json
```

The first command must report `saved=true`, `error=null` and
a real `output_npy_sha256`. It refuses to overwrite an existing NPY.
The second command must report
`ready_for_primary_inference=true`, `n_catalog_cases=12`,
`n_fully_evaluable_acquisitions=12`, `errors=[]`.
**If it fails (especially on REV11610), stop and inspect each missing
width in the JSON instead of launching DDPM.** A second source-blind
candidate-selection stage could prioritize native *coverage only*, not
heldout flux or observed prediction error, but that would require
explicitly recording selection bias and rerunning source and preflight
checks on a new distinct cohort.

The original 2006 HRO2 long 825-minute source outage was never filled.
The ordinary full-month `audit_and_prepare_2006_omni.py --convert`
still correctly refuses conversion.

R2-8 and R3-1 remain OPEN. No result yet establishes 2006 real
withheld-target reconstruction accuracy or exact checkpoint lineage.


## 2026-11-23: all 12 January 2006 real SSUSI masked-target preflights PASSED

The author completed the `--write` stage for the January 2006
*gap-preserving* HRO2 solar-wind NPY with `saved=true`, no error,
SHA-256 `6a7996b5a48cd74677c8499f6e01a7f5fce2b57104e5aee97f9a870058265335`,
and 281 unresolved five-minute samples containing at least one NaN
across the FULL MONTH. The selected EDR intervals remain independently
verified to contain **only raw-valid OMNI** data.

Then the author submitted `real_2006_01_preflight.json`, whose every
acquisition and withheld mask was checked against the original
science/coverage criteria. Main summary:

- `n_catalog_cases=12`
- `n_fully_evaluable_acquisitions=12`
- `ready_for_primary_inference=true`, `errors=[]`
- All 12 acquisitions have a finite, date-matched 2006 OMNI interval
  mean and full date-matched 1-minute 2006 OVATION image coverage.
- 1/2/4-hour MLT random sectors: **12 each = 36 heldout sectors**.
  Their corresponding real observed SSUSI hidden-cell counts total
  **1,587 / 2,848 / 5,890**, respectively.
- The sparsely observed `F16_2006017T143452_REV11610` (849/7680
  native observed cells) nevertheless meets the SAME flux-blind,
  60–80 MLAT, 50%-coverage and >=20-pixel criteria for all three
  prescribed sector widths: 130, 249 and 422 native observed hidden
  cells for 1/2/4 h. Do NOT discard it on global support fraction alone.

An exact report summarizing the author's output, without any invented
model performance numbers, is committed as
`author_2006_01_real_preflight_verified.json`.

### INFERENCE UNBLOCKED — use exact 2006 sources, no model changes

```bash
cd /home/docker/code/Aurora_DDPM_total
git pull --ff-only origin main

# Low-cost diagnostic, no model inference: 2006 input means vs stored
# 1996 training normalizer. This does not change cohort or model.
python paper_modif/Second_revision/R2-4_R2-8_R3-1/audit_2006_solar_normalization.py

# Run REAL January 2006 independent-calendar-year satellite holdout
# reconstruction on the *qualified* 12-acquisition manifest.
python paper_modif/Second_revision/R2-1_R2-2_R2-3_R3-3_R3-4_R3-5/multievent_heldout_benchmark.py \
  --prepared-root paper_modif/Second_revision/R2-4_R2-8_R3-1/real_2006_01_omni_qualified_inputs \
  --output-root paper_modif/Second_revision/R2-4_R2-8_R3-1/real_2006_01_benchmark \
  --selection-mode random \
  --windows-per-width 1 \
  --min-events 12 \
  --seed 2026 \
  --omni /home/docker/data/private/AuroraData/omni_real_data/omni_5min/2006/omni_20060101_5min_gap_preserving.npy \
  --ovation /home/docker/data/private/AuroraData/generated_aurora_data/2006_omni_aurora/aurora_img_20060101.npy

# Acquisition-equal comparison, paired across the same real SSUSI pixels.
# Do this only after the benchmark created metrics_per_holdout.csv.
python paper_modif/Second_revision/R2-1_R2-2_R2-3_R3-3_R3-4_R3-5/event_level_statistics.py \
  --metrics paper_modif/Second_revision/R2-4_R2-8_R3-1/real_2006_01_benchmark/metrics_per_holdout.csv \
  --output-root paper_modif/Second_revision/R2-4_R2-8_R3-1/real_2006_01_benchmark
```

Expected completed benchmark *if all inference executions succeed*:
36 heldout sectors, **180 method-scoring records** (conditional DDPM,
unconditional DDPM, direct OVATION Prime, observed-fused OVATION and
spatial interpolation). All five methods must be scored only on the
SAME genuinely observed, deliberately hidden SSUSI native pixels.
**These expected numbers are not measured 2006 scores.**

The `audit_2006_solar_normalization.py` diagnostic records each
original OMNI interval mean compared to the actual stored model
normalization [min, min+range], including per-parameter input
saturation at the [0,1] clipping bounds. This is relevant to R2-8:
1996 training may not cover all 2006 solar-wind states.
Never omit or select acquisitions according to clipping, never
renormalize to make 2006 test scores better, and do not substitute
it for original checkpoint training provenance.

Submit four small result files:
- `real_2006_01_solar_conditioning_range_audit.json`
- `real_2006_01_benchmark/benchmark_summary.json`
- `real_2006_01_benchmark/metrics_per_holdout.csv`
- `real_2006_01_benchmark/event_level_summary.json`

Keep the original trained conditional/unconditional checkpoints,
the identical 2005 holdout protocol and all model baselines unchanged.
The 12-acquisition cohort is selected by **source OMNI completeness
and UTC spacing**, not at random from all 2006 conditions. 24-hour
separation does not prove 12 independent storms or solar-cycle
generalization.

**R2-8/R3-1 remain OPEN** until the 2006 observed-target results are
computed, compared transparently with 2005 and the synthetic-only
2001/2005/2009 tests, and the unresolved exact trained-checkpoint
provenance is adequately documented. No English reviewer Responses yet.


## 2026-11-23 — January 2006 12-EDR independent-year real benchmark DONE; fix event-level NaN parsing

The author uploaded the *full 2006 benchmark console output* as
`粘贴的文本 (1)(20261123-092331).txt`. All **12 original-checkpoint,
independent-year F16/SSUSI acquisitions** were evaluated at 1, 2 and
4 MLT-hour flux-blind random masked sectors: **36 true observed
hidden SSUSI sectors, 180 method-scoring rows including observed-fused
OVATION**, no apparent inference failure.

**Important provenance:** these are the ORIGINAL conditional and
unconditional model checkpoints used for the January 2005 benchmark,
without retraining, but the full historical 1996 training source/run
metadata remains unverified. The January 2006 target pixels are actual
DMSP F16/SSUSI measurements deliberately withheld during inference;
this does establish a *separate real-observation calendar year* test,
not a solar-cycle-wide representative test.

Approximate *event-equal* RMSE from console values rounded to four
decimals (NOT the precise original CSV; see
`author_2006_real_console_benchmark_preliminary.json`):
conditional **0.662**, unconditional **0.642**, direct OVATION
**0.784**, interpolation **0.635**. Mean paired conditional-minus-
baseline RMSE is approximately **+0.0193 vs unconditional
(conditional wins 3/12 EDRs)**, **−0.1223 vs direct OVATION
(12/12 wins)**, **+0.0262 vs interpolation (4/12 wins)**.
Do not make claims about statistical significance until the exact
event-clustered summary has run.

**High-leverage real case:** the low-coverage yet valid
`F16_2006017T143452_REV11610` has three-width mean conditional
RMSE ~**3.619**, accounting for ~**45.6%** of the total
12-orbit *sum of EDR mean RMSEs*. It PASSED all target-flux-blind
native heldout checks. It must stay in the primary analysis; report
acquisition-level heterogeneity and a clearly labeled leave-one-out
sensitivity analysis rather than silently excluding difficult events.

**Statistics crash (NO model rerun required):** the original
`event_level_statistics.py` attempted `float("")` on blank
fields in `metrics_per_holdout.csv`. The benchmark correctly
reports undefined Pearson r, R², peak ratios and gradient ratios
for constant/zero SSUSI targets as NaN; pandas CSV serializes
these as blank. The event statistics script now:
- treats such blank/NaN diagnostics as **undefined**, not zero;
- **rejects** missing/malformed essential RMSE, MAE and peak magnitudes;
- verifies that direct versus observed-fused OVATION have the same
  *finite-or-undefined* status and identical finite metrics;
- reports per-acquisition valid metric counts `n_valid_*`;
- computes conditional-minus-baseline paired metric differences only
  from sectors where BOTH metrics are defined, reporting
  `n_valid_paired_*`.

Regression tests are committed in
`test_event_level_statistics_missing_values.py`.

### Run ONLY statistics and normalization diagnostic — do not rerun 2006 DDPM

```bash
cd /home/docker/code/Aurora_DDPM_total
git pull --ff-only origin main

# Validate the NaN/blank scientific metric policy and all fourth-batch code
python -m unittest discover \
  -s paper_modif/Second_revision/R2-4_R2-8_R3-1 -p 'test_*.py'

# Read the PREVIOUSLY GENERATED original 2006 real-SSUSI CSV;
# writes event_level_summary.json and case/paired CSVs, no inference.
python paper_modif/Second_revision/R2-1_R2-2_R2-3_R3-3_R3-4_R3-5/event_level_statistics.py \
  --metrics paper_modif/Second_revision/R2-4_R2-8_R3-1/real_2006_01_benchmark/metrics_per_holdout.csv \
  --output-root paper_modif/Second_revision/R2-4_R2-8_R3-1/real_2006_01_benchmark

# If not done previously, audit solar conditioning against stored
# 1996 normalization bounds WITHOUT excluding any 2006 EDRs.
python paper_modif/Second_revision/R2-4_R2-8_R3-1/audit_2006_solar_normalization.py
```

Send the exact generated inputs for next scientific work:
`real_2006_01_benchmark/metrics_per_holdout.csv`,
`real_2006_01_benchmark/event_level_summary.json`,
`real_2006_01_benchmark/benchmark_summary.json`,
and `real_2006_01_solar_conditioning_range_audit.json`.
If available, also send `metrics_by_acquisition.csv`.
This will allow exact orbit-weighted 2005-versus-2006 interpretation
and robust peak/gradient attenuation measurements, followed by a
complete cumulative revised Word manuscript.

The author originally requested no new English responses to R2-4,
R2-8 or R3-1 until each is genuinely resolved, including checkpoint
lineage and full manuscript source consistency. No new reviewer
Responses are released here.


## 2026-11-23 — VERIFIED complete January 2006 observed-target results (replaces preliminary console rounding)

The author submitted the **original full-precision**
`real_2006_01_benchmark/metrics_per_holdout.csv`,
`benchmark_summary.json`, `event_level_summary.json` (after fixing
blank/NaN diagnostics), and `real_2006_01_solar_conditioning_range_audit.json`.
All 180 CSV rows were independently recomputed and matched to the author's
event summary (12 F16 EDR acquisitions, 36 target-flux-blind sectors, 5
method rows per sector). The exact numeric audit is committed as
`verified_2006_independent_year_real_holdout_results.json`.

All 5 model/display methods used **identical hidden measured SSUSI target
pixels** and 1/2/4-h MLT windows, with no unequal truth/coverage/center
entries or nonzero flux-based selection scores. Direct OVATION and fused
OVATION are **exactly identical** on measured hidden pixels, so only
**four distinct predictions** are meaningful in the performance table.

**Twelve-acquisition equally weighted mean errors** (RMSE/MAE in
erg cm^-2 s^-1; lower is better):

| Method | RMSE | MAE | Mean absolute peak error | Ratio of aggregate predicted/measured peaks | Ratio of aggregate predicted/measured gradient |
| --- | ---: | ---: | ---: | ---: | ---: |
| Conditional DDPM | 0.661642 | 0.286331 | 4.749005 | 0.108367 | 0.183580 |
| Unconditional DDPM | 0.642301 | 0.255975 | 4.765793 | 0.102457 | 0.275689 |
| Direct OVATION Prime | 0.783899 | 0.408588 | 4.602032 | 0.161523 | 0.183573 |
| Spatial interpolation | 0.635479 | 0.239251 | 3.297304 | 0.394146 | 0.443703 |

**Paired event-level conditional minus baseline RMSE**:

| Baseline | Mean paired difference | Event-bootstrap 95% CI | Acquisitions where conditional improves |
| --- | ---: | --- | ---: |
| Unconditional DDPM | +0.019341 | [+0.008547, +0.030063] | 3/12 |
| Direct OVATION | -0.122257 | [-0.163156, -0.082216] | 12/12 |
| Spatial interpolation | +0.026163 | [-0.022373, +0.074900] | 4/12 |

Thus the model improves against the **unmodified OVATION generator**
(conditional RMSE 15.6% lower), but **does not establish superiority
over unconditional DDPM or interpolation** on a genuinely measured,
independently dated satellite cohort. No post hoc model or checkpoint
selection was conducted on 2006 cases. The reported intervals resample
the 12 acquisition clusters and cannot establish independence of
actual geomagnetic storms.

**Physical morphology:** conditional reconstructed peak magnitudes
reach only 0.108 of the measured aggregate mean peaks; the
reconstructed-to-observed gradient magnitude ratio is 0.184, versus
0.394 and 0.444 for interpolation. The mean of event means of
individual sector peak ratios is a **different estimator**
(conditional 0.382), dominated by weak denominators and missingness.
Report the aggregate physical ratio alongside any sector-ratio means.
Two completely zero-truth sectors in the January 10 EDR have
scientifically undefined correlation and relative peak/gradient
metrics, correctly left blank/NaN, not imputed as zero. Interpolation
Pearson r is defined for 11 EDR means; other methods 12.

**High-leverage measured event:** preflight-valid REV11610
(17 January 2006, 849 native-supported map cells) has conditional
mean RMSE 3.61924 and contributes 45.6% of the sum of 12
acquisition-mean conditional RMSEs. It **must remain in the primary
12-acquisition analysis**. The labeled leave-one-out sensitivity
reduces absolute conditional mean RMSE to 0.39277, but conditional
minus unconditional mean RMSE stays **+0.01939**, while conditional
minus interpolation stays **+0.00958**.

**1996-training normalizer external-year check:** zero of twelve
2006 EDR mean vectors clipped against the saved [min, min+range]
range for any Bx/By/Bz/V/P input; all individual raw-valid 5-min values
inside the selected EDRs also remain within these stored bounds.
This rules out simple inference-bound truncation for **these 12
cases only**, not broader input-regime shift nor the unresolved
trained-checkpoint provenance.

**2005 vs 2006 caveat:** the original January 2005 random 12-EDR
observed-target sample has conditional RMSE 1.503129, versus 0.661642
in the different January 2006 sample. Their measured mean hidden
fluxes are **0.9161 versus 0.2467 erg cm^-2 s^-1**. Do NOT present
lower absolute 2006 RMSE as better cross-year accuracy or an effect
of the year: the target intensities and acquisition populations differ.

**Cumulative manuscript:** the full 34-page working Word manuscript
has now been updated with independent-year methodology in Section 2.5,
a new Section 4.7, new **Table 12**, updated Abstract/Key Points/Plain
Language Summary/Discussion, and corrections to outdated statements
claiming all real SSUSI cohorts were drawn from January 2005. It
retains earlier Kp>5 results in Table 11 and the prior tables/figures
and 20 original mathematical formula objects. The manuscript was
rendered and visually checked as a complete document; the Word
artifact is delivered separately to the author. No final English
reviewer Responses are released.

**Open R2-8/R3-1:** measured test data now cover a second calendar
year (2006) independent of the nominal 1996 synthetic training source
and previously studied 2005 SSUSI. This is substantive but limited
external-year evidence. The exact original training log, final
model source version and saved-checkpoint lineage are still
not proven, and available samples do not establish representative
storm independence or solar-cycle-wide performance.


## 2026-11-23 — CHECKPOINT LINEAGE: evaluated conditional checkpoint differs from two historical best snapshots

Author-provided `checkpoint_provenance_audit.json` and the subsequent
`sha256sum`/historical `find` logs were checked and archived in
`author_checkpoint_lineage_followup_20261123.json`. They establish:

- **Evaluated** conditional (both Jan 2005 / Jan 2006 real-SSUSI tests):
  `Aurora_DDPM_total/ckpt/cond/aurora_diff_best.pth`,
  SHA-256 `875b88edf91ca2a56d7a3041d55f956ca1ff3243bd137a44ef3e534fbce7ed12`,
  **410,372,478 bytes**.
- Evaluated unconditional:
  `Aurora_DDPM_total/ckpt/uncond/aurora_diff_best.pth`,
  SHA-256 `8061c33f406e47ff2e793e96473c9d5eec1649bb70e02a1bc570bbcaace4bac3`,
  **438,683,928 bytes**.
- Historical `Aurora_DDPM_final/ckpt/cond/ckptv4_unetv1/aurora_diff_best.pth`:
  SHA-256 `9da2949204ce94b187ca2e263e7a44f1602175448eb5e6bab037fefbef9e504a`.
- Historical `Aurora_DDPM_final/ckpt/cond/ckptv1_unetv1/aurora_diff_best.pth`:
  SHA-256 `cbbb41400701b5e4473c4fb4dc7f3e47c19250faafdaea58d225c31914f6b660`.

**These three conditional checkpoint files are not byte-identical.**
The repository's present `train.py` declares 1996 April–May
training, imports `models.unet_v1`, and points its best checkpoint to
`Aurora_DDPM_final/.../ckptv4_unetv1`; but the evaluated file is NOT
that byte-identical snapshot. Two distinct historical
`models/unet_v1.py` versions also exist (5 and 7 default wind
dimensions). It would be **scientifically incorrect** to claim that
the present `train.py` directly generated the evaluated weights
without original run logs/source snapshot/weight-level evidence.

The author's historical listing used `find ... | head -100`, so
later archive candidates were **not inspected at all**. This
incompleteness does not imply that any observed checkpoint has been
found to be an exact match.

### Last bounded provenance check (NO weight loading or retraining)

```bash
cd /home/docker/code/Aurora_DDPM_total
git pull --ff-only origin main

python -m unittest discover \
  -s paper_modif/Second_revision/R2-4_R2-8_R3-1 -p 'test_*.py'

python paper_modif/Second_revision/R2-4_R2-8_R3-1/audit_historical_checkpoint_matches.py
```

This reads **every** checkpoint pathname and size under the two known
historical `ckpt` directories. It computes streaming SHA-256
**only** for archived files with the same size as one of the evaluated
conditional or unconditional weights. This avoids reading GBs of
non-candidate checkpoints; it does not unpickle or import old training
files, and it does not mutate models.

Return only the small
`historical_checkpoint_match_audit.json` report; weight files stay
local. Inspect `byte_identical_archive_paths`,
`matching_size_archive_paths`, and `n_archive_checkpoint_files`.

- **Exact SHA match:** establishes binary file identity only; still
  check original model source revision, normalization and train/data
  provenance before attributing the checkpoint to a precise run.
- **No exact SHA match:** no serialized-byte identity within those
  archive directories; historical source version is **unverified**.
  It remains possible that numerically identical tensors were saved
  with different serialization/metadata; do not claim this is
  disproven. A bounded shape-only comparison can be considered if
  indispensable, but original log absence should be disclosed
  instead of indefinitely searching the filesystem.

These findings do **not** invalidate the empirical January 2005/2006
real-SSUSI results obtained with the explicitly identified evaluated
checkpoints. They limit claims about the *nominal 1996 training
run and model implementation lineage*. Do not retrain on heldout SSUSI
or change architecture before final reviewer-response preparation.

**R3-1 remains unclosed on exact lineage**. R2-8 external-year SSUSI
observed-target testing is complete within the documented year/activity
sampling limits. Revise final manuscript language according to the
actual provenance audit outcome, then proceed to the complete
reviewer-by-reviewer response (not before).


## 2026-11-23 — FINAL HISTORICAL ARCHIVE IDENTITY DISCOVERY (supersedes prior 2-file negative result)

The author completed the full, **untruncated** checkpoint inventory
with `audit_historical_checkpoint_matches.py` and uploaded the original
`historical_checkpoint_match_audit.json`. The scan counted **1,054
archive checkpoint files** in the two history trees, including **61
best-named snapshots**, and SHA-hashed two files matching the evaluated
weight-file sizes. **Both evaluated model files have exact full SHA-256
matches**:

| Evaluated real-SSUSI checkpoint | Original archived snapshot whose bytes match exactly | SHA-256 |
| --- | --- | --- |
| `Aurora_DDPM_total/ckpt/cond/aurora_diff_best.pth` | `Aurora_DDPM_final/ckpt/cond/ckptv2_unetv3/aurora_diff_best.pth` | `875b88edf91ca2a56d7a3041d55f956ca1ff3243bd137a44ef3e534fbce7ed12` |
| `Aurora_DDPM_total/ckpt/uncond/aurora_diff_best.pth` | `Aurora_DDPM/ckpt/diffusion_ckpt_simplenet/ckpt_v8/aurora_diff_best.pth` | `8061c33f406e47ff2e793e96473c9d5eec1649bb70e02a1bc570bbcaace4bac3` |

**This materially changes the previous provenance conclusion**:
no matching byte identity was found in the *two initially checked*
historical `ckptv4_unetv1` and `ckptv1_unetv1` snapshots, but a full
inventory DOES find the evaluated conditional snapshot under the
different `ckptv2_unetv3` directory. Similarly, the baseline
unconditional snapshot matches `diffusion_ckpt_simplenet/ckpt_v8`.
These are the exact archived model files used for the fixed-weight
January 2005 and January 2006 tests. Both previous model performance
tables remain valid for the tested weights; no NPU inference rerun is
required merely to establish file identity.

**Important scientific evidence boundary:** A byte-identical archived
checkpoint is **NOT proof of its complete historical training run,
original source import, or input dataset composition**. The present
`train.py` references `models.unet_v1` and best save directory
`ckptv4_unetv1`, whereas the evaluated conditional checkpoint matches
`ckptv2_unetv3`. The archive path's `unetv3` label is informative
provenance, but it does not authenticate the actual historical source
class. The originally described April–May 1996 OVATION + OMNI training
workflow and Figure 3 100-epoch loss trace cannot be **directly
attributed** to these specific archived bytes without contemporaneous
logs, archived training source and dataset fingerprints. File mtimes
are not reliable training start/finish timestamps.

Full findings are in
`verified_historical_checkpoint_identity_20261123.json`.

**Full cumulative manuscript updated again:** the 12-table manuscript
retains all validated 2005 and 2006 SSUSI comparisons, explicitly
identifies both evaluated archive snapshots and complete SHA-256
in Section 3.3, presents Section 3.2 as the *inference-side*
architecture, and qualifies the 1996 training scenario, Table 6
training settings and Figure 3 development loss curve rather than
falsely certifying their linkage to the evaluated checkpoints.
Limitations and Conclusions distinguish confirmed archive file
identity from still-unverified run/source/data provenance.

The user-requested next phase is a full consistency check followed by
an appropriately bounded reviewer response; DO NOT declare exact
training-lineage closure solely on directory names or SHA identity.


## 2026-11-23 — NEXT REVIEWER PRIORITY: R3-8 fixed-mask stochastic DDPM uncertainty

The full 2005/2006 heldout SSUSI evidence and both evaluated historical
weight-file SHA-256 identities have been independently verified. A
full reviewer completion audit (`R2_R3_2026_11_23_completion_gate.md`)
found that the **highest-priority still-unanswered measurable item**
is Reviewer #3's request for **multiple stochastic reconstructions
and uncertainty**, rather than another large cross-year/storm benchmark.

**Do not confuse two different sampler parameters:**
`model.sample(n_sample=5)` would change the reverse/forward jumping
schedule inside ONE RePaint-type sample, not perform five independently
seeded reconstructions. Keep `N_SAMPLE=1`,
`NUM_INFERENCE_STEPS=300`, the **original** conditional
`ckpt/cond/aurora_diff_best.pth` and original log/flux normalizer,
and run the same fixed SSUSI masks under five different *random
seeds*. This is an unchanged-model stochastic uncertainty analysis.

A new audited script
`audit_2006_stochastic_reconstruction.py` holds the January 2006
**12 EDR cases / 36 flux-blind observed-pixel sectors (1/2/4-h MLT)**
fixed. Each mask is regenerated from the SAME `selection_seed=2026`
and original native `obs_mask` geometry, then checked against
the original saved `metrics_per_holdout.csv` for the identical
sector center, number of observed hidden pixels, coverage and
three widths. All forecast seeds differ only in diffusion PRNG
state; masks, data, solar condition, weights and inference step
count remain unchanged.

**First command: model-free dry-run check of 36 exact original masks**

```bash
cd /home/docker/code/Aurora_DDPM_total
git pull --ff-only origin main
python -m unittest discover \
  -s paper_modif/Second_revision/R2-4_R2-8_R3-1 -p 'test_*.py'

python paper_modif/Second_revision/R2-4_R2-8_R3-1/audit_2006_stochastic_reconstruction.py \
  --dry-run
```

Expected preflight summary:
`ready_for_stochastic_inference=true`,
`n_original_fixed_masks=36`,
`n_repeated_draws_per_mask=5`,
`n_requested_conditional_model_runs=180`,
`all_mask_centers_and_support_counts_match_primary_csv=true`.
No GPU/NPU inference is performed and no weights are loaded during
this flag, although the original benchmark module is imported.
**If any check fails, stop and return the error**: do not
reselect targets based on flux, change the benchmark 2026 seed,
or relax native support thresholds.

**Second command: fixed-mask five-seed real-SSUSI inference, NO retrain**

```bash
python paper_modif/Second_revision/R2-4_R2-8_R3-1/audit_2006_stochastic_reconstruction.py
```

Writes, without touching original 2006 benchmark outputs:
`real_2006_01_stochastic_uncertainty/stochastic_protocol.json`,
`real_2006_01_stochastic_uncertainty/sector_results/*.json`,
and `real_2006_01_stochastic_uncertainty/stochastic_uncertainty_summary.json`.

If interrupted during the 180 model draws, rerun with **`--resume`**
to retain already completed, identity/seed/coverage-checked
sector JSON files. No plotting, image saving, or full raw
predictive-map archive is done by default; it reports all 36
per-mask measurements, the 12 event-equal mean metrics and
a descriptive pixelwise 10%–90% draw-quantile inclusion statistic.

**Interpretation:** these standard deviations quantify
conditional **sampling randomness** for exactly fixed observed
pixels, model, and solar forcing. They are NOT event-cluster
bootstrap confidence intervals, are NOT coverage-calibrated
posterior predictive intervals, and do not prove the DDPM
predictive distribution is accurate. Never report a
`best-of-five` reconstruction chosen by comparison with heldout
SSUSI truth. Keep the verified main Table 12 RMSE 0.661642
and all existing baselines unchanged.

**Send the terminal summaries and**
`real_2006_01_stochastic_uncertainty/stochastic_uncertainty_summary.json`;
the complete `sector_results` directory is only required if
the numeric summary appears inconsistent or a full appendix
is needed. All local outputs remain gitignored.

**After R3-8:** complete R2 Figure 4–7 direct IMF Bz/V/P
annotations and first-use instrument expansions; explain
R3 Table 2 vs Table 3 RMSE protocol discrepancy; correct
OVATION Bx/P driver explanation, citations and Data/Software
Availability; then produce revised clean + marked Word,
standalone figures and final comprehensive reviewer response.
No more 2006 data re-selection, model retraining or
2007–2009 input-generation is currently warranted.


## 2026-11-25 — REVIEWER R3-8 FIVE-SEED STOCHASTIC UNCERTAINTY: VERIFIED

Author completed `audit_2006_stochastic_reconstruction.py` after
all 36 January 2006 real F16/SSUSI native-supported flux-blind
heldout sectors were verified against original 2006 benchmark CSV.
The model, original train checkpoint SHA, conditioning, mask geometry,
widths, native observed truth and original reverse-sampling parameters
(`n_sample=1` internal jumps, 300 inference steps) were held fixed.
Five independent diffusion RNG streams per target region generated
**180 conditional reconstructed maps** across **12 orbital clusters**.
The first seed reproduced each original 2006 benchmark RMSE to a
max absolute difference of **2.22 × 10^-16**, verifying that the
mask/seed replay did not shift the primary experimental protocol.

The author's complete `stochastic_uncertainty_summary.json` was
independently parsed and *all four reported 12-acquisition equal
means were recomputed exactly from its per-acquisition entries*:

| Fixed-mask stochastic measure | 12-EDR equal-weight mean |
| --- | ---: |
| Mean per-draw hidden-target RMSE | 0.6622203492 |
| Across-five-draw RMSE standard deviation, mean over masks/EDRs | 0.0019772814 |
| Across-five-draw per-pixel flux standard deviation, mean over hidden pixels, masks and EDRs | 0.0196996626 |
| Fraction of actual heldout SSUSI flux values within five-draw empirical 10th–90th-percentile envelope | 0.1134590638 (11.35%) |

Flux/RMSE/SD quantities are in erg cm^-2 s^-1. The mean
within-mask RMSE spread is only **0.30%** of the mean RMSE,
whereas low observed-pixel inclusion demonstrates that this
empirical five-draw uncertainty envelope is much narrower
than the observed error about the available point estimates.
The high-error original REV11610 remains a valid and
included event: five-draw mean RMSE 3.622756; draw-RMSE SD
0.004371; observed-pixel envelope inclusion 0.023818.

**Statistical interpretation:** This measures intrinsic sampling
stochasticity conditional on fixed 2006 observations, solar
wind forcing and fixed trained weights. The independently resampled
2006 *acquisition-cluster bootstrap* CI measures a DIFFERENT
quantity, variation in the ensemble of acquisition cases.
Five draws cannot provide a robust or calibrated posterior
predictive distribution; the 10th–90th five-sample quantiles
are **descriptive empirical bounds, NOT a validated 80%
prediction interval**. In particular, do not claim the
11.35% figure is a formal 80%-coverage miscalibration test
or compare it against nominal 80% as if sample quantiles were
population quantiles. The low realized inclusion and small
sampling spread, however, reinforce the observed mismatch
between the generated spatial structures and hidden real
SSUSI peaks/gradients.

**Do not revise the original fixed-one-draw 2006 Table 12:**
conditional RMSE 0.661641975 and its paired baselines remain
the registered primary experiment. The five-seed RMSE
0.662220349 is a supplemental conditional random-draw mean,
not the updated benchmark and not a best-of-five selection.
The original evaluated checkpoint is SHA
`875b88edf91ca2a56d7a3041d55f956ca1ff3243bd137a44ef3e534fbce7ed12`.

Detailed verified fact record:
`verified_2006_stochastic_uncertainty_20261125.json`.
The event-level source report is local and not committed
because all private SSUSI case outputs are gitignored.

**Reviewer status:** R3-8 is now **empirically addressed**
but only for finite-draw stochastic sampling spread, NOT
for predictive interval calibration. Remaining high-priority
request: R2-6 explicitly label IMF Bz, V and dynamic
pressure directly on original Figures 4–7; then
R3-9 Table 2/3 explanation, R2 minor terminology,
R3-6 OVATION feature caveat, editorial data/software
availability and complete point-by-point English response
plus marked/clean manuscript package.

**Working manuscript after R3-8:** The locally rendered and structurally checked cumulative 36-page Word draft is `Solar-Wind-Conditioned Diffusion Model - R3-8 Stochastic Uncertainty Cumulative Working Manuscript.docx` (delivered to the author separately, not committed because it contains the full journal manuscript). New subsection **4.8**, new Section 2.5 evaluation-protocol paragraph, and an explicit Discussion limitation retain all 12 prior tables, 20 equation objects and 4 embedded images. This remains a WORKING manuscript; Figures 4–7 source solar-wind annotations, Table 2-vs-3 discrepancy, data/software release and marked-plus-clean manuscript packages are NOT DONE.


## 2026-11-25 — R2-6: Figure 4–7 original-OMNI labels; 6/7 already in working Word

Following the reviewer-completion gate, we independently matched
Fig. 4–7 conditions against the historical first-revision
`First_revision/R2-2_R2-4/event_context_results/event_context.csv`,
and for Fig. 6/7 independently cross-checked the audited
`Second_revision/R2-7_R3-10_R3-12_R3-13/timecorrected_figures_and_psd_verified.json`.
The four correct *physical-input* labels are:

| Figure | Absolute UTC meaning | Bz (nT) | Vsw (km/s) | Pdyn (nPa) |
| --- | --- | ---: | ---: | ---: |
| 4 (Polar/UVI) | 1996-04-03 02:15, exact-time 5-min OMNI record | -1.57 | 325 | 2.91 |
| 5 (Polar/UVI) | 1996-04-01 08:40, exact-time 5-min OMNI record | -3.80 | 365 | 2.69 |
| 6 (F16/SSUSI) | 2005-01-01 14:44:34–16:26:25, EDR interval MEAN | +1.69 | 459 | 3.18 |
| 7 (F16/SSUSI) | 2005-01-04 02:12:15–03:54:07, EDR interval MEAN | -2.13 | 656 | 2.89 |

**Do not label the 2005 SSUSI interval means as nearest-midpoint
5-min measurements.** That was a documented mismatch in an earlier
Figure 6/7 generation script; the corrected figures use
acquisition-mean OMNI and acquisition-mean 1-minute OVATION.

An immutable, original-data-validated annotation program
`annotate_figures_4_7_solar_wind.py` is committed, with
`test_annotate_figures_4_7_solar_wind.py`. It reads the original
historical CSV, crosschecks corrected SSUSI means and acquisition UTC,
and deterministically appends a black, white-on-black annotation
band with driver values and the correct time basis. It verifies that
**every original scientific-panel image pixel is byte-identical** to
the unchanged image region of the generated PNG. No inference,
training, OVATION rerun, or SSUSI preprocessing is performed.

**Already completed locally:** The author-provided
time-corrected 2005 Figure 6 and Figure 7 PNGs were
annotated, visually checked, and embedded in the latest
37-page cumulative WORKING Word draft. Figure 4/5 original
Polar/UVI PNGs were not supplied in the Word/available local
container, so **their plots themselves have NOT yet been
annotated**. Their captions were updated using the checked
1996 OMNI measurements. All 12 original tables, 20 equations,
4 image objects are retained; the working DOCX is delivered
to the author separately. The fine-grained audit is
`author_r2_6_verified_omni_figure_context_20261125.json`.

### Finish R2-6 without any NPU computation

Run from the author's repo checkout:

```bash
cd /home/docker/code/Aurora_DDPM_total
git pull --ff-only origin main

python -m unittest discover \
  -s paper_modif/Second_revision/R2-4_R2-8_R3-1 \
  -p 'test_*.py'

python paper_modif/Second_revision/R2-4_R2-8_R3-1/annotate_figures_4_7_solar_wind.py
```

Expected output in:
`paper_modif/Second_revision/R2-4_R2-8_R3-1/figure4_7_solar_wind_labeled/`

- `Figure4_solar_wind_labeled.png`
- `Figure5_solar_wind_labeled.png`
- `Figure6_solar_wind_labeled.png`
- `Figure7_solar_wind_labeled.png`
- `figure4_7_omni_label_audit.json`

The JSON must report `ready_for_R2_6=true` and
`n_annotated_figures=4`, and all four source PNG pixels
must be preserved. The script does NOT silently substitute
obsolete `First_revision/R2-1_R2-3/swath_reconstruction_diagnostics/`
2005 figures if the time-corrected source is unavailable:
it fails instead. If one of the corrected SSUSI source
PNGs does not exist on the server, pass its original
time-corrected path with `--figure6` or `--figure7`
rather than selecting an old reconstruction.

Return the **two newly labeled Polar files (Figure 4/5)**
and the small audit JSON. Then the cumulative Word manuscript
can embed and visually verify all four final annotated
science figure panels (currently only Figure 6/7 embedded).
**R2-6 remains PARTIAL until those Figure 4/5 PNGs are
confirmed and inserted into the final figure package.**

After R2-6, next open issue is R3-9's Table 2 versus
Table 3 RMSE protocol discrepancy, followed by explicit
R3-6 OVATION feature dependence and minor nomenclature,
then data/software release, clean+marked manuscript,
and the complete point-by-point English R2/R3 response.


## 2026-11-26 — Reviewer R3-9: Table 2 RMSE 0.233 versus Table 3 2005 RMSE 0.055 reconciled in text

**No new model runs or additional SSUSI cases.** The reviewer flagged
a seemingly inconsistent synthetic-OVATION conditional DDPM RMSE for
2005. We inspected the original 37-page cumulative Word draft,
`test_ddpm_indice.py`, `utils/normlize.py`,
`data/dataset_diff.py`, the R1-2 multiyear generator/benchmark,
the complete source-matched `r1_2_results/metrics_per_sample.csv`,
the 240-case manifest and the official phase summary.
Findings are in
`verified_R3_9_table2_vs_table3_protocol_20261126.json`.

- **Table 2 (0.233):** original January-2005 OVATION 1-min cube,
  candidate legacy script taking every 360 frames (6 h). Only
  manuscript-reported aggregate metrics are preserved; original
  per-frame metric output and exact historical executed source version
  were NOT found, so 0.233 cannot be independently recomputed.
- **Table 3 (0.055):** freshly generated OVATION maps from hourly OMNI,
  48 evenly distributed 2005 phase samples with UTC range
  2005-01-04 19:00 to 2005-12-28 04:00. Independent recomputation
  of 48 conditional records in the tracked R1-2 CSV gives
  **0.055028124338264206** (archived phase summary agrees).
- **Not a different mask:** both inspected test implementations use
  60°–80° MLAT and 18–06 MLT = **1,920/7,680 = 25%** of the
  80×96 map, with ordinary masked-target RMSE = sqrt(mean(error²)).
  The earlier manuscript description **"70% masked" was incorrect**
  for this benchmark. A separate aurora-dependent mask generator
  in `data/dataset_diff.py` aims for at least 70% but is not
  the fixed Table2 or Table3 evaluation mask.
- **Not an identical input-normalization/data pipeline:** the legacy
  January script calls `normalize(mn_datas)` and
  `OmniDataset(solar_data)` on **January 2005 TEST arrays**, and
  creates the seven-variable sequence `Bx,By,Bz,Vx,Vy,Vz,P`;
  the checked `SolarWindEncoder` actually treats the FIRST FIVE
  positions as `Bx,By,Bz,Vsw,Pdyn`. The R1-2 multiyear code instead
  correctly uses **five** fields and saved **training**
  normalizers. The exact executed original January source cannot
  be authenticated from its present-day file; a current-source field
  mismatch is a material concern and is *not* proof of the precise
  run responsible for 0.233. Inference-time normalization learned
  from test targets also limits the legacy statistic's independent
  test validity.
- **Correct conclusion:** the January-only Table2 0.233 and
  yearwide 2005 Table3 0.055 are **not commensurate**. Differences
  in sample population, source map generation, solar forcing
  and normalization are established, but their relative causal
  numerical contributions CANNOT be apportioned from available
  evidence. The 0.233 is retained as a clearly labeled legacy
  descriptive value, not compared directly to the authenticated
  0.0550 or to independent real SSUSI benchmarks.

**Manuscript action:** Section 2.4/2.5, Section 4.3–4.5 and Table2/3
captions have been amended accordingly. The cumulative working
Word `Solar-Wind-Conditioned Diffusion Model - R3-9 Table 2-3
Protocol Reconciliation Cumulative Working Manuscript.docx`
is delivered separately, rendered and checked: **38 pages,
12 tables, 20 equations, 4 embedded pictures**. Existing numerical
tables and 2005/2006 real SSUSI scores remain unmodified.
R3-9 textual explanation is ready with historical run-provenance
qualification, but the original unarchived Table2 per-sample output
cannot be independently certified. **No final reviewer Response
is released before the complete 25-response submission package.**
