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
