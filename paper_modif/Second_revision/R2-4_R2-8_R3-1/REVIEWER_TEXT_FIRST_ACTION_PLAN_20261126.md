# JGR: Space Physics 2026JA035824R — evidence-first, text-first completion plan (2026-11-26)

## Policy (approved by author)
Use **existing evidence + scientifically precise manuscript wording** wherever this directly answers the reviewer. Do **not** create an experiment merely to supply a rhetorically attractive result. Run a **minimal, preregistered verification** only if an essential factual claim cannot be substantiated by existing code/source/results, and no honest scope limitation can answer the criticism. Never characterize an unverified physical premise or source-version relation as a proved fact.

Evidence sources: original Reviewer #2 PDF (8 major + 2 minor), original Reviewer #3 formal text (general methodological criticism + 14 specific), editor's revision letter; current 37-page working DOCX, verified 2005 and 2006 held-out SSUSI CSVs, 6+6 Kp real holdout, 180 fixed-mask five-seed runs, source code and checkpoint SHA match audits, four-file Figure4–7 solar-wind labels.

**No DDPM training, 2007–2009 expansion, extra 2006 orbit selection, or new reverse sampling planned.** The substantive remaining tasks are **text/analysis and submission materials**.

## Classification codes
- **T — Text with existing audited evidence:** revise the main manuscript, limits, tables/captions/figures references; prepare evidence-specific response. No new experiment.
- **A — Audit/reconciliation of EXISTING products:** check original code/CSV and recompute a summary if needed. Not new model inference.
- **C — Conditional minimum extra experiment:** only if there is required primary evidence, appropriate pre-QC sources exist, and a bounded analysis can meaningfully change a scientifically important claim.
- **P — Publication packaging:** non-experimental issue (figure and software availability, clean/marked manuscript, editor package).

## Reviewer #2 — ten specific comments

| ID | Decision | Exact implementation and evidence boundary |
| --- | --- | --- |
| R2-major-1 | T | Explain that held-out real SSUSI conditional DDPM improves direct OVATION but generally not unconditional DDPM or interpolation. Avoid broad "DDPM best" claims. |
| R2-major-2 | T | Describe and cite the 2005+2006 genuinely measured withheld SSUSI 1/2/4-h MLT sectors and disjoint inputs/targets. |
| R2-major-3 | T | Include direct OVATION and acknowledge the observation-fused OVATION method duplicates direct OVATION on hidden pixels. |
| R2-major-4 | T | Cite 2005 6 Kp>5 versus 6 Kp<=3 measured cases, activity limits (about Kp5.3–6.0); no Kp>=7 generality. |
| R2-major-5 | T (+ optional A if text ambiguous) | Distinguish actual noisy-observation reinjection in reverse sampler, denoiser not mask-conditioned during training, output known-pixel preservation; explain OVATION-fusion baseline and mask lineage. |
| R2-major-6 | P/T | Four OMNI labeled Figure4–7 PNGs verified, including Polar exact 5-min time vs F16 interval mean. Approve readable labels; package complete standalone figures and matching captions; do not run NPU. |
| R2-major-7 | T | Explain different Polar threshold-defined "known" pixels vs native SSUSI swath support; visibility does not establish independent retrieval accuracy. |
| R2-major-8 | T | Qualify nominal 1996 two-month training, distinct 2005+2006 real SSUSI, limited 2005 Kp>5 cases; no training-distribution sufficiency proof. Checkpoint bytes traceable to named historical archive, original source/dataset and Figure3 trace not authenticated. |
| R2-minor-1 | T | Expand Polar/UVI, IMAGE/FUV and DMSP/SSUSI on first occurrence, uniformly. |
| R2-minor-2 | T | Fix ILBHL, jth, ΩM, mathematical scripts, symbols and figure/units/reference typography. |

## Reviewer #3 — general criticism plus fourteen specifics

| ID | Decision | Exact implementation and evidence boundary |
| --- | --- | --- |
| R3-general | T | Correct the false post-hoc-only generation description: actual sampler uses repeated forward-noised observation reinjection; distinguish this from a denoiser trained on masked inputs. Point to measured 2005/2006 hidden SSUSI results but concede event-specific peaks/small-scale fidelity is weak. |
| R3-1 | T | Retain 2006 calendar-disjoint genuine satellite validation and known evaluated-checkpoint hashes; do not certify 1996 training source/data or Figure3 run lineage. |
| R3-2 | T | Explain inverse-diffusion observation injection, mask convention and no direct mask input to UNet; no false claims of learned mask-aware training. |
| R3-3 | T | Synthetic OVATION target means direct OVATION is near-identity by construction; independently time-matched OVATION is tested fairly on real SSUSI. |
| R3-4 | T | Give real SSUSI heldout RMSE/MAE and matched-baseline evidence; revise Abstract/Key Points/Discussion for limits. |
| R3-5 | T | Explain sector selection, raw-valid measured target definition, number of EDR clusters, aggregate vs per-sector peak/gradient-ratio denominators, substantial auroral peak/edge smoothing. |
| R3-6 | T | Explicitly state OVATION generating targets depend directly on By/Bz/V in the described pipeline, not Bx/P; low Bx/P permutation effect does not show physical irrelevance. |
| R3-7 | T, optional C only for retained stronger claim | Demonstrate one K/V token implies softmax weight 1 and effective broadcast global condition, not selective spatial attention. Withdraw any empirically unproven advantage over broadcast/FiLM. A retrained conditioning-module ablation is only needed if author insists that module selectivity/superiority is a main original contribution. |
| R3-8 | T (experiment DONE) | 5 seed draws on each of 36 fixed real SSUSI sectors, 180 draws; average RMSE SD 0.001977, pixel SD 0.019700 and 11.35% five-sample empirical envelope inclusion. Already in Section4.8; expressly *not* a calibrated 80% prediction interval. |
| R3-9 | **A first, then T** | Directly audit Table2 Jan2005 conditioned synthetic RMSE 0.233 against Table3 2005 declining-phase synthetic 0.055. Match original file IDs, timestamp cohorts, mask geometry/coverage, error formula, aggregation and normalization, then insert an explicit two-sentence methodological reconciliation and captions. If source files cannot establish comparability, mark results as different tests, avoid causal attribution, and correct/withdraw unsupported pooled comparison; do not invent an explanation. |
| R3-10 | T | MLAT 50–90 deg, 80 centers, MLT 0..23.75h with 0.25h center spacing, northern maps, nearest-center binning and averaging multiple valid native SSUSI observations. Cite corrected time-aligned figures/experiments. |
| R3-11 | T | Historical Polar mask is measured-flux-threshold-defined and not a confirmed OVATION-positive filter; thresholded out weak observed precipitation. Native SSUSI support uses UT_N not OVATION. State coverage selection bias and missing support uncertainty. |
| R3-12 | **T with explicit uncertainty; C only if raw unthresholded source exists** | State LBHL-to-flux empirical calibration assumption and 40 erg cm^-2 s^-1 artifact cutoff is QC heuristic, NOT validated physical rejection threshold. If genuine pre-cutoff calibrated Polar flux and quality flags are accessible, optional fixed 20/40/60 cutoff sensitivity (counts/robustness) suffices. Do not recompute "sensitivity" using already censored products. If unavailable, do not assert physical optimality: report limitation, consider moving fragile Polar-specific quantitative claim to SI. |
| R3-13 | T | Keep corrected 8194 spatial edges, 79 segments across 2 EDRs; descriptive only, no pixel/segment-based independent-event significance or confidence interval. |
| R3-14 | T | Report ~22.92 +/- 0.30 s and steps/calls; remove near-real-time and operational readiness claims. |

## Non-experimental mandatory editorial deliverables (P)

1. Make one authoritative final text of all equations/tables, figure references and limits. Word working draft has twelve captions, only four embedded images; **the editor explicitly asks for separate complete final figure files**, so decide whether a reading-copy DOCX embeds figures while the Article File uses captions and standalone uploads. Do not interpret "not embedded" as "missing" if a verified standalone Figure exists.
2. Check standalone Figure1–12 files and captions: Figure4–7 label/readability verified (large-text candidates need final approval), Figure3 loss-curve must not be certified as the evaluated checkpoint's training trace.
3. Correct Open Research statement. A private GitHub project does not justify "all software publicly available"; obtain author authorization for a preserved public code release and repository citation, or give an accurate documented restriction consistent with editor's availability requirements.
4. Create the final **clean revision**, separately **marked changes** revision, complete figure submissions, and one **25-entry point-by-point response** (Reviewer #2 10; Reviewer #3 general+14). All answers must cite correct final section/table/figure/page. Do not invent responses or claim a completed action before manuscript change.
5. Corresponding-author ORCID, author approvals, COI, supporting-information bundle and journal checklist.

## Minimum sequential execution with explicit stop rules

**Phase A — Source-consistency pass, NO NPU:**
1. R3-9 original Table2 vs Table3 data/mask/protocol audit. Deliver table with source products, exact populations, mask formula, error normalization and explanation. Required for any confident textual reconciliation.
2. One integrated method patch: R2-5 + R3-general + R3-2 + R3-6 + R3-7 + R3-11; correct training provenance boundaries R2-8/R3-1 and Figure3.
3. One limitations/results patch: R2-1/2/3/4/7, R3-3/4/5/8/10/13/14. Existing primary/secondary measured results unchanged.
4. R2-minor and references cleanup; check all captions and table precision.

**Phase B — The only possible NEW scientific measurement:**
- R3-12 pre-QC Polar cutoff sensitivity **ONLY if uncensored data exist** and threshold robustness is essential; otherwise the explicit limitation is the scientifically correct response.
- R3-7 FiLM/broadcast ablation **ONLY if insisting on a superior selective attention claim**, otherwise simplify the claim, no new model training.
- Any discovered source mismatch in R3-9 could require a narrow recalculation of *existing* metrics, not a fresh diffuse-model study unless the underlying saved outputs are demonstrably incorrect.

**Phase C — Submission production, NO model experiment:**
complete independent Figure package, Open Research availability, clean/marked DOCX, 25 comprehensive responses and final editor checklist.

**Hard gate before claiming "done":** each comment's source fact has a traceable file/section, each requested artifact exists, no unsupported causal/generalization claims, training provenance unproven portions explicitly bounded, and all 25 Responses map to actual changes. 
