# Second-revision Reviewer #2 / Reviewer #3 / Editorial completion gate
Date: 2026-11-23. Status: **NOT COMPLETE / DO NOT SUBMIT YET**.

Basis: Original Reviewer #2 2-page PDF (eight major comments, two minor),
original Reviewer #3 full formal review (14 specific issues after the
general critique), the editor's instructions in the original notice,
the full verified-checkpoint-identity cumulative Word manuscript,
the actual evaluation/reverse-sampling code in this repository,
and validated 2005–2006 real SSUSI results.

Legend: **Addressed** = current evidence/manuscript substantively responds
within honest scope; **Partial** = manuscript acknowledges an important
limitation but reviewer request not fully satisfied; **Open** =
an explicitly requested item/artifact not delivered or not verified.

## Reviewer #2

| ID | Requested issue | Status | Evidence / remaining action |
| --- | --- | --- | --- |
| R2-major-1 | Limitations of OVATION training; DDPM value beyond OVATION | Addressed with qualified conclusion | Real measured-pixel benchmarks and direct OVATION; acknowledge no systematic gain versus unconditional/interpolation |
| R2-major-2 | Artificially mask valid satellite observations | Addressed | 2005 and 2006 flux-blind F16/SSUSI observed-truth sector holdouts |
| R2-major-3 | Direct OVATION baseline on independent satellite targets | Addressed | Tables 8–12, paired event statistics, duplicate observed-fused OVATION clearly identified |
| R2-major-4 | Kp >5 disturbed events | Addressed within limited Kp range | Six strict Kp>5 real acquisitions (Kp 5.3–6.0); no extrapolation to Kp>=7 |
| R2-major-5 | Is model only prior generation and post hoc replacement? Prior-dependent mask? | Partial | Actual models/ddpm.py repeatedly forward-noises and reinjects valid observed pixels into reverse sampling. Trained on complete OVATION rather than masked examples; document exactly and avoid claiming learned mask-aware network or proven improvement over simple OVATION fusion |
| R2-major-6 | IMF Bz, speed, dynamic pressure directly IN Figures 4–7 | Open / not verified | Values appear in main text, not shown as confirmed labels in the finalized figures/captions. Export final Figure 4–7 images and verify panel-level annotations. |
| R2-major-7 | Why Polar/UVI appears better than SSUSI | Addressed | Section 4.2 contrasts intensity-threshold constraints, swath support, acquisition geometry, and measurement quality |
| R2-major-8 | Is Apr–May 1996 sufficient across geomagnetic storms? | Partial | Independent January 2006 observed-year transfer, selected strict Kp>5 2005. Still only two months nominal training and no extreme storms; exact historical training dataset-to-weights linkage unverified |
| R2-minor-1 | Spell out Polar/UVI, IMAGE/FUV, DMSP/SSUSI at first mention | Open | Intro paragraph still uses abbreviated instrument/mission names without full expansion |
| R2-minor-2 | Fix ILBHL, jth, ΩM and global formatting | Partial | Main notation improved, but literal 'jth' and 'ΩM' remain; perform targeted notation/Word equation and citation proofreading |

## Reviewer #3

| ID | Requested issue | Status | Evidence / remaining action |
| --- | --- | --- | --- |
| R3-1 | Polar/UVI example times overlap OVATION Apr–May 1996 training period | Partial | New *genuine* Jan 2006 measured F16/SSUSI test, same checkpoints as 2005; exact original training-input provenance still unverified |
| R3-2 | Observed pixels/mask during denoising vs post hoc fusion | Partial / substantial code clarification | models/ddpm.py sample/step actually reinjects forward-noised observed pixels each step; observed mask is used in the sampler, not an explicit denoiser input, and training is on complete maps; report RePaint-style inference mechanism accurately and avoid claims of mask-aware training |
| R3-3 | Synthetic OVATION vs direct OVATION baseline | Addressed | Synthetic target equals same OVATION reference by construction; direct OVATION on independent SSUSI in Tables 8–12 |
| R3-4 | Quantitative heldout real satellite data, revise strength of claims | Addressed | 2005/2006 SSUSI real withheld pixels, RMSE/MAE and caveats |
| R3-5 | Explain 0.122/0.102 peak/gradient attenuation, mask design | Addressed | Corrected/expanded 2005 & 2006 real peak/gradient and 1/2/4h flux-blind sector protocols, physical errors and smoothing caveats |
| R3-6 | OVATION uses By/Bz/V; Bx/P are not direct generation drivers | Partial | Permutation importance appropriately noncausal but manuscript should explicitly note absent Bx/P from direct OVATION-generator inputs as an *a priori* explanation for weak synthetic-target sensitivity |
| R3-7 | Single solar conditioning token: is attention actually selective? | Partial | Section 3.2 gives (B,HW,32) queries and one (B,1,32) K/V, acknowledges broadcast not spatial selectivity. No retrained FiLM/broadcast baseline as suggested |
| R3-8 | Multiple stochastic samples / reconstruction uncertainty | **Addressed with important caveat** | Author successfully completed five RNG-seeded original-sampler runs per each of 36 unchanged real-SSUSI holdout masks, 180 conditional reconstructions. Acquisition-equal mean RMSE 0.662220; mean within-mask per-draw RMSE SD 0.001977; mean hidden-pixel stochastic SD 0.019700. Original first draw reproduced to 2.22e-16 in RMSE. Mean observed-truth inclusion within five-draw q10–q90 empirical envelopes only 11.35%. This is NOT a calibrated 80% posterior prediction interval; do not use it as evidence of good uncertainty calibration. Integrate into final manuscript and reviewer response. |
| R3-9 | Explain Table 2 RMSE 0.233 vs Table 3 0.055 | Partial | Different synthetic mask/cohort protocols and sampling are described separately, but NO direct comparison/explanation at the apparent discrepancy; add concise explicit reconciliation |
| R3-10 | Grid extent, spacing, hemisphere, resampling, duplicate observations | Addressed | Section 2.1 grid and Section 2.3 SSUSI native-to-grid assignment/multiple-value averaging |
| R3-11 | OVATION-dependent observation mask, missed real structures | Partial / strongly clarified | Polar/UVI stated to use flux threshold independent of OVATION; threshold masks weak measured precipitation. SSUSI support is native UT_N not OVATION; note Polar evidence is threshold-defined and cannot validate truly missed structures |
| R3-12 | LBHL conversion uncertainties; 40 erg threshold | Partial | Section 2.2 acknowledges conversion assumptions and uncalibrated empirically chosen 40 cutoff. No independent uncensored sensitivity or physical threshold justification |
| R3-13 | 8,202 correlated edges, 79 PSD segments only two cases | Addressed | Regridded 8,194 support-matched edges, 79 segments, two EDR clusters, descriptive only |
| R3-14 | 23 s/image, thousands of transitions not near-real-time | Addressed with limitations | Table 6 reports 22.92±0.30 s, 5520 transitions/2909 denoising calls; Discussion acknowledges computational cost; ensure final text does not imply operational near-real-time |

## Editorial submission requirements

| Item | Status |
| --- | --- |
| Full revised science manuscript | **Working draft exists, not submission-final**; section/figure/table citation proofreading needed |
| One complete point-by-point R2+R3 response file | **Open**; earlier Response_Letter is not an authenticated final R2/R3 reply |
| Marked/highlighted changes version | **Open**; latest 35-page Word file is the incorporated working manuscript, not a current tracked/marked revision |
| Final clean manuscript + complete standalone publication-quality figures | **Partial**; 12 figure captions versus four embedded media objects; external final figure set not independently verified |
| Data/software availability, public repository, durable citation | **Open**; manuscript claims "data and software ... publicly available" while author's significant evaluation code is in a private GitHub repository; no preserved publicly accessible release/DOI verified. The "Papitashvili & & King" duplicated citation also needs correction |
| COI declaration | Addressed | Article File includes no-conflict statement |
| All authors' revision approval and corresponding author ORCID | Not independently verifiable in manuscript; corresponding author must confirm before submission |
| Journal formatting/reference crosscheck | Open | Check abbreviation first occurrence, math typography, all figure panels and in-text references |

## Next bounded actions, in order

1. **One narrow new measurement**, not more model training: fixed-case,
   fixed-mask, multiple DDPM random draws (prefer complete 12-acquisition
   2006 cohort, at least five seeds); separately quantify within-case
   stochastic variability and prediction uncertainty, no checkpoint tuning.
2. Correct explicit Text/Table 2 vs Table 3 protocol discrepancy,
   OVATION generating-feature explanation, acronyms, jth/ΩM and broken
   Open Research citation, and Figure 4–7 annotations.
3. Make publicly accessible preserved software/data or honestly qualify
   access restrictions and provide accessible archive/citation pathways.
4. Produce final clean manuscript, marked revised manuscript, standalone
   figures and a complete point-by-point response file. Do not claim
   exact source/dataset training lineage when only archived checkpoint
   byte identity is verified.

This is a **documented readiness audit**, not a new performance result
or evidence that reviewer concerns are all met.
