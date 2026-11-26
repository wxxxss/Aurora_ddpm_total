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
| R2-major-5 | Is model only prior generation and post hoc replacement? Prior-dependent mask? | **Text response implemented; experimental architecture superiority not established** | Revised Section 3.3 explicitly distinguishes repeated forward-noised observation reinjection in reverse diffusion (models/ddpm.py), final known-pixel copying, and the absence of an explicitly mask-conditioned trained U-Net. Real SSUSI heldout baselines constrain claims; no controlled learned-mask-aware or post-hoc-only denoiser comparison. |
| R2-major-6 | IMF Bz, speed, dynamic pressure directly IN Figures 4–7 | **Figure evidence completed; final artwork/Word consistency pending** | All original OMNI values and time bases verified for four cases. Corrected 2005 Figures 6/7 have driver banners directly in PNG panels and are embedded in the 37-page cumulative working Word draft; four captions updated. Author now uploaded four verified, OMNI-annotated Figure 4–7 PNGs and `figure4_7_omni_label_audit.json` (`ready_for_R2_6=true`, `n_annotated_figures=4`; all scientific source pixels preserved). The initial tiny label typography was subsequently enlarged in local candidates. Update final figure upload set and manuscript figure references/captions; current 37-page Word still embeds only 6/7, not 4/5. No model rerun. |
| R2-major-7 | Why Polar/UVI appears better than SSUSI | Addressed | Section 4.2 contrasts intensity-threshold constraints, swath support, acquisition geometry, and measurement quality |
| R2-major-8 | Is Apr–May 1996 sufficient across geomagnetic storms? | Partial | Independent January 2006 observed-year transfer, selected strict Kp>5 2005. Still only two months nominal training and no extreme storms; exact historical training dataset-to-weights linkage unverified |
| R2-minor-1 | Spell out Polar/UVI, IMAGE/FUV, DMSP/SSUSI at first mention | Open | Intro paragraph still uses abbreviated instrument/mission names without full expansion |
| R2-minor-2 | Fix ILBHL, jth, ΩM and global formatting | Partial | Main notation improved, but literal 'jth' and 'ΩM' remain; perform targeted notation/Word equation and citation proofreading |

## Reviewer #3

| ID | Requested issue | Status | Evidence / remaining action |
| --- | --- | --- | --- |
| R3-1 | Polar/UVI example times overlap OVATION Apr–May 1996 training period | Partial | New *genuine* Jan 2006 measured F16/SSUSI test, same checkpoints as 2005; exact original training-input provenance still unverified |
| R3-2 | Observed pixels/mask during denoising vs post hoc fusion | **Text response implemented with scope limits** | Revised Sections 3.3 and 5 detail S=1 native/selected retained pixels, S=0 missing; missing values sanitized before sampling; observed pixels re-noised/reinjected each reverse step and copied exactly at end. UNet inputs image/timestep/five-component solar vector only, not direct binary mask; no proof mask-aware training. |
| R3-3 | Synthetic OVATION vs direct OVATION baseline | Addressed | Synthetic target equals same OVATION reference by construction; direct OVATION on independent SSUSI in Tables 8–12 |
| R3-4 | Quantitative heldout real satellite data, revise strength of claims | Addressed | 2005/2006 SSUSI real withheld pixels, RMSE/MAE and caveats |
| R3-5 | Explain 0.122/0.102 peak/gradient attenuation, mask design | Addressed | Corrected/expanded 2005 & 2006 real peak/gradient and 1/2/4h flux-blind sector protocols, physical errors and smoothing caveats |
| R3-6 | OVATION uses By/Bz/V; Bx/P are not direct generation drivers | **Text response implemented** | Revised Sections 2.1/2.4/4.6/5 explicitly distinguish OVATION target-generator By/Bz/Vsw from supplementary DDPM Bx/Pdyn; small synthetic-target permutation effects cannot imply physical irrelevance/causality. |
| R3-7 | Single solar conditioning token: is attention actually selective? | **Mechanism explicitly corrected; empirical module ablation not done** | Revised Sections 3.2/5 show Q=(B,HW,32), ONE K/V tensor each (B,1,32) split over 8 heads; length-one softmax attention has weight 1, effectively broadcast global context. No evidence of attention spatial-key selection or a superior architecture compared with separately retrained broadcast/FiLM. |
| R3-8 | Multiple stochastic samples / reconstruction uncertainty | **Addressed with important caveat** | Author successfully completed five RNG-seeded original-sampler runs per each of 36 unchanged real-SSUSI holdout masks, 180 conditional reconstructions. Acquisition-equal mean RMSE 0.662220; mean within-mask per-draw RMSE SD 0.001977; mean hidden-pixel stochastic SD 0.019700. Original first draw reproduced to 2.22e-16 in RMSE. Mean observed-truth inclusion within five-draw q10–q90 empirical envelopes only 11.35%. This is NOT a calibrated 80% posterior prediction interval; do not use it as evidence of good uncertainty calibration. Integrate into final manuscript and reviewer response. |
| R3-9 | Explain Table 2 RMSE 0.233 vs Table 3 0.055 | **Text reconciliation completed; original Table 2 run numeric lineage remains unverified** | Source-based R3-9 audit `verified_R3_9_table2_vs_table3_protocol_20261126.json`: Table 2 legacy Jan-2005 6-hour one-minute cube versus Table 3 48 2005-yearwide freshly generated hourly-OMNI OVATION maps (exact 0.0550281243 recomputed). Both evaluation codes use identical 1920/7680 = 25% fixed 60–80 MLAT, 18–06 MLT masks and identical masked-pixel RMSE formula. Original manuscript 70% quantitative-mask claim was wrong. Jan-only candidate script fits flux/solar scaling to 2005 test set and supplies seven fields vs model's five interpreted fields; original exact script/pointwise Table 2 output has not been recovered, so DO NOT attribute numeric 0.233-to-0.055 difference causally or assert strict comparability. Main Word Sections 2.4/2.5, 4.3–4.5, Table 2/3 captions updated. |
| R3-10 | Grid extent, spacing, hemisphere, resampling, duplicate observations | Addressed | Section 2.1 grid and Section 2.3 SSUSI native-to-grid assignment/multiple-value averaging |
| R3-11 | OVATION-dependent observation mask, missed real structures | **Text response implemented with historical-mask caution** | Revised Sections 2.2/3.3/5 distinguish Polar ≥1 flux-selection constraint from native SSUSI UT_N measured support. Historical saved Polar outputs change OVATION-nonpositive low-flux pixels (172/172 Figure4 and 175/175 Figure5), contradicting strict OVATION-positive gating, but historical figure-run mask is not fully authenticated. Thresholded weak measured emission may be overwritten; do not claim validation of truly unobserved structures. |
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

1. **COMPLETED:** R3-8 fixed-mask five-seed stochastic uncertainty (180 draws) is already in Word Section 4.8. Figure 4–7 solar-wind source annotation audit now reports four figures complete. Final publication artwork typography and placement remain to be approved; NO new model inference needed.
2. **COMPLETED manuscript text patch for R3-9 and R2-5/R3-general/R3-2/R3-6/R3-7/R3-11** (not full reviewer-response closure): Table2/3 methodological difference, source-backed repeated observed-pixel injection, global broadcast-like one-token attention, OVATION generator By/Bz/V dependency, and Polar threshold versus native SSUSI support. Next: check optional uncensored Polar threshold sources for R3-12, then repair initialisms, notation, Open Research, and assemble Figure 4–7 final art.
3. Make publicly accessible preserved software/data or honestly qualify
   access restrictions and provide accessible archive/citation pathways.
4. Produce final clean manuscript, marked revised manuscript, standalone
   figures and a complete point-by-point response file. Do not claim
   exact source/dataset training lineage when only archived checkpoint
   byte identity is verified.

This is a **documented readiness audit**, not a new performance result
or evidence that reviewer concerns are all met.


## Final status crosscheck 2026-11-25 (after R2-6 figures and R3-8)

Original author response document `Response_Letter(1).docx` contains
**25** `[To be completed after revisions.]` placeholders: Reviewer #2
8 major + 2 minor, Reviewer #3 1 general critique + 14 numbered points.
Thus the experimental evidence is more advanced than the **formal
response letter (still not drafted)**.

Among the 24 numbered comments, 12 have sufficient substantive
evidence, 11 require partially addressed manuscript/analysis changes,
and 1 (R2 minor initialism expansion) remains unwritten. This is an
evidence-level snapshot, NOT an acceptance or submission readiness score.
Reviewer #3 general inpainting-method criticism is a separate high-risk
partial item: the actual `models/ddpm.py` sampler reinjects noised
observed pixels during reverse diffusion, but the network is not
trained with masked inputs, and the model is not known to reconstruct
missing event-specific fine structures.

**Manuscript check:** current 37-page R2-6 working DOCX has all 12 tables,
12 figure captions, but only four embedded images: Figures 6, 7, 8 and
11. Standalone figures rather than inline images may be the eventual
journal submission convention; what is needed is a consistent chosen
layout and a complete, approved figure upload set (not necessarily
inserting all 12 figures into the DOCX). R2-6 has four uploaded labeled
source-preserving files and corresponding OMNI audit, but Figure4/5
have not yet been integrated into Word.

**Prioritized next tasks:** First reconcile Table 2 RMSE .233 versus
Table 3 .055 (R3-9) directly from actual different mask/cohort protocols.
Then explain OVATION input feature limitations (R3-6), inpainting
sampler versus mask-aware denoiser (R2-5/R3-2/R3 general), selective
attention limitation (R3-7), prior-dependent Polar masking and 40
erg/cm2/s threshold (R3-11/R3-12). Fix first-use acronym expansion,
ILBHL, jth, ΩM, and double-ampersand citation. Correct the Open Research
claim of fully public software while validation code repository remains
private. Finally select clean/marked final manuscript conventions,
assemble final numbered figure files and create/verify 25 point-by-point
responses, including exact section/table/figure references. Original
training code/run/dataset provenance is still unverified even though
both checkpoint files were proven byte-identical to named archived files.

## 2026-11-26: R3-9 original Table2 versus Table3 audited and qualified in manuscript

A complete source-based comparison is now archived in `verified_R3_9_table2_vs_table3_protocol_20261126.json`. The two 2005 results are **different test populations/data-generation and normalization pipelines**. They do **not** use different missing-region geometry or RMSE formulas: the original 2005 candidate script `test_ddpm_indice.py` and R1-2 `benchmark_utils.py` both mask 60–80 MLAT, 18–06 MLT = 1,920/7,680 = 25%, correcting the manuscript's old 70% claim. The source-backed R1-2 Table3 declining-phase sample has **48** 2005 timestamps from January 4 through December 28 and mean RMSE **0.0550281243382642**, reproduced exactly from tracked per-sample CSV. The older Table2 historical 0.233 has no committed per-frame result log; its candidate script reads seven OMNI columns including `Vx,Vy,Vz` where the currently inspected model encoder expects five scalar positions `Bx,By,Bz,Vsw,Pdyn`; it also fits normalizers on the January test data. Consequently Table2 is a **legacy descriptive result without fully authenticated run provenance**, not a valid direct numerical comparator to Table3; no unsupported attribution or new model experiment has been made. Word revised in Sections 2.4/2.5, 4.3–4.5 and captions, with the original numbers preserved. The 38-page cumulative revised working DOCX was rendered and layout-checked and preserves 12 tables, 20 equation objects and four embedded pictures. **Next:** integrated method text revisions (R2-5/R3-general/R3-2/R3-6/R3-7/R3-11); final responses remain unissued.

## 2026-11-26 — METHOD TEXT PATCH COMPLETE (39-page cumulative working draft)

Core textual explanations for Reviewer #2 major 5 and Reviewer #3 general, 2, 6, 7, 11 have been implemented directly in Sections 2.1–2.2, 2.4, 3.2–3.3, 4.6 and 5 of the cumulative Word manuscript, with no new DDPM inference or revised numerical result. Verified provenance record: `verified_R2_5_R3_general_R3_2_R3_6_R3_7_R3_11_text_revision_20261126.json`. The local delivered 39-page working manuscript retains the original 12 tables, 20 OMML equation objects and 4 byte-identical embedded figure images. The revised manuscript now explicitly acknowledges that a single attention key is mathematically broadcast-like and is **not** shown superior to separately retrained FiLM/global conditioning; the U-Net is not explicitly mask-trained even though the reverse sampler repeatedly reinjects measurements; and Polar's brightness-selected mask can remove valid faint precipitation rather than true unsupported coverage. R3-6 input feature importance on synthetic OVATION must not be interpreted causally. R3 general's strong claims remain bounded by real SSUSI measurements, which do **not** support superior performance versus unconditional or interpolation baselines.

**Next:** R3-12 only-if-source-available uncensored Polar preprocessing screening (existing source audit first; do not perform invalid sensitivity on censored arrays); then R2 minor first-use names/notation, figure package and journal-required data/software availability, and complete clean/marked Word + 25 author-reviewed replies. Historical training-source run identity is still unverified even after exact binary checkpoint archive match. This remains NOT READY TO SUBMIT.
