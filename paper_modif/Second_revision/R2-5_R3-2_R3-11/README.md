# R2-5 / R3-2 / R3-11: Observation-constrained sampling and masks

The current paper claims a final observation overwrite, while `models/ddpm.py` additionally injects noisy known pixels at each reverse step. This batch audits the actual sampling rule and the observation-mask convention. The attention implementation audited alongside is related to R3-7.

Commands from repository root (CPU-compatible; requires PyTorch and NumPy):

```bash
python paper_modif/Second_revision/R2-5_R3-2_R3-11/audit_sampling.py
```

This synthetic smoke test proves *code-path behavior*, not reconstruction accuracy on satellite observations. For the scientific response, independently verify which checkpoint/model/version produced manuscript Figures 4-7 and ensure that masks use `1=observed, 0=missing` in `DDPM.sample`. Ground-truth-hidden pixels must never reach the sampler, including the initial noise state. Follow-up real-observation validation is tracked separately.
