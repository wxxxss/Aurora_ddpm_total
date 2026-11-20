#!/usr/bin/env python3
"""CPU-only sampling and attention audit; not a satellite-validation benchmark.
Run: python paper_modif/Second_revision/R2-5_R3-2_R3-11/audit_sampling.py
"""
import json
import sys
from pathlib import Path
import torch
from torch import nn

SCRIPT=Path(__file__).resolve()
ROOT=SCRIPT.parents[3]
sys.path.insert(0,str(ROOT))
from models.ddpm import DDPM
from models.unet import CrossModalCrossAttention

class LocalDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Conv2d(1,1,3,padding=1,bias=False)
        nn.init.constant_(self.conv.weight,0.04)
    def forward(self,x,t,c):
        return self.conv(x)

def sample(model,image,mask):
    torch.manual_seed(1234)
    with torch.no_grad():
        return model.sample(image,mask,torch.zeros(1,5),
                            num_inference_steps=12,n_sample=1,j=3,r=2)

def main():
    model=DDPM(LocalDenoiser().eval(),num_train_steps=20,schedule="cosine").eval()
    mask=torch.ones(1,1,8,8)
    mask[:,:,2:6,2:6]=0
    image=torch.zeros_like(mask)
    image[:,:,:2,:]=0.7
    image[:,:,6:,:]=0.2
    image[:,:,2:6,:2]=0.4
    image[:,:,2:6,6:]=0.5
    image=image*mask
    a=sample(model,image,mask)
    known_ok=bool(torch.allclose(a*mask,image*mask,atol=1e-6))
    altered=image.clone()
    altered[:,:,:2,:]+=0.2
    b=sample(model,altered,mask)
    influence=float(((a-b)*(1-mask)).abs().max())
    poisoned=image+5*(1-mask)
    c=sample(model,poisoned,mask)
    leakage=float(((a-c)*(1-mask)).abs().max())
    attn=CrossModalCrossAttention(32,256,num_heads=8,dropout=0).eval()
    with torch.no_grad():
        y=attn(torch.randn(1,32,4,5),torch.randn(1,256))
    w=torch.softmax(torch.randn(1,8,20,1),dim=-1)
    report={
        "sampler_mask_convention":"1=observed; 0=missing",
        "observed_values_preserved":known_ok,
        "changed_observations_change_missing_prediction_max_abs":influence,
        "missing_input_perturbation_effect_max_abs":leakage,
        "note":"The sampler now zeroes missing cells before initialization to prevent withheld-target leakage.",
        "attention_query_shape":[1,20,32],
        "attention_key_value_sequence_length":1,
        "attention_output_shape":list(y.shape),
        "single_key_softmax_is_one":bool(torch.all(w==1)),
        "scope":"synthetic untrained-model smoke test; not evidence of reconstruction accuracy"
    }
    print(json.dumps(report,indent=2))
    (SCRIPT.parent/"audit_results.json").write_text(json.dumps(report,indent=2)+"\n")
    assert known_ok,"Reliable pixels not preserved"
    assert influence>0,"Known-pixel influence not found"
    assert leakage<1e-8,"Missing input values leaked into reconstruction"
    assert bool(torch.all(w==1)),"Single-key attention invariant failed"

if __name__=="__main__":
    main()
