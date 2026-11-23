#!/usr/bin/env python3
"""Lightweight provenance inspection for the checkpoints used in the paper.

No model inference, no checkpoint unpickling and no private raw-file upload.
Hash actual conditional/unconditional checkpoint, normalization file and
the training/loader scripts that exist on the author's local checkout.

A SHA-256 hash identifies a file version but cannot prove which data loader
or git commit was used during training. Original training logs, scripts or
experiment metadata are needed to establish causal training lineage.

Use from repository root:
 python paper_modif/Second_revision/R2-4_R2-8_R3-1/audit_checkpoint_provenance.py

If the paper used different paths, specify --cond-checkpoint / --uncond-checkpoint
and --cond-norm / --uncond-norm. No weights are read or posted.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]

def fingerprint(path:Path):
    path=Path(path)
    d={"relative_or_local_path":str(path),"exists":path.is_file()}
    if not path.is_file():return d
    hash=hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk=stream.read(8<<20)
            if not chunk:break
            hash.update(chunk)
    d.update(bytes=path.stat().st_size,sha256=hash.hexdigest())
    return d

def source_training_config(path):
    path=Path(path)
    if not path.is_file():return {"source_missing":True}
    text=path.read_text(encoding="utf-8",errors="replace")
    patterns={
      "years":r"(?m)^\s*years\s*=\s*(\[[^\n]+\])",
      "months":r"(?m)^\s*months\s*=\s*(range\([^\n]+\))",
      "save_dir":r"(?m)^\s*save_dir\s*=\s*(.+)$",
      "n_epochs":r"(?m)^\s*NUM_EPOCH\s*=\s*(\d+)",
      "train_ratio":r"(?m)^\s*train_ratio\s*=\s*([0-9.]+)",
    }
    out={}
    for name,pat in patterns.items():
        matches=re.findall(pat,text)
        out[name]=matches[:5]
    out["training_architecture_import_unet_v1"]="from models.unet_v1 import UNet" in text
    out["training_data_loader"]="data.dataset_diff2" if "from data.dataset_diff2 import" in text else "not identified"
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cond-checkpoint",type=Path,default=ROOT/"ckpt/cond/aurora_diff_best.pth")
    p.add_argument("--uncond-checkpoint",type=Path,default=ROOT/"ckpt/uncond/aurora_diff_best.pth")
    p.add_argument("--cond-norm",type=Path,default=ROOT/"ckpt/cond/norm_params.pkl")
    p.add_argument("--uncond-norm",type=Path,default=ROOT/"ckpt/uncond/norm_params.pkl")
    p.add_argument("--output",type=Path,default=HERE/"checkpoint_provenance_audit.json")
    args=p.parse_args()
    objects={
      "conditional_checkpoint":args.cond_checkpoint,
      "unconditional_checkpoint":args.uncond_checkpoint,
      "conditional_normalization":args.cond_norm,
      "unconditional_normalization":args.uncond_norm,
      "current_train_source":ROOT/"train.py",
      "current_data_loader_source":ROOT/"data/dataset_diff2.py",
      "current_diffusion_source":ROOT/"models/ddpm.py",
      "current_conditional_unet_source":ROOT/"models/unet.py",
      "training_imports_unet_v1_source":ROOT/"models/unet_v1.py",
    }
    result={
      "file_fingerprints":{name:fingerprint(path) for name,path in objects.items()},
      "current_public_train_py_config":source_training_config(objects["current_train_source"]),
      "checkpoint_lineage_verified":False,
      "checkpoint_dataset_fingerprints_verified":False,
      "no_checkpoint_unpickling":True,
      "limitations":[
        "Hashes identify file bytes but do not establish that the present train.py or dataset_diff2.py created the selected model weights.",
        "The main training script imports models.unet_v1, which may be an unpublished/local architecture version. Compare original training source and inference checkpoint architecture if that file is absent.",
        "Training provenance needs the ORIGINAL run command, training source commit, logs, data creation workflow and normalization; files presently tracked in GitHub may postdate training.",
        "Do not treat successfully inferred 5-minute pairing of 1996 input arrays as proof of the actual checkpoint's training lineage.",
      ]
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    digest={k:{"exists":v["exists"],"bytes":v.get("bytes"),
               "sha256_prefix":v.get("sha256","")[:16]}
            for k,v in result["file_fingerprints"].items()}
    print(json.dumps({"files":digest,
                      "current_train_py":result["current_public_train_py_config"],
                      "checkpoint_lineage_verified":False,
                      "output":str(args.output)},indent=2))
if __name__=="__main__":main()
