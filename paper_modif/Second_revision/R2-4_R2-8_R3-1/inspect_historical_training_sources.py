#!/usr/bin/env python3
"""Compare the two previously untracked local UNet-v1 copies to training code.

The user located models/unet_v1.py in Aurora_DDPM_final and Aurora_DDPM.
This tool records SHA-256 and source import markers, without loading weights
or claiming that either version created the published checkpoint.

Run from Aurora_DDPM_total checkout root:
 python paper_modif/Second_revision/R2-4_R2-8_R3-1/inspect_historical_training_sources.py

File-hash equality shows identical source bytes, NOT checkpoint provenance.
Original run command, training logs and saved run metadata are still needed.
"""
from __future__ import annotations
import argparse,hashlib,json,re
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
DEFAULT_COPIES=[
 Path("/home/docker/code/Aurora_DDPM_final/models/unet_v1.py"),
 Path("/home/docker/code/Aurora_DDPM/models/unet_v1.py"),
 ROOT/"models/unet_v1.py",
]
DEFAULT_TRAIN=[
 Path("/home/docker/code/Aurora_DDPM_final/train.py"),
 Path("/home/docker/code/Aurora_DDPM/train.py"),
 ROOT/"train.py",
]

def describe_file(p):
    p=Path(p)
    result={"path":str(p),"exists":p.is_file()}
    if not p.is_file():return result
    raw=p.read_bytes()
    text=raw.decode("utf-8",errors="replace")
    result.update({
        "size_bytes":len(raw),
        "sha256":hashlib.sha256(raw).hexdigest(),
        "line_count":len(text.splitlines()),
        "defines_unet":bool(re.search(r"(?m)^class\s+UNet\b",text)),
        "imports_unet_v1":"models.unet_v1" in text,
        "imports_diff2_loader":"data.dataset_diff2" in text,
        "uses_1996_year_literal":"[1996]" in text,
        "contains_save_checkpoint":bool(re.search(r"torch\.save|save_checkpoint",text)),
    })
    return result

def inspect(copies,train_scripts):
    candidates=[describe_file(p) for p in copies]
    scripts=[describe_file(p) for p in train_scripts]
    hashes={item["sha256"] for item in candidates if item["exists"]}
    return {
        "candidate_unet_v1_files":candidates,
        "candidate_training_scripts":scripts,
        "n_distinct_available_unet_source_hashes":len(hashes),
        "same_unet_v1_bytes_if_multiple_copies":len(hashes)==1 if len([i for i in candidates if i["exists"]])>1 else None,
        "checkpoint_training_lineage_verified":False,
        "interpretation":[
            "Two locally available UNet-v1 files remove the claim that the architecture is wholly missing. They do not show which source was used to create current saved weights.",
            "Original conditional checkpoint save directory in current train.py differs from tested checkpoint file; source and checkpoint hashes alone do not establish how weights were produced.",
            "Seek contemporaneous training logs, run launch commands, and archived source snapshots/normalization parameters before claiming exact checkpoint provenance.",
            "No model weights are loaded or deserialized by this tool.",
        ]
    }

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--unet-copies",type=Path,nargs="+",default=DEFAULT_COPIES)
    p.add_argument("--training-scripts",type=Path,nargs="+",default=DEFAULT_TRAIN)
    p.add_argument("--output",type=Path,default=HERE/"historical_training_sources_audit.json")
    args=p.parse_args()
    result=inspect(args.unet_copies,args.training_scripts)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"n_distinct_unet_v1_versions":result["n_distinct_available_unet_source_hashes"],
                      "same_unet_v1_bytes":result["same_unet_v1_bytes_if_multiple_copies"],
                      "sources":result["candidate_unet_v1_files"],
                      "training_scripts":result["candidate_training_scripts"],
                      "checkpoint_training_lineage_verified":False,
                      "output":str(args.output)},indent=2))

if __name__=="__main__":
    main()
