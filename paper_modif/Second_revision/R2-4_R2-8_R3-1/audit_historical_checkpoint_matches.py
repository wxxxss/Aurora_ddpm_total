#!/usr/bin/env python3
"""Bounded, read-only inventory of historical model checkpoint file identity.

The 2005 and 2006 F16/SSUSI benchmarks used the CURRENT Aurora_DDPM_total
conditional/unconditional checkpoints. Their lineage to old
Aurora_DDPM_final train.py and unet_v1.py remains unverified. In particular,
the three compared conditional best.pth binary SHA-256 values do not match.

The prior shell find | head -100 inventory omitted most directories.
This inventory:
- traverses only the supplied local checkpoint directories;
- inventories ALL *.pth/*.pt/*.ckpt files (no head truncation);
- hashes candidate historical files ONLY if their size matches the known
  currently used conditional/unconditional checkpoint sizes; uses
  streaming SHA-256, never deserializes or executes weights;
- reports all exact binary matches; a negative match is NOT proof the
  tensor parameters or original training source differ;
- records file-level architecture labeling only as filename provenance,
  NOT as verified training lineage.
- writes a small JSON report, not weights or private satellite files.

From Aurora_DDPM_total repository root:
  python paper_modif/Second_revision/R2-4_R2-8_R3-1/audit_historical_checkpoint_matches.py

Other archive dirs can be supplied with --roots /path/ckpt /other/path/ckpt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
DEFAULT_CONDITIONAL=REPO/"ckpt/cond/aurora_diff_best.pth"
DEFAULT_UNCONDITIONAL=REPO/"ckpt/uncond/aurora_diff_best.pth"
DEFAULT_ARCHIVES=[
    Path("/home/docker/code/Aurora_DDPM_final/ckpt"),
    Path("/home/docker/code/Aurora_DDPM/ckpt"),
]
EXTENSIONS={".pth",".pt",".ckpt"}
EXCLUDED_DIRS={".git","__pycache__",".venv","venv"}

def streaming_sha256(path:Path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as src:
        for block in iter(lambda:src.read(8<<20),b""):
            digest.update(block)
    return digest.hexdigest()

def file_record(path,*,hash_file=False):
    path=Path(path)
    stat=path.stat()
    record={
        "path":str(path),
        "bytes":stat.st_size,
        "last_modified_utc":datetime.fromtimestamp(
            stat.st_mtime,tz=timezone.utc).isoformat(),
    }
    if hash_file:
        record["sha256"]=streaming_sha256(path)
    return record

def list_checkpoints(roots):
    """Stat only matching suffixes; no weight deserialization, no truncation."""
    records=[]
    seen=set()
    roots_found=[]
    for root in roots:
        root=Path(root)
        roots_found.append({"root":str(root),"exists":root.is_dir()})
        if not root.is_dir():
            continue
        for dir_name,subdirs,files in os.walk(root,followlinks=False):
            subdirs[:]=sorted(name for name in subdirs if name not in EXCLUDED_DIRS)
            for filename in sorted(files):
                path=Path(dir_name)/filename
                if path.suffix.lower() not in EXTENSIONS or not path.is_file():
                    continue
                canonical=str(path.resolve())
                if canonical in seen:
                    continue
                seen.add(canonical)
                records.append(file_record(path))
    records.sort(key=lambda r:r["path"])
    return records,roots_found

def audit(conditional,unconditional,roots):
    reference={}
    for name,p in [
        ("conditional",Path(conditional)),
        ("unconditional",Path(unconditional))
    ]:
        if not p.is_file():
            raise FileNotFoundError(f"Actual inference checkpoint missing: {p}")
        reference[name]=file_record(p,hash_file=True)
    archive_files,roots_found=list_checkpoints(roots)
    target_sizes={k:v["bytes"] for k,v in reference.items()}
    matching_sizes=set(target_sizes.values())
    candidate_paths={x["path"] for x in archive_files
                     if x["bytes"] in matching_sizes}
    archive_hashes={p:streaming_sha256(Path(p)) for p in sorted(candidate_paths)}
    for item in archive_files:
        if item["path"] in archive_hashes:
            item["sha256"]=archive_hashes[item["path"]]
    matches={}
    same_size={}
    for name,ref in reference.items():
        matches[name]=[x["path"] for x in archive_files
                       if x.get("sha256")==ref["sha256"]]
        same_size[name]=[x["path"] for x in archive_files
                         if x["bytes"]==ref["bytes"]]
    bests=[x for x in archive_files if Path(x["path"]).name=="aurora_diff_best.pth"]
    return {
        "reference_inference_checkpoints":reference,
        "archive_roots":roots_found,
        "n_archive_checkpoint_files":len(archive_files),
        "n_archive_best_named_checkpoints":len(bests),
        "historical_best_named_checkpoints":bests,
        "matching_size_archive_paths":same_size,
        "byte_identical_archive_paths":matches,
        "n_archive_files_sha256_read":len(candidate_paths),
        "historical_checkpoint_file_inventory":archive_files,
        "binary_identity_confirmed_for_conditional":bool(matches["conditional"]),
        "binary_identity_confirmed_for_unconditional":bool(matches["unconditional"]),
        "training_lineage_verified":False,
        "interpretation":[
            "Matching full SHA-256 establishes byte-identical serialized weights file, but not who trained it or exactly which source/data were used.",
            "Different file bytes and sizes do not rule out numerically identical model tensors saved in a different archive format or wrapped with optimizer/metadata.",
            "Metadata timestamps may change when copying, so file modification time is not an original training date.",
            "If no binary match is found across the searched archive roots, original training checkpoint lineage remains not independently verified and must be disclosed.",
            "Do not mutate historical checkpoints, train on 2006, or upload model weight files."
        ]
    }

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--conditional",type=Path,default=DEFAULT_CONDITIONAL)
    p.add_argument("--unconditional",type=Path,default=DEFAULT_UNCONDITIONAL)
    p.add_argument("--roots",type=Path,nargs="+",default=DEFAULT_ARCHIVES)
    p.add_argument("--output",type=Path,
        default=HERE/"historical_checkpoint_match_audit.json")
    args=p.parse_args()
    result=audit(args.conditional,args.unconditional,args.roots)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(
        json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    summary={
        "n_archive_checkpoint_files":result["n_archive_checkpoint_files"],
        "n_archive_best_named_checkpoints":result["n_archive_best_named_checkpoints"],
        "n_archive_files_hashed":result["n_archive_files_sha256_read"],
        "matching_size_archive_paths":result["matching_size_archive_paths"],
        "byte_identical_archive_paths":result["byte_identical_archive_paths"],
        "training_lineage_verified":False,
        "output":str(args.output)
    }
    print(json.dumps(summary,indent=2,ensure_ascii=False))
if __name__=="__main__":
    main()
