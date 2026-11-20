#!/usr/bin/env python3
"""Static review of mask-producing code paths across manuscript versions.
Run from repository root:
  python paper_modif/Second_revision/R2-5_R3-2_R3-11/audit_mask_provenance.py
No scientific conclusions are inferred from a script existing in the repository:
the checkpoint and input files used to render each paper figure must be verified.
"""
import json
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
PATHS={
    "legacy_polar":"repaired_polar.py",
    "legacy_ssusi":"repaired_ssusi.py",
    "round1_ssusi":"paper_modif/First_revision/R2-1_R2-3/rerun_ssusi_reconstruction.py",
    "round1_polar_figures":"paper_modif/First_revision/R2-2_R2-4/redraw_polar_figures.py",
    "sampling":"models/ddpm.py"
}
PATTERN=re.compile(r"mask|need_repair|real_close_to_zero|sim_has_value|OBS_SUPPORT_THRESHOLD|np\\.where\\(obs_mask",re.I)
def main():
    record={}
    for tag,path in PATHS.items():
        location=ROOT/path
        if not location.is_file():
            record[tag]={"path":path,"missing_file":True}
            continue
        code=location.read_text(encoding="utf-8")
        hits=[{"line":i,"text":line.strip()} for i,line in
              enumerate(code.splitlines(),1) if PATTERN.search(line) and
              not line.strip().startswith("#")]
        record[tag]={"path":path,"matches":hits[:80]}
    record["interpretation"]=(
      "Legacy Polar uses an intensity threshold (polar < 1); legacy SSUSI "
      "uses an intensity and OVATION support criterion; first-revision SSUSI "
      "uses the direct swath observation mask. These scripts must NOT be "
      "assumed to describe the same data version or the same Figure 4-7 runs."
    )
    out=Path(__file__).resolve().parent/"mask_provenance.json"
    out.write_text(json.dumps(record,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps(record,indent=2,ensure_ascii=False))
    print("Saved",out)

if __name__=="__main__":
    main()
