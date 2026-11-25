#!/usr/bin/env python3
"""Only annotate existing Figure 4-7 PNGs. Never regenerate scientific data.

Four audited source cases are read from the original 2005/1996
event_context_results/event_context.csv. Figure 4/5 use the exactly
time-matched 5-minute 1996 OMNI row; Figure 6/7 use 2005 SSUSI
EDR-interval OMNI means independently verified in the second revision.
The 2005 PNG defaults are specifically the TIME-CORRECTED output, NOT
the earlier midpoint-indexed first-revision images.

Run after git pull from Aurora_DDPM_total repo root:
  python paper_modif/Second_revision/R2-4_R2-8_R3-1/annotate_figures_4_7_solar_wind.py
No NPU, checkpoint deserialization, model inference or source map changes.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
POLAR=REPO/"paper_modif/First_revision/R2-2_R2-4"
SSUSI=REPO/"paper_modif/Second_revision/R2-7_R3-10_R3-12_R3-13"
DEFAULT_CONTEXT=POLAR/"event_context_results/event_context.csv"
DEFAULT_SSUSI_AUDIT=SSUSI/"timecorrected_figures_and_psd_verified.json"
SOURCES={
 "Figure4":POLAR/"polar_redraw_results/Figure4_Polar_UVI_manuscript.png",
 "Figure5":POLAR/"polar_redraw_results/Figure5_Polar_UVI_manuscript.png",
 "Figure6":SSUSI/"center_aligned_reconstruction_timecorrected/Figure6/Figure6_comparison_vmax5.png",
 "Figure7":SSUSI/"center_aligned_reconstruction_timecorrected/Figure7/Figure7_comparison_vmax5.png",
}
EXPECTED_INTERVALS={
 "Figure4":"1996-04-03 02:15",
 "Figure5":"1996-04-01 08:40",
 "Figure6":"2005-01-01 14:44:34--16:26:25",
 "Figure7":"2005-01-04 02:12:15--03:54:07",
}
FIELDS=("Bz","V","P")

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda:fh.read(8<<20),b""):h.update(block)
    return h.hexdigest()

def source_context(path):
    with Path(path).open("r",newline="",encoding="utf-8-sig") as fh:
        src={r["event"].replace(" ",""):r for r in csv.DictReader(fh)}
    for fig,interval in EXPECTED_INTERVALS.items():
        if fig not in src or src[fig]["interval"]!=interval:
            raise ValueError(f"Missing or date-mismatched original OMNI event context: {fig}")
        row=src[fig]
        for field in FIELDS:
            if not math.isfinite(float(row[field])):
                raise ValueError(f"{fig}: invalid source {field}")
        if fig in ("Figure4","Figure5"):
            if (row["condition_matched_time"][:16].replace("T"," ")!=interval or
                abs(float(row["condition_offset_s"]))>1e-6 or
                "1996" not in row["condition_source"]):
                raise ValueError(f"{fig}: expected exactly matched 1996 Polar 5-minute observation")
        elif "2005" not in row["condition_source"]:
            raise ValueError(f"{fig}: wrong original 2005 source")
    return src

def verify_ssusi(ctx,path):
    audited=json.loads(Path(path).read_text(encoding="utf-8"))
    for fig in ("Figure6","Figure7"):
        case=audited["time_alignment_validation"][fig]
        if (case["omni_conditioning"]!="EDR acquisition interval mean" or
            case["ovation_baseline"]!="1-minute map average over EDR acquisition"):
            raise ValueError(f"{fig}: source is not time-corrected OMNI and OVATION EDR means")
        row=ctx[fig]
        if (case["acquisition_start"].replace("T"," ")+"--"+
            case["acquisition_end"].split("T")[-1])!=row["interval"]:
            raise ValueError(f"{fig}: mismatched SSUSI EDR acquisition interval")
        for field in FIELDS:
            if abs(float(row[field])-float(case["solar_5vector"][field]))>0.003:
                raise ValueError(f"{fig}: audited {field} disagrees with original event_context.csv")
    return audited

def font(size,bold=False):
    paths=("DejaVuSans-Bold.ttf","DejaVuSans.ttf") if bold else ("DejaVuSans.ttf",)
    for name in paths:
        try:return ImageFont.truetype(name,size)
        except OSError:pass
    return ImageFont.load_default()

def label(fig,source,destination,row):
    with Image.open(source) as src:
        original=src.convert("RGB")
    width,height=original.size
    if width<900 or height<400:
        raise ValueError(f"{fig}: source resolution unexpectedly small: {width}x{height}")
    band=max(92,round(width*0.052))
    image=Image.new("RGB",(width,height+band),"black")
    image.paste(original,(0,0))
    draw=ImageDraw.Draw(image)
    draw.line([(round(width*.07),height+2),(round(width*.93),height+2)],
              fill="#6e828a",width=max(1,round(width/1000)))
    headline=(f"IMF Bz {float(row['Bz']):+.2f} nT     Solar wind speed "
              f"{float(row['V']):.0f} km/s     Pdyn {float(row['P']):.2f} nPa")
    if fig in ("Figure4","Figure5"):
        subtitle=f"{row['interval']} UT  |  exact-time matched 5-minute OMNI measurement"
        basis="exact-UTC 5-minute OMNI measurement"
    else:
        subtitle=(row["interval"].replace("--"," to ")+" UT  |  5-minute OMNI acquisition-interval mean")
        basis="acquisition-interval mean of 5-minute OMNI"
    for text,ft,y,color in (
        (headline,font(round(width*.016),True),height+round(band*.13),"white"),
        (subtitle,font(round(width*.011)),height+round(band*.53),"#d8e4e8")
    ):
        box=draw.textbbox((0,0),text,font=ft)
        if box[2]>width*.96:
            raise ValueError(f"{fig}: text width overflows source image")
        draw.text(((width-box[2])/2,y),text,font=ft,fill=color)
    destination.parent.mkdir(parents=True,exist_ok=True)
    image.save(destination,format="PNG",optimize=True)
    with Image.open(destination) as rendered:
        if rendered.crop((0,0,width,height)).tobytes()!=original.tobytes():
            raise AssertionError(f"{fig}: source scientific figure panels changed")
    return {
        "figure":fig,"source_png":str(source),
        "source_sha256":sha256(source),"labeled_png":str(destination),
        "labeled_sha256":sha256(destination),
        "original_pixels_preserved":True,
        "original_dimensions":[width,height],
        "labeled_dimensions":[width,height+band],
        "IMF_Bz_nT":float(row["Bz"]),
        "solar_speed_km_s":float(row["V"]),
        "Pdyn_nPa":float(row["P"]),
        "utc":row["interval"],"omni_time_basis":basis,
    }

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--context",type=Path,default=DEFAULT_CONTEXT)
    p.add_argument("--timecorrected-audit",type=Path,default=DEFAULT_SSUSI_AUDIT)
    for fig,default in SOURCES.items():
        p.add_argument("--"+fig.lower(),type=Path,default=default)
    p.add_argument("--output-root",type=Path,default=HERE/"figure4_7_solar_wind_labeled")
    p.add_argument("--allow-missing",action="store_true",
                   help="Partial staging only; never marks all of R2-6 complete")
    args=p.parse_args()
    ctx=source_context(args.context)
    verify_ssusi(ctx,args.timecorrected_audit)
    results=[]
    missing=[]
    for fig in EXPECTED_INTERVALS:
        path=getattr(args,fig.lower())
        if not path.is_file():
            if not args.allow_missing:
                raise FileNotFoundError(
                    f"{fig}: missing original PNG {path}; do not use obsolete "
                    "midpoint-indexed 2005 reconstruction figures.")
            missing.append(fig)
            continue
        out=args.output_root/(fig+"_solar_wind_labeled.png")
        results.append(label(fig,path,out,ctx[fig]))
    report={
       "ready_for_R2_6":len(results)==4,
       "n_annotated_figures":len(results),
       "missing":missing,
       "source_context_file":str(args.context),
       "source_context_sha256":sha256(args.context),
       "timecorrected_ssusi_audit_sha256":sha256(args.timecorrected_audit),
       "source_pngs_original_pixel_data_untouched":True,
       "figures":results,
       "caveat":"This only labels an existing scientific image; no reconstruction or solar-wind inference is repeated."
    }
    args.output_root.mkdir(parents=True,exist_ok=True)
    (args.output_root/"figure4_7_omni_label_audit.json").write_text(
        json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))

if __name__=="__main__":
    main()
