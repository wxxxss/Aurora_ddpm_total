#!/usr/bin/env python3
"""Collect event context for Reviewer 2 Comments 2 and 4.

Local data are authoritative for model conditioning and 5-min OMNI context.
Optional CDAWeb queries add 3-hour Kp and Polar definitive ephemeris. Remote
failures are reported but never replaced with guessed values.
"""
from __future__ import annotations

import argparse, json, socket
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_HRO2 = Path('/home/docker/data/ro-share/omni/omni_cdaweb/hro2_5min')
DEFAULT_POLAR = Path('/home/docker/data/private/AuroraData/real_aurora_data_polar/1996/resampled_5min_1996_0405.npy')
DEFAULT_2005 = Path('/home/docker/data/private/AuroraData/omni_real_data/omni_5min/2005/omni_20050101_5min.npy')
RANGES = {
    'Bx': (-100,100), 'By': (-100,100), 'Bz': (-100,100),
    'V': (100,2500), 'P': (0,100), 'AE': (0,5000), 'SYM_H': (-1000,1000),
}
EVENTS = [
    dict(event='Figure 4', instrument='Polar/UVI', start='1996-04-03T02:15:00', end='1996-04-03T02:15:00', orbit=None, polar=True),
    dict(event='Figure 5', instrument='Polar/UVI', start='1996-04-01T08:40:00', end='1996-04-01T08:40:00', orbit=None, polar=True),
    dict(event='Figure 6', instrument='DMSP F16/SSUSI', start='2005-01-01T14:44:34', end='2005-01-01T16:26:25', orbit='06228', polar=False),
    dict(event='Figure 7', instrument='DMSP F16/SSUSI', start='2005-01-04T02:12:15', end='2005-01-04T03:54:07', orbit='06263', polar=False),
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hro2-root', type=Path, default=DEFAULT_HRO2)
    p.add_argument('--polar-npy', type=Path, default=DEFAULT_POLAR)
    p.add_argument('--omni-2005-npy', type=Path, default=DEFAULT_2005)
    p.add_argument('--output-dir', type=Path, default=HERE/'event_context_results')
    p.add_argument('--skip-remote', action='store_true')
    return p.parse_args()


def _clean(values, field):
    x = np.asarray(values, float).reshape(-1).copy()
    lo, hi = RANGES[field]
    x[(~np.isfinite(x)) | (x < lo) | (x > hi)] = np.nan
    return x


def summarize_omni_window(df, start, end):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    t = pd.to_datetime(df['utc'])
    w = df.loc[(t >= start) & (t <= end)]
    out = dict(window_start=start.isoformat(), window_end=end.isoformat(), n_samples=int(len(w)))
    for f in ('Bx','By','Bz','V','P'):
        if f in w:
            x = _clean(w[f], f); v = x[np.isfinite(x)]
            out[f'{f}_mean'] = float(v.mean()) if len(v) else None
            out[f'{f}_min'] = float(v.min()) if len(v) else None
            out[f'{f}_max'] = float(v.max()) if len(v) else None
            out[f'{f}_n'] = int(len(v))
    for f, prefix in (('AE','AE'), ('SYM_H','SYM_H')):
        if f in w:
            x = _clean(w[f], f); v = x[np.isfinite(x)]
            out[f'{prefix}_mean'] = float(v.mean()) if len(v) else None
            out[f'{prefix}_n'] = int(len(v))
            if prefix == 'AE': out['AE_max'] = float(v.max()) if len(v) else None
            else: out['SYM_H_min'] = float(v.min()) if len(v) else None
    return out


def find_polar_crossing(times, magnetic_latitude, event_time, threshold=60.0):
    t = pd.DatetimeIndex(pd.to_datetime(times)); m = np.asarray(magnetic_latitude, float).reshape(-1)
    if len(t) != len(m) or not len(t): raise ValueError('Inconsistent Polar ephemeris arrays.')
    good = np.isfinite(m) & (np.abs(m) >= threshold) & (np.abs(m) <= 90.5)
    segs=[]; i=0
    while i < len(good):
        if not good[i]: i += 1; continue
        j=i
        while j+1 < len(good) and good[j+1]: j += 1
        segs.append((i,j)); i=j+1
    if not segs: return dict(crossing_start=None, crossing_end=None, contains_event=False)
    event = pd.Timestamp(event_time)
    inside = [s for s in segs if t[s[0]] <= event <= t[s[1]]]
    if inside: s=inside[0]; contains=True
    else:
        s=min(segs, key=lambda q: abs(((t[q[0]]+(t[q[1]]-t[q[0]])/2)-event).total_seconds())); contains=False
    return dict(crossing_start=pd.Timestamp(t[s[0]]), crossing_end=pd.Timestamp(t[s[1]]), contains_event=contains)


def find_monthly_hro2(root, when):
    when = pd.Timestamp(when); d=Path(root)/str(when.year)
    pats=[f'omni_hro2_5min_{when.year}{when.month:02d}01_v*.cdf', f'*{when.year}{when.month:02d}01*.cdf']
    for pat in pats:
        m=sorted(d.glob(pat))
        if m: return m[-1]
    raise FileNotFoundError(f'No HRO2 5-min CDF for {when:%Y-%m} under {d}')


def load_hro2_context(path):
    try: import cdflib
    except ImportError as e: raise RuntimeError('cdflib is required.') from e
    cdf=cdflib.CDF(str(path))
    try:
        info=cdf.cdf_info(); avail=set(info.rVariables)|set(info.zVariables)
        candidates={
            'Bx':['BX_GSE'], 'By':['BY_GSM'], 'Bz':['BZ_GSM'], 'P':['Pressure'],
            'AE':['AE_INDEX','AE'], 'SYM_H':['SYM_H','SYM_H_INDEX'], 'V':['flow_speed','V']}
        mp={k:next((x for x in v if x in avail),None) for k,v in candidates.items()}
        if 'Epoch' not in avail or any(mp[x] is None for x in ('Bx','By','Bz','V','P')):
            raise KeyError(f'Unable to resolve required HRO2 variables; available={sorted(avail)[:80]}')
        data={'utc':pd.to_datetime(cdflib.cdfepoch.to_datetime(cdf.varget('Epoch')))}
        for k,v in mp.items():
            if v is not None: data[k]=np.asarray(cdf.varget(v)).squeeze().reshape(-1).astype(float)
    finally:
        try: cdf.close()
        except Exception: pass
    return pd.DataFrame(data).sort_values('utc').reset_index(drop=True), mp


def _structured_df(path):
    arr=np.load(path, allow_pickle=True)
    if not arr.dtype.names: raise ValueError(f'Expected structured NPY: {path}')
    data={}
    for f in ('utc','Bx','By','Bz','V','P'):
        if f not in arr.dtype.names: continue
        if f == 'utc': data[f]=pd.to_datetime(arr[f])
        else: data[f]=np.asarray(arr[f]).reshape(len(arr),-1)[:,0].astype(float)
    return pd.DataFrame(data)


def nearest_condition_from_polar_npy(path, event_time):
    df=_structured_df(path); target=pd.Timestamp(event_time)
    if any(f not in df for f in ('utc','Bx','By','Bz','V','P')): raise KeyError('Polar NPY missing conditioning fields.')
    delta=np.abs((pd.to_datetime(df.utc)-target).dt.total_seconds().to_numpy()); i=int(np.nanargmin(delta)); r=df.iloc[i]
    return dict(matched_time=pd.Timestamp(r.utc).isoformat(), offset_s=float(delta[i]), **{f:float(r[f]) for f in ('Bx','By','Bz','V','P')})


def interval_condition_from_npy(path, start, end):
    s=summarize_omni_window(_structured_df(path), start, end)
    return {f:s.get(f'{f}_mean') for f in ('Bx','By','Bz','V','P')}


def _cdas_array(data,name):
    x=data[name]; x=x.values if hasattr(x,'values') else x
    return np.asarray(x).squeeze()


def _cdas_time(data):
    for n in ('Epoch','EPOCH','EPOCH_1800','epoch','time'):
        try: return pd.DatetimeIndex(pd.to_datetime(_cdas_array(data,n)))
        except Exception: pass
    for n in getattr(data,'coords',{}):
        try:
            t=pd.DatetimeIndex(pd.to_datetime(np.asarray(data.coords[n].values).squeeze()))
            if len(t): return t
        except Exception: pass
    raise KeyError('No CDAWeb epoch coordinate found.')


def fetch_kp_cdaweb(start,end):
    try: from cdasws import CdasWs
    except ImportError as e: raise RuntimeError('cdasws is not installed.') from e
    socket.setdefaulttimeout(25); start=pd.Timestamp(start); end=pd.Timestamp(end)
    _,data=CdasWs().get_data('OMNI2_H0_MRG1HR',['KP1800'],(start-pd.Timedelta(hours=2)).to_pydatetime(),(end+pd.Timedelta(hours=2)).to_pydatetime())
    if data is None: raise RuntimeError('No OMNI2 Kp returned.')
    t=_cdas_time(data); k=np.asarray(_cdas_array(data,'KP1800'),float).reshape(-1)
    ok=np.isfinite(k)&(k>=0)&(k<=90); t=t[ok]; k=k[ok]
    if not len(k): return dict(Kp=None,matched_time=None)
    inside=(t>=start)&(t<=end)
    if np.any(inside): ids=np.where(inside)[0]; i=int(ids[np.argmax(k[ids])])
    else: i=int(np.argmin(np.abs((t-(start+(end-start)/2)).total_seconds())))
    return dict(Kp=float(k[i]/10.0), matched_time=pd.Timestamp(t[i]).isoformat())


def fetch_polar_ephemeris(event_time):
    try: from cdasws import CdasWs
    except ImportError as e: raise RuntimeError('cdasws is not installed.') from e
    socket.setdefaulttimeout(25); e=pd.Timestamp(event_time); q0=e-pd.Timedelta(hours=6); q1=e+pd.Timedelta(hours=6)
    _,data=CdasWs().get_data('PO_OR_DEF',['MAG_LATITUDE','ORB_REV_NUM','EDMLT_TIME'],q0.to_pydatetime(),q1.to_pydatetime())
    if data is None: raise RuntimeError('No PO_OR_DEF data returned.')
    t=_cdas_time(data); m=np.asarray(_cdas_array(data,'MAG_LATITUDE'),float).reshape(-1); o=np.asarray(_cdas_array(data,'ORB_REV_NUM')).reshape(-1)
    i=int(np.argmin(np.abs((t-e).total_seconds()))); c=find_polar_crossing(t,m,e)
    out=dict(nearest_time=pd.Timestamp(t[i]).isoformat(), orbit=str(int(round(float(o[i])))), mag_latitude_deg=float(m[i]),
             crossing_start=c['crossing_start'].isoformat() if c['crossing_start'] is not None else None,
             crossing_end=c['crossing_end'].isoformat() if c['crossing_end'] is not None else None,
             crossing_contains_event=bool(c['contains_event']))
    try: out['edmlt_raw_nearest']=float(np.asarray(_cdas_array(data,'EDMLT_TIME')).reshape(-1)[i])
    except Exception: out['edmlt_raw_nearest']=None
    return out


def _fmt(x,n=1):
    if x is None: return '--'
    try:
        x=float(x); return '--' if not np.isfinite(x) else f'{x:.{n}f}'
    except Exception: return str(x)


def latex_escape(x):
    s=str(x)
    for a,b in (('&',r'\&'),('%',r'\%'),('_',r'\_'),('#',r'\#')): s=s.replace(a,b)
    return s


def build_latex_table(rows: Sequence[Dict[str,Any]]):
    L=[r'\begin{table*}[htbp]',r'\centering',
       r'\caption{Geophysical context of the real-observation reconstruction cases. Solar-wind and IMF values are the conditioning values used for each case; AE and SYM-H summarize the corresponding context window. Kp is obtained from the 3-hour OMNI2 index when available.}',
       r'\label{tab:event_context}',r'\scriptsize',r'\resizebox{\textwidth}{!}{%',r'\begin{tabular}{lllllrrrrrrrr}',r'\hline',
       r'Event & Instrument & Time / interval (UT) & Orbit & Polar-region crossing (UT) & $B_x$ & $B_y$ & $B_z$ & $V_{\mathrm{sw}}$ & $P_{\mathrm{dyn}}$ & Kp & AE$_{\max}$ & SYM-H$_{\min}$ \\',
       r' & & & & & \multicolumn{3}{c}{(nT)} & (km s$^{-1}$) & (nPa) & & (nT) & (nT) \\',r'\hline']
    for r in rows:
        c=[r.get('event',''),r.get('instrument',''),r.get('interval',''),r.get('orbit') or '--',r.get('polar_crossing') or '--',
           _fmt(r.get('Bx')),_fmt(r.get('By')),_fmt(r.get('Bz')),_fmt(r.get('V'),0),_fmt(r.get('P'),2),_fmt(r.get('Kp')),_fmt(r.get('AE_max'),0),_fmt(r.get('SYM_H_min'),0)]
        L.append(' & '.join(latex_escape(v) for v in c)+r' \\')
    L += [r'\hline',r'\end{tabular}%',r'}',r'\end{table*}']
    return '\n'.join(L)+'\n'


def _interval_text(e):
    a,b=pd.Timestamp(e['start']),pd.Timestamp(e['end'])
    return a.strftime('%Y-%m-%d %H:%M') if a==b else f'{a:%Y-%m-%d %H:%M:%S}--{b:%H:%M:%S}'


def collect_event(e,args,cache):
    start,end=pd.Timestamp(e['start']),pd.Timestamp(e['end']); ctx_start=start-pd.Timedelta(minutes=30) if e['polar'] else start
    errs={}; ctx={}; source=None; key=(start.year,start.month)
    try:
        source=find_monthly_hro2(args.hro2_root,start)
        if key not in cache: cache[key],_=load_hro2_context(source)
        ctx=summarize_omni_window(cache[key],ctx_start,end)
    except Exception as ex: errs['context']=f'{type(ex).__name__}: {ex}'
    try:
        cond=nearest_condition_from_polar_npy(args.polar_npy,start) if e['polar'] else interval_condition_from_npy(args.omni_2005_npy,start,end)
    except Exception as ex:
        errs['conditioning']=f'{type(ex).__name__}: {ex}'; cond={f:ctx.get(f'{f}_mean') for f in ('Bx','By','Bz','V','P')}
    kp=dict(Kp=None,matched_time=None); eph={}
    if not args.skip_remote:
        try: kp=fetch_kp_cdaweb(start,end)
        except Exception as ex: errs['kp']=f'{type(ex).__name__}: {ex}'
        if e['polar']:
            try: eph=fetch_polar_ephemeris(start)
            except Exception as ex: errs['ephemeris']=f'{type(ex).__name__}: {ex}'
    orbit=e['orbit'] or eph.get('orbit')
    if eph.get('crossing_start'):
        cross=f"{pd.Timestamp(eph['crossing_start']):%H:%M}--{pd.Timestamp(eph['crossing_end']):%H:%M}"
    elif not e['polar']: cross=f'{start:%H:%M}--{end:%H:%M}'
    else: cross=None
    r=dict(event=e['event'],instrument=e['instrument'],interval=_interval_text(e),start=start.isoformat(),end=end.isoformat(),
           context_start=ctx_start.isoformat(),context_end=end.isoformat(),orbit=orbit,polar_crossing=cross,source_hro2_cdf=str(source) if source else None,
           condition_source=str(args.polar_npy if e['polar'] else args.omni_2005_npy),context=ctx,ephemeris=eph,remote_kp=kp,errors=errs)
    for f in ('Bx','By','Bz','V','P'): r[f]=cond.get(f)
    r.update(Kp=kp.get('Kp'),AE_max=ctx.get('AE_max'),AE_mean=ctx.get('AE_mean'),SYM_H_min=ctx.get('SYM_H_min'),SYM_H_mean=ctx.get('SYM_H_mean'))
    if 'matched_time' in cond: r['condition_matched_time']=cond['matched_time']; r['condition_offset_s']=cond.get('offset_s')
    return r


def main():
    args=parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    print('='*92+'\nR2.2 + R2.4 EVENT GEOPHYSICAL CONTEXT\n'+'='*92)
    print(f'OMNI HRO2 root:  {args.hro2_root}\nPolar model NPY: {args.polar_npy}\n2005 model NPY:  {args.omni_2005_npy}\nRemote CDAWeb:   {"disabled" if args.skip_remote else "enabled (non-fatal)"}')
    rows=[]; cache={}
    for e in EVENTS:
        r=collect_event(e,args,cache); rows.append(r); print('\n'+'-'*92)
        print(f"{r['event']} | {r['instrument']} | {r['interval']} UT")
        print(f"Orbit:            {r['orbit'] or 'unavailable'}\nPolar crossing:   {r['polar_crossing'] or 'unavailable'}")
        print(f"Condition vector: Bx={_fmt(r['Bx'])} nT, By={_fmt(r['By'])} nT, Bz={_fmt(r['Bz'])} nT, V={_fmt(r['V'],0)} km/s, Pdyn={_fmt(r['P'],2)} nPa")
        print(f"Geomagnetic:      Kp={_fmt(r['Kp'])}, AEmax={_fmt(r['AE_max'],0)} nT, SYM-Hmin={_fmt(r['SYM_H_min'],0)} nT")
        c=r['context']
        if c: print(f"Context mean:     Bz={_fmt(c.get('Bz_mean'))} nT, AE={_fmt(c.get('AE_mean'),0)} nT, SYM-H={_fmt(c.get('SYM_H_mean'),0)} nT, N={c.get('n_samples',0)}")
        for k,v in r['errors'].items(): print(f'NOTE {k}: {v}')
    j=args.output_dir/'event_context.json'; c=args.output_dir/'event_context.csv'; t=args.output_dir/'event_context_table.tex'
    j.write_text(json.dumps(rows,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
    cols=['event','instrument','interval','orbit','polar_crossing','Bx','By','Bz','V','P','Kp','AE_max','AE_mean','SYM_H_min','SYM_H_mean','condition_matched_time','condition_offset_s','source_hro2_cdf','condition_source']
    pd.DataFrame(rows).reindex(columns=cols).to_csv(c,index=False); t.write_text(build_latex_table(rows),encoding='utf-8')
    print('\n'+'='*92+'\nDONE\n'+'='*92+f'\nJSON:  {j}\nCSV:   {c}\nLaTeX: {t}\n\nPlease send the complete console output and event_context.csv.')


if __name__ == '__main__': main()
