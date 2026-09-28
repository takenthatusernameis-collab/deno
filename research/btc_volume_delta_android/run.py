
#!/usr/bin/env python3
from __future__ import annotations
import json, math, os, re, shutil
from pathlib import Path
import numpy as np
import pandas as pd
from huggingface_hub import HfApi, snapshot_download
from scipy import stats
import vectorbt as vbt

HF_REPO="linxy/USDT-M_Perpetual_Futures"
START=pd.Timestamp("2024-01-01T00:00:00Z")
END=pd.Timestamp("2026-09-23T00:00:00Z")
HOLDOUT=pd.Timestamp("2025-10-01T00:00:00Z")
FREQ="4h"
HORIZONS=(1,4,6,12)
ZWIN=180
LIQWIN=180
FEE=0.00055
SLIP=0.00050
GATE=100
CACHE=Path("research/btc_volume_delta_android/cache")
ART=Path("research/btc_volume_delta_android/artifacts")
MAJORS={"BTC","ETH","SOL","BNB","XRP","ADA","DOGE"}
STABLES={"USDT","USDC","BUSD","FDUSD","TUSD","USDP","DAI","PYUSD","USDE","USDD","USD1","EUR","TRY","BRL","GBP","AUD"}
WRAPPED={"WBTC","WETH","STETH","WSTETH","WBETH","CBETH","RETH","SFRXETH","MATICX","BETH"}
LEV=re.compile(r"(?:UP|DOWN|BULL|BEAR)$")

def is_excluded(s):
    b=s[:-4] if s.endswith("USDT") else s
    return not s.endswith("USDT") or b in MAJORS or b in STABLES or b in WRAPPED or LEV.search(b) is not None

def prior_z(s,w=ZWIN):
    mu=s.shift(1).rolling(w,min_periods=w).mean()
    sd=s.shift(1).rolling(w,min_periods=w).std(ddof=1)
    return (s-mu)/sd.replace(0.0,np.nan)

def prior_pct(s,w=ZWIN):
    def rank_last(a):
        x=a[-1]
        p=a[:-1]
        p=p[np.isfinite(p)]
        if not np.isfinite(x) or len(p)==0: return np.nan
        return ((p<x).sum()+0.5*(p==x).sum())/len(p)
    return s.rolling(w+1,min_periods=w+1).apply(rank_last,raw=True)

def fwd_ret(op,cl,h):
    return cl.shift(-h).div(op.shift(-1))-1.0

def masks(close,vol,low_tail=0.30,high_tail=0.30):
    med=close.mul(vol).shift(1).rolling(LIQWIN,min_periods=LIQWIN).median()
    elig=pd.DataFrame(True,index=med.index,columns=med.columns)
    for s in med.columns:
        if is_excluded(s): elig[s]=False
    valid=med.notna()&elig
    r=med.where(valid).rank(axis=1,pct=True,method="average")
    blue=valid&(r>0.80)
    nonblue=valid&~blue
    rr=med.where(nonblue).rank(axis=1,pct=True,method="average")
    low=nonblue&(rr<=low_tail)
    high=nonblue&(rr>=1.0-high_tail)
    return med,blue,nonblue,low,high

def cs_mean(fwd,mask,start=HOLDOUT):
    return fwd.where(mask).loc[start:].mean(axis=1).dropna()

def hac_mean_test(raw):
    raw=np.asarray(raw,float)
    raw=raw[np.isfinite(raw)]
    n=len(raw)
    if n<3: return np.nan,np.nan
    xc=raw-raw.mean()
    lag=min(n-1,max(1,int(4*(n/100.0)**(2.0/9.0))))
    lrv=float(np.dot(xc,xc)/n)
    for k in range(1,lag+1):
        g=float(np.dot(xc[k:],xc[:-k])/n)
        lrv += 2*(1-k/(lag+1))*g
    se=math.sqrt(max(lrv/n,0.0))
    t=float(raw.mean()/se) if se>0 else np.nan
    p=float(2*stats.t.sf(abs(t),df=max(n-1,1))) if np.isfinite(t) else np.nan
    return t,p

def predictive(feature,op,cl,low,high,pos_cut,neg_cut,name):
    rows=[]
    for h in HORIZONS:
        fw=fwd_ret(op,cl,h)
        for lname,mask in (("low_liquidity",low),("high_liquidity",high)):
            d=cs_mean(fw,mask)
            sig=feature.reindex(d.index)
            for side,cond in (("positive",sig>=pos_cut),("negative",sig<=neg_cut)):
                y=d.where(cond).dropna()
                t,p=hac_mean_test(y.values)
                pooled=fw.where(mask).loc[y.index]
                rows.append({
                    "control":name,"horizon_bars":h,"liquidity":lname,"signal_side":side,
                    "n_symbol_day":int(pooled.count().sum()),"n_dates":len(y),
                    "distinct_symbols":int(pooled.notna().any(axis=0).sum()),
                    "mean_forward_return":float(y.mean()) if len(y) else np.nan,
                    "median_forward_return":float(y.median()) if len(y) else np.nan,
                    "std_forward_return":float(y.std(ddof=1)) if len(y)>1 else np.nan,
                    "hit_rate":float((y>0).mean()) if len(y) else np.nan,
                    "hac_t":t,"hac_p":p})
            pos=d.where(sig>=pos_cut).dropna()
            neg=d.where(sig<=neg_cut).dropna()
            idx=pos.index.union(neg.index)
            pooled=fw.where(mask).loc[idx]
            rows.append({
                "control":name,"horizon_bars":h,"liquidity":lname,
                "signal_side":"positive_minus_negative","n_symbol_day":int(pooled.count().sum()),
                "n_dates":int(len(pos)+len(neg)),
                "distinct_symbols":int(pooled.notna().any(axis=0).sum()),
                "mean_forward_return":float(pos.mean()-neg.mean()) if len(pos) and len(neg) else np.nan,
                "median_forward_return":np.nan,"std_forward_return":np.nan,"hit_rate":np.nan,
                "hac_t":np.nan,"hac_p":np.nan})
    return pd.DataFrame(rows)

def per_symbol(feature,op,cl,low):
    rows=[]
    for h in HORIZONS:
        fw=fwd_ret(op,cl,h).loc[HOLDOUT:]
        for s in fw.columns:
            y=fw[s].where(low[s].loc[HOLDOUT:].fillna(False))
            p=y.where(feature>=1).dropna()
            n=y.where(feature<=-1).dropna()
            if len(p) and len(n):
                rows.append({"symbol":s,"horizon_bars":h,"n_positive":len(p),"n_negative":len(n),
                             "positive_mean":float(p.mean()),"negative_mean":float(n.mean()),
                             "spread":float(p.mean()-n.mean())})
    return pd.DataFrame(rows)

def calendar_effects(feature,op,cl,low):
    rows=[]
    for h in HORIZONS:
        d=cs_mean(fwd_ret(op,cl,h),low)
        sig=feature.reindex(d.index)
        q=pd.DataFrame({"ret":d,"sig":sig})
        q["quarter"]=q.index.to_period("Q").astype(str)
        for quarter,part in q.groupby("quarter"):
            p=part.loc[part.sig>=1,"ret"]; n=part.loc[part.sig<=-1,"ret"]
            rows.append({"quarter":quarter,"horizon_bars":h,
                         "n_positive_dates":len(p),"n_negative_dates":len(n),
                         "positive_mean":float(p.mean()) if len(p) else np.nan,
                         "negative_mean":float(n.mean()) if len(n) else np.nan,
                         "spread":float(p.mean()-n.mean()) if len(p) and len(n) else np.nan})
    return pd.DataFrame(rows)

def run_portfolio(close_oos,signal_full,mask_full,cost_mult=1.0):
    sig=signal_full.shift(1)
    raw=np.where(sig.to_numpy()[:,None]>=1,1.0,np.where(sig.to_numpy()[:,None]<=-1,-1.0,0.0))
    raw=pd.DataFrame(raw,index=mask_full.index,columns=mask_full.columns)
    raw=raw.where(mask_full,0.0)
    n=raw.abs().sum(axis=1).replace(0,np.nan)
    weights_full=raw.div(n,axis=0).fillna(0.0)
    weights=weights_full.loc[close_oos.index]
    pf=vbt.Portfolio.from_orders(close=close_oos,size=weights,size_type="targetpercent",
                                 fees=FEE*cost_mult,slippage=SLIP*cost_mult,
                                 init_cash=100000,freq=FREQ,cash_sharing=True,call_seq="auto")
    ret=pd.Series(pf.returns(),index=close_oos.index)
    eq=(1+ret.fillna(0)).cumprod()
    total=float(eq.iloc[-1]-1) if len(eq) else np.nan
    years=len(ret)/(6*365)
    cagr=((1+total)**(1/years)-1) if years>0 and 1+total>0 else np.nan
    st=pf.stats()
    return pf,ret,weights,{
        "total_return_pct":total*100,
        "cagr_pct":cagr*100 if np.isfinite(cagr) else None,
        "sharpe":float(st.get("Sharpe Ratio",np.nan)),
        "sortino":float(st.get("Sortino Ratio",np.nan)),
        "max_drawdown_pct":float(st.get("Max Drawdown [%]",np.nan)),
        "total_trades":float(st.get("Total Trades",np.nan))}

def main():
    ART.mkdir(parents=True,exist_ok=True)
    if CACHE.exists(): shutil.rmtree(CACHE)
    CACHE.mkdir(parents=True,exist_ok=True)
    log=[]
    api=HfApi()
    revision=api.dataset_info(HF_REPO,revision="main").sha
    files=api.list_repo_files(HF_REPO,repo_type="dataset",revision=revision)
    four_h=sorted([p for p in files if re.match(r"^[^/]+/[^/]+_4h\.parquet$",p)])
    log.append({"stage":"discover","revision":revision,"four_h_files":len(four_h)})
    print(json.dumps(log[-1]),flush=True)
    diagnostic_only=os.getenv("BTC_ONLY_DIAGNOSTIC")=="1"
    patterns=["BTCUSDT/BTCUSDT_4h.parquet"] if diagnostic_only else ["*/*_4h.parquet"]
    snapshot=snapshot_download(repo_id=HF_REPO,repo_type="dataset",revision=revision,
                               allow_patterns=patterns,local_dir=str(CACHE),max_workers=8)
    paths=sorted(Path(snapshot).glob("*/*_4h.parquet"))
    log.append({"stage":"download","files_downloaded":len(paths),"diagnostic_only":diagnostic_only})
    print(json.dumps(log[-1]),flush=True)
    if diagnostic_only:
        p=Path(snapshot)/"BTCUSDT/BTCUSDT_4h.parquet"
        raw=pd.read_parquet(p)
        raw_idx=pd.to_datetime(raw["open_time"],utc=True)
        diag={
            "path":str(p),"shape":[int(x) for x in raw.shape],
            "columns":list(raw.columns),"dtypes":{c:str(raw[c].dtype) for c in raw.columns},
            "raw_min_open_time":str(raw_idx.min()),"raw_max_open_time":str(raw_idx.max()),
            "raw_head_open_time":[str(x) for x in raw_idx.head(5)],
            "raw_head_values":raw[["open_time","open","close","volume","taker_buy_volume"]].head(5).astype(str).to_dict("records"),
            "numeric_volume_nonnull":int(pd.to_numeric(raw["volume"],errors="coerce").notna().sum()),
            "numeric_taker_nonnull":int(pd.to_numeric(raw["taker_buy_volume"],errors="coerce").notna().sum()),
            "reindex_nonnull_open":int(pd.to_numeric(raw["open"],errors="coerce").set_axis(raw_idx).reindex(pd.date_range(START,END,freq=FREQ,inclusive="left")).notna().sum()),
        }
        (ART/"btc_file_diagnostic.json").write_text(json.dumps(diag,indent=2))
        print(json.dumps(diag,indent=2),flush=True)
        return

    idx=pd.date_range(START,END,freq=FREQ,inclusive="left")
    opens={}; closes={}; vols={}; buys={}; loaded=[]
    for path in paths:
        s=path.parent.name
        if not s.endswith("USDT") or (s != "BTCUSDT" and is_excluded(s)): continue
        try:
            df=pd.read_parquet(path,columns=["open_time","open","close","volume","taker_buy_volume"])
            df["open_time"]=pd.to_datetime(df["open_time"],utc=True)
            df=df.drop_duplicates("open_time").set_index("open_time").sort_index()
            if df.empty: continue
            opens[s]=pd.to_numeric(df["open"],errors="coerce").reindex(idx)
            closes[s]=pd.to_numeric(df["close"],errors="coerce").reindex(idx)
            vols[s]=pd.to_numeric(df["volume"],errors="coerce").reindex(idx)
            buys[s]=pd.to_numeric(df["taker_buy_volume"],errors="coerce").reindex(idx)
            loaded.append(s)
        except Exception as e:
            log.append({"stage":"load_warning","symbol":s,"error":repr(e)})
    if "BTCUSDT" not in loaded: raise RuntimeError("BTCUSDT unavailable")
    op=pd.DataFrame(opens,index=idx); cl=pd.DataFrame(closes,index=idx)
    vol=pd.DataFrame(vols,index=idx); buy=pd.DataFrame(buys,index=idx)
    log.append({"stage":"load","loaded_symbols":len(loaded)})
    print(json.dumps(log[-1]),flush=True)

    delta=(2*buy["BTCUSDT"]-vol["BTCUSDT"])/vol["BTCUSDT"].replace(0,np.nan)
    z=prior_z(delta); pct=prior_pct(delta)
    alt=[s for s in cl.columns if s!="BTCUSDT"]
    _,_,_,low,high=masks(cl[alt],vol[alt])

    pp=predictive(z,op[alt],cl[alt],low,high,1.0,-1.0,"btc_delta")
    pp_ret=predictive(prior_z(cl["BTCUSDT"].pct_change()),op[alt],cl[alt],low,high,1.0,-1.0,"btc_return_only")
    rng=np.random.default_rng(20260928)
    sh=z.copy(); valid=sh.notna()
    sh.loc[valid]=rng.permutation(sh.loc[valid].to_numpy(copy=True))
    pp_sh=predictive(sh,op[alt],cl[alt],low,high,1.0,-1.0,"shuffled_delta")
    rt=z.copy(); vals=rt.loc[valid].values
    if len(vals): rt.loc[valid]=np.roll(vals,min(777,len(vals)-1))
    pp_rt=predictive(rt,op[alt],cl[alt],low,high,1.0,-1.0,"random_time")

    diagnostics={
        "loaded_symbols":len(loaded),
        "btc_delta_fraction_finite":int(delta.notna().sum()),
        "btc_z_finite":int(z.notna().sum()),
        "btc_z_oos_finite":int(z.loc[HOLDOUT:].notna().sum()),
        "btc_z_oos_positive_ge_1":int((z.loc[HOLDOUT:]>=1.0).sum()),
        "btc_z_oos_negative_le_minus_1":int((z.loc[HOLDOUT:]<=-1.0).sum()),
        "low_mask_oos_true_cells":int(low.loc[HOLDOUT:].sum().sum()),
        "low_mask_oos_active_symbols":int(low.loc[HOLDOUT:].any(axis=0).sum()),
        "high_mask_oos_true_cells":int(high.loc[HOLDOUT:].sum().sum()),
        "pp_rows":pp[["horizon_bars","liquidity","signal_side","n_symbol_day","n_dates"]].to_dict("records"),
    }
    (ART/"diagnostics_pre_gate.json").write_text(json.dumps(diagnostics,indent=2))
    print(json.dumps({"stage":"pre_gate_diagnostics",**diagnostics},indent=2),flush=True)

    gate=pp[(pp.liquidity=="low_liquidity")&pp.signal_side.isin(["positive","negative"])]
    counts=gate.groupby("horizon_bars").n_symbol_day.sum()
    if counts.empty or int(counts.min())<GATE:
        raise RuntimeError(f"OOS evidence gate failed: {counts.to_dict()}")

    robust=[pp.assign(robustness_dimension="baseline",robustness_value=1.0)]
    for t in (0.75,1.25,1.5):
        robust.append(predictive(z,op[alt],cl[alt],low,high,t,-t,"btc_delta").assign(
            robustness_dimension="z_threshold",robustness_value=t))
    robust.append(predictive(pct,op[alt],cl[alt],low,high,0.75,0.25,"btc_delta_percentile").assign(
        robustness_dimension="normalization",robustness_value="causal_percentile_75_25"))
    for tail in (0.20,0.40):
        _,_,_,low_t,high_t=masks(cl[alt],vol[alt],tail,0.30)
        robust.append(predictive(z,op[alt],cl[alt],low_t,high_t,1.0,-1.0,"btc_delta").assign(
            robustness_dimension="low_liquidity_tail",robustness_value=tail))
    pp_all=pd.concat([pp,pp_ret,pp_sh,pp_rt],ignore_index=True)
    rob=pd.concat(robust,ignore_index=True)

    oos=idx>=HOLDOUT
    _,ret,weights,pstats=run_portfolio(cl.loc[oos,alt],z,low,1.0)
    stress={}
    for m in (1.5,2.0):
        _,_,_,s=run_portfolio(cl.loc[oos,alt],z,low,m)
        stress[f"{m:.1f}x"]=s

    active=low.loc[HOLDOUT:]
    manifest=[]
    for s in active.columns:
        ix=active.index[active[s].fillna(False)]
        if len(ix):
            manifest.append({"symbol":s,"first_oos_primary":str(ix.min()),
                             "last_oos_primary":str(ix.max()),"oos_primary_bars":len(ix)})
    manifest=pd.DataFrame(manifest).sort_values("symbol").reset_index(drop=True)

    pp_all.to_csv(ART/"predictive_power.csv",index=False)
    rob.to_csv(ART/"robustness.csv",index=False)
    per_symbol(z,op[alt],cl[alt],low).to_csv(ART/"per_symbol_effects.csv",index=False)
    calendar_effects(z,op[alt],cl[alt],low).to_csv(ART/"calendar_effects.csv",index=False)
    pd.DataFrame({"timestamp":cl.loc[oos,alt].index,"return":ret.values,
                  "equity_index":(1+ret.fillna(0)).cumprod().values}).to_csv(ART/"oos_returns.csv",index=False)
    weights.diff().abs().sum(axis=1).rename("absolute_target_weight_turnover").to_csv(ART/"turnover.csv",index=True)
    manifest.to_json(ART/"universe_manifest.json",orient="records",indent=2)

    versions={}
    for name in ("vectorbt","pandas","numpy","scipy","huggingface_hub"):
        try:
            m=__import__(name); versions[name]=getattr(m,"__version__","unknown")
        except Exception: versions[name]="unknown"

    summary={
        "dataset":HF_REPO,"dataset_revision":revision,"interval":FREQ,
        "start":str(START),"end":str(END),"holdout_start":str(HOLDOUT),
        "four_h_files_discovered":len(four_h),"loaded_symbols":len(loaded),
        "primary_oos_symbols":len(manifest),
        "oos_signal_observations_by_horizon":{str(int(k)):int(v) for k,v in counts.items()},
        "primary_spec":{"delta":"(2*taker_buy_volume-volume)/volume","z_window":180,"threshold":1.0,
                        "liquidity_window":180,"blue_chip_exclusion":0.20,"low_tail":0.30,
                        "high_tail":0.30,"fee_per_side":FEE,"slippage_per_side":SLIP},
        "portfolio_stats":pstats,"cost_stress":stress,"vectorbt_version":versions["vectorbt"],
        "github_sha":os.getenv("GITHUB_SHA"),"github_actions_used":True}
    (ART/"summary.json").write_text(json.dumps(summary,indent=2))
    (ART/"execution_log.json").write_text(json.dumps(log,indent=2))
    (ART/"provenance.json").write_text(json.dumps({
        "dataset":HF_REPO,"revision":revision,"four_h_files_discovered":len(four_h),
        "taker_buy_volume_required":True,"vectorbt_version":versions["vectorbt"],
        "github_sha":os.getenv("GITHUB_SHA")},indent=2))
    (ART/"REPORT.md").write_text(
        "# BTC Volume Delta -> Non-Blue-Chip Altcoins\n\n"
        "Real-data execution via the Android-triggered public GitHub Actions runner.\n\n"
        f"- Dataset: {HF_REPO}\n- Immutable revision: {revision}\n"
        f"- 4h files discovered: {len(four_h)}\n- Loaded symbols: {len(loaded)}\n"
        f"- OOS holdout: {HOLDOUT} onward\n"
        f"- Minimum pooled OOS signal observations across horizons: {int(counts.min())}\n\n"
        "## Portfolio statistics\n\n" + json.dumps(pstats,indent=2) +
        "\n\n## Cost stress\n\n" + json.dumps(stress,indent=2) +
        "\n\nDetailed predictive, robustness, per-symbol, calendar, universe, turnover, provenance, and execution artifacts are stored in this directory.\n")
    print(json.dumps(summary,indent=2),flush=True)

if __name__=="__main__":
    main()
