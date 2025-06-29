# portfolio_pipeline_async.py
"""
ASYNC USA‑Equities Pipeline v3.2  – self‑calculated ROE‑TTM
==========================================================
Runs fully asynchronous (aiohttp + aiolimiter).  Calculates ROE‑TTM from raw
statements, downloads prices (FMP bulk → Tiingo → FMP single fallback), scores
{ROE, momentum}, back‑tests a 15‑stock equal‑weight portfolio, and exports:
  • data/equity_curve.csv  • data/portfolio_latest.csv  • data/portfolio_report.pdf

Quick start
-----------
```bash
pip install python-dotenv aiohttp aiolimiter pandas numpy tqdm matplotlib
python portfolio_pipeline_async.py --progress \
  --exchange NASDAQ --mincap 1000000000 --universe 500 --fast-prices
```
"""
from __future__ import annotations
import os, io, math, argparse, asyncio
from pathlib import Path
from typing import Dict
import aiohttp, pandas as pd, numpy as np
from dotenv import load_dotenv
from aiolimiter import AsyncLimiter
from tqdm.asyncio import tqdm_asyncio as tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ─── CLI ───────────────────────────────────────────────────────────────────
P = argparse.ArgumentParser()
P.add_argument("--exchange", default="NYSE,NASDAQ")
P.add_argument("--mincap", type=int, default=1_000_000_000)
P.add_argument("--universe", type=int, default=500)
P.add_argument("--start",   default="2018-01-01")
P.add_argument("--fast-prices", action="store_true")
P.add_argument("--concurrent",  type=int, default=50)
P.add_argument("--progress", action="store_true")
A = P.parse_args()
EXCH   = [e.strip().upper() for e in A.exchange.split(',')]
MINCAP = A.mincap; UNIV = A.universe; START = A.start
FAST   = A.fast_prices; CONC = A.concurrent; SHOW = A.progress

# ─── ENV/dirs ─────────────────────────────────────────────────────────────
load_dotenv()
FMP_KEY, ALPHA_KEY, TIINGO_KEY = os.getenv("FMP_KEY"), os.getenv("ALPHA_VANTAGE_KEY"), os.getenv("TIINGO_KEY")
DATA, CACHE = Path("data"), Path("data/cache")
PRICE_DIR = CACHE / "prices"
for d in (DATA, CACHE, PRICE_DIR): d.mkdir(exist_ok=True)

# ─── HTTP helpers ─────────────────────────────────────────────────────────
HEADERS = {"User-Agent": "port-pipe/3.2"}
lim_fmp, lim_ti, lim_oth = AsyncLimiter(5,1), AsyncLimiter(100,1), AsyncLimiter(5,1)
async def fetch(sess,url,lim,tag,tries=4):
    delay=2
    for _ in range(tries):
        async with lim:
            async with sess.get(url,timeout=40) as r:
                if r.status in (429,502,503,504):
                    w=int(r.headers.get("Retry-After",delay)) if r.status==429 else delay
                    if SHOW: print(f"[RATE] {tag} {r.status}→{w}s")
                    await asyncio.sleep(w); delay*=1.5; continue
                r.raise_for_status(); ct=r.headers.get("Content-Type","")
                return await (r.json() if "json" in ct else r.text())
    raise RuntimeError(f"{tag} retries exceeded")

# ─── Universe + ROE ───────────────────────────────────────────────────────
async def screener(sess):
    eq='&'.join(f"exchange={e.lower()}" for e in EXCH)
    url=f"https://financialmodelingprep.com/api/v3/stock-screener?{eq}&marketCapMoreThan={MINCAP}&apikey={FMP_KEY}&limit=0"
    js=await fetch(sess,url,lim_fmp,"screener"); return pd.json_normalize(js) if isinstance(js,list) else pd.DataFrame()

async def enrich_mcap(sess, df):
    """Fill marketCap via /profile; handle non‑string symbols safely."""
    if "marketCap" in df.columns and df["marketCap"].notna().any():
        return df
    if SHOW:
        print("[INFO] marketCap via /profile …")

    # Ensure symbol column is string and drop NaN symbols
    df = df[df["symbol"].notna()].copy()
    df["symbol"] = df["symbol"].astype(str)

    mcap: Dict[str, float] = {}
    batches = [df.symbol.tolist()[i:i + 100] for i in range(0, len(df), 100)]

    async def batch_request(batch_syms: list[str]):
        url = ("https://financialmodelingprep.com/api/v3/profile/" +
               ",".join(batch_syms) + f"?apikey={FMP_KEY}")
        try:
            return await fetch(sess, url, lim_fmp, "profile")
        except Exception as e:
            if SHOW:
                print(f"    [WARN] profile batch failed ({batch_syms[0]}…): {e}")
            return []

    tasks = [asyncio.create_task(batch_request(b)) for b in batches]

    for fut in tqdm.as_completed(tasks, disable=not SHOW):
        for rec in await fut:
            cap = rec.get("mktCap") or rec.get("marketCap")
            sym = str(rec.get("symbol", ""))
            if cap and sym:
                mcap[sym] = cap

    df["marketCap"] = df.symbol.map(mcap)
    return df
    #if SHOW: print("[INFO] marketCap via /profile …")
    #mcap:Dict[str,float]={}
    #batches=[df.symbol.tolist()[i:i+100] for i in range(0,len(df),100)]
    #tasks=[fetch(sess,f"https://financialmodelingprep.com/api/v3/profile/{','.join(b)}?apikey={FMP_KEY}",lim_fmp,"prof") for b in batches]
    #for fut in tqdm.as_completed(tasks,disable=not SHOW):
    #    for rec in await fut:
    #        cap=rec.get("mktCap") or rec.get("marketCap");
    #        if cap:mcap[rec["symbol"]]=cap
    #df["marketCap"]=df.symbol.map(mcap);return df

async def roe_one(sess,sym):
    js=await fetch(sess,f"https://financialmodelingprep.com/api/v3/ratios-ttm/{sym}?apikey={FMP_KEY}",lim_fmp,"roe")
    if isinstance(js,list) and js and js[0].get("returnOnEquityTTM") is not None:
        return js[0]["returnOnEquityTTM"]
    inc=await fetch(sess,f"https://financialmodelingprep.com/api/v3/income-statement/{sym}?limit=4&apikey={FMP_KEY}",lim_fmp,"inc")
    bal=await fetch(sess,f"https://financialmodelingprep.com/api/v3/balance-sheet-statement/{sym}?limit=4&apikey={FMP_KEY}",lim_fmp,"bs")
    if inc and bal:
        ni=sum(r.get("netIncome",0) for r in inc); eq=bal[0].get("totalStockholdersEquity")
        if eq:return ni/eq
    return None

async def enrich_roe(sess,df):
    if SHOW: print("[INFO] ROE download …")
    sem=asyncio.Semaphore(CONC)
    async def _one(sym):
        async with sem:
            try:return sym,await roe_one(sess,sym)
            except Exception as e:
                if SHOW:print(f"    ROE {sym} err:{e}");return sym,None
    pairs=await tqdm.gather(*[_one(s) for s in df.symbol],disable=not SHOW)
    df["roe"]=pd.Series({s:v for s,v in pairs});return df

async def universe(sess):
    df=await screener(sess)
    if "symbol" not in df.columns or df.empty:
        csv=await fetch(sess,"https://www.alphavantage.co/query?function=LISTING_STATUS&state=active&apikey="+ALPHA_KEY,lim_oth,"alpha")
        df=pd.read_csv(io.StringIO(csv))
    df=df[["symbol"]].drop_duplicates()
    df=await enrich_mcap(sess,df)
    df=await enrich_roe(sess,df)
    df=df.dropna(subset=["marketCap"]).sort_values("marketCap",ascending=False)
    return df.head(UNIV) if UNIV else df

# ─── Prices ───────────────────────────────────────────────────────────────
async def bulk_prices(sess,s):
    frames=[];chunk=100
    for i in range(0,len(s),chunk):
        sub=s[i:i+chunk]
        url=f"https://financialmodelingprep.com/api/v3/historical-price-full/{','.join(sub)}?from={START}&apikey={FMP_KEY}&serietype=line"
        js=await fetch(sess,url,lim_fmp,"bulk")
        if isinstance(js,dict) and "historical" in js:js=[js]
        for rec in js:
            if not isinstance(rec,dict) or "historical" not in rec:continue
            df=pd.DataFrame(rec["historical"]);df.index=pd.to_datetime(df.date)
            frames.append(df[["close"]].rename(columns={"close":rec["symbol"]}))
    if not frames: raise RuntimeError("bulk empty")
    return pd.concat(frames,axis=1).sort_index()

def cache(sym):return PRICE_DIR/f"{sym}.csv"
async def tiingo(sess,sym):
    fp=cache(sym)
    if fp.exists():return pd.read_csv(fp,index_col=0,parse_dates=True)
    url=f"https://api.tiingo.com/tiingo/daily/{sym}/prices?token={TIINGO_KEY}&startDate={START}&format=json"
    js=await fetch(sess,url,lim_ti,"tiingo")
    df=pd.DataFrame(js);df.index=pd.to_datetime(df.date)
    out=df[["adjClose"]].rename(columns={"adjClose":sym});out.to_csv(fp);return out
async def fmp_single(sess,sym):
    url=f"https://financialmodelingprep.com/api/v3/historical-price-full/{sym}?from={START}&apikey={FMP_KEY}&serietype=line"
    js=await fetch(sess,url,lim_fmp,"single")
    df=pd.DataFrame(js["historical"]);df.index=pd.to_datetime(df.date)
    return df[["close"]].rename(columns={"close":sym})

async def prices(sess,syms):
    if FAST:
        try:return await bulk_prices(sess,syms)
        except Exception as e:
            if SHOW:print(f"[WARN] bulk err({e})→per-ticker")
    sem=asyncio.Semaphore(CONC);frames=[]
    async def one(sym):
        async with sem:
            try:frames.append(await tiingo(sess, sym))
            except Exception: frames.append(await fmp_single(sess, sym))
    await tqdm.gather(*[one(s) for s in syms], disable=not SHOW)
    return pd.concat(frames, axis=1).sort_index()

# ─── Scoring & back-test ──────────────────────────────────────────────────
def momentum(series: pd.Series, win: int = 126) -> float:
    ma = series.rolling(win).mean().iloc[-1]
    return (series.iloc[-1] / ma) - 1 if not math.isnan(ma) else np.nan


def score(univ: pd.DataFrame, pr: pd.DataFrame) -> pd.DataFrame:
    mom = {c: momentum(pr[c].dropna()) for c in pr.columns}
    univ = univ.set_index("symbol")
    univ["mom"] = pd.Series(mom)
    univ["roe"].fillna(0, inplace=True)

    for col in ["roe", "mom"]:
        univ[f"z_{col}"] = (univ[col] - univ[col].mean()) / univ[col].std(ddof=0)

    univ["score"] = univ[["z_roe", "z_mom"]].mean(axis=1)
    return univ.sort_values("score", ascending=False)


def build_port(scores: pd.DataFrame, k: int = 15) -> pd.DataFrame:
    port = scores.head(k).copy()
    port["target_weight"] = 1 / k
    return port


def rebalance_dates(pr: pd.DataFrame):
    return list(pr.resample("ME").first().index)      # month-end


def backtest(pr: pd.DataFrame, port: pd.DataFrame, dates):
    tickers = port.index.tolist()
    w = port["target_weight"].values
    rets = pr[tickers].pct_change(fill_method=None).fillna(0)

    weights = np.zeros(len(tickers))
    curve = []
    for dt, row in rets.iterrows():
        if dt in dates:
            weights[:] = w
        curve.append((row.values * weights).sum())

    return (pd.Series(curve, index=rets.index) + 1).cumprod()
# ───── BLOCK 3 END ─────
# ─── Main orchestration ───────────────────────────────────────────────────
async def main():
    async with aiohttp.ClientSession() as sess:
        if SHOW: print("[1/4] Universe …")
        univ = await universe(sess)
        print(f"   → {len(univ)} tickers")

        if SHOW: print("[2/4] Prices …")
        pr = await prices(sess, univ.symbol.tolist())

        if SHOW: print("[3/4] Scoring …")
        sc = score(univ, pr)

        if SHOW: print("[4/4] Back-test …")
        port = build_port(sc)
        curve = backtest(pr, port, rebalance_dates(pr))

        # ── export CSVs
        curve.to_csv(DATA / "equity_curve.csv")
        port.to_csv(DATA / "portfolio_latest.csv")

        # ── PDF report
        pdf_path = DATA / "portfolio_report.pdf"
        with PdfPages(pdf_path) as pdf:
            plt.figure(figsize=(8, 5))
            plt.plot(curve)
            plt.title("Equity curve"); plt.tight_layout(); pdf.savefig(); plt.close()

            plt.figure(figsize=(8.5, 6)); plt.axis("off")
            plt.table(cellText=port.reset_index().values,
                      colLabels=port.reset_index().columns,
                      cellLoc="center", loc="center")
            plt.title("Latest portfolio (Top 15)"); pdf.savefig(); plt.close()

        print(f"✓ Done – CSV + PDF saved under data/  ({pdf_path})")


# ─── Guard ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    asyncio.run(main())
# ───── BLOCK 4 END ─────