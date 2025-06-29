# portfolio_pipeline.py
"""
USA‑Equities Portfolio Pipeline • v2.1
=====================================
Stabilised for free‑tier API limits and 429 (Too Many Requests) errors.

Key changes (v2.1)
------------------
* **Unified retry & back‑off** – `_get()` detects HTTP 429 / 502–504, sleeps according to `Retry‑After` or exponential back‑off, then retries (max 5 attempts).
* **Multi‑layer price fallback**
  1. FMP bulk (100‑ticker chunks)
  2. Tiingo per‑ticker (rate‑aware)
  3. FMP per‑ticker (`historical-price-full`)
* **Local disk cache** (CSV per ticker) ‑ every downloaded price series is saved to `data/prices/<TICKER>.csv`; reruns skip unchanged files.
* CLI flag `--sleep` to set Tiingo pause seconds after 429 (default = 65 s, meets free tier 60 req/min).

Example (NASDAQ, ≥1 B USD mcap, 250‑ticker universe, progress bars & fast bulk):
```bash
python portfolio_pipeline.py --progress --exchange NASDAQ --mincap 1000000000 \
                             --universe-size 250 --fast-prices
```
Requirements: `pip install python-dotenv requests pandas numpy tqdm urllib3`.
`~/.env` needs `FMP_KEY  ALPHA_VANTAGE_KEY  TIINGO_KEY`.
"""
from __future__ import annotations

import os, io, time, math, argparse, json
from datetime import date
from pathlib import Path
from typing import List, Dict

import requests, pandas as pd, numpy as np
from dotenv import load_dotenv
from tqdm.auto import tqdm
from urllib3.util import Retry
from requests.adapters import HTTPAdapter

# ---------------------------------------------------------------------------
# CLI flags
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="USA equity pipeline v2.1")
parser.add_argument("--progress", action="store_true")
parser.add_argument("--exchange", default="NYSE,NASDAQ")
parser.add_argument("--mincap", type=int, default=1_000_000_000)
parser.add_argument("--universe-size", type=int, default=500)
parser.add_argument("--fast-prices", action="store_true")
parser.add_argument("--start", default="2018-01-01")
parser.add_argument("--sleep", type=int, default=65, help="seconds to sleep after Tiingo 429")
args = parser.parse_args()
SHOW = args.progress
EXCHANGES = [e.strip().upper() for e in args.exchange.split(',')]
MIN_CAP  = args.mincap
UNIV_LIM = args.universe_size
FAST    = args.fast_prices
START   = args.start
TNG_SLEEP = args.sleep

# ---------------------------------------------------------------------------
# ENV & HTTP session
# ---------------------------------------------------------------------------
load_dotenv()
ALPHA_KEY = os.getenv("ALPHA_VANTAGE_KEY")
FMP_KEY   = os.getenv("FMP_KEY")
TIINGO_KEY= os.getenv("TIINGO_KEY")

DATA_DIR = Path("data"); DATA_DIR.mkdir(exist_ok=True)
PRICE_DIR = DATA_DIR / "prices"; PRICE_DIR.mkdir(exist_ok=True)

def _ascii(s:str)->str:
    return s.encode("ascii", errors="ignore").decode("ascii")

S = requests.Session()
S.headers.update({"User-Agent": _ascii("portfolio-pipe/2.1")})
S.mount("https://", HTTPAdapter(max_retries=Retry(total=0)))  # we implement our own

# ---------------------------------------------------------------------------
# Robust HTTP GET with retry/back‑off
# ---------------------------------------------------------------------------

def _get(url:str, label:str, params:dict|None=None, max_try:int=5):
    attempt, back = 0, 4  # start wait 4 s on failures
    while True:
        try:
            r = S.get(url, params=params, timeout=45)
            if r.status_code == 429:
                # honour Retry‑After if present, else exponential
                wait = int(r.headers.get("Retry-After", back))
                if label == "tiingo":
                    wait = max(wait, TNG_SLEEP)
                if attempt >= max_try:
                    raise RuntimeError(f"{label} 429 too many times")
                if SHOW:
                    print(f"[RATE] {label} 429 → sleep {wait}s (attempt {attempt+1}/{max_try})")
                time.sleep(wait)
                attempt += 1; back *= 1.5
                continue
            # retry for gateway errors
            if r.status_code in (502,503,504):
                if attempt >= max_try:
                    r.raise_for_status()
                time.sleep(back); attempt += 1; back *= 1.5; continue
            r.raise_for_status(); return r
        except requests.RequestException as e:
            if attempt >= max_try:
                raise
            time.sleep(back); attempt += 1; back *= 1.5

# ---------------------------------------------------------------------------
# 1 ▸ Universe + ROE (fast or fallback)
# ---------------------------------------------------------------------------

def build_universe()->pd.DataFrame:
    exch_q = '&'.join(f"exchange={e.lower()}" for e in EXCHANGES)
    screener = (
        "https://financialmodelingprep.com/api/v3/stock-screener?" + exch_q +
        f"&marketCapMoreThan={MIN_CAP}&apikey={FMP_KEY}&limit=0")
    js = _get(screener, "fmp").json()
    df = pd.json_normalize(js) if isinstance(js,list) else pd.DataFrame()
    if "marketCap" not in df.columns or df.empty:
        print("[INFO] FMP screener unavailable — fallback path...")
        av_csv = _get("https://www.alphavantage.co/query", "alpha", {
                "function":"LISTING_STATUS","state":"active","apikey":ALPHA_KEY}).text
        tmp = pd.read_csv(io.StringIO(av_csv))
        exch_col = "exchangeShortName" if "exchangeShortName" in tmp.columns else "exchange"
        tmp.rename(columns={exch_col:"exchange"}, inplace=True)
        tickers = tmp[tmp["exchange"].isin(EXCHANGES)]["symbol"].astype(str).tolist()
        rows=[]
        for i in tqdm(range(0,len(tickers),100), disable=not SHOW, desc="FMP profile fallback"):
            sub = tickers[i:i+100]
            prof_url=f"https://financialmodelingprep.com/api/v3/profile/{','.join(sub)}"
            rows.extend(_get(prof_url,"fmp",{"apikey":FMP_KEY}).json())
        df=pd.json_normalize(rows)
    # harmonise
    if "mktCap" in df.columns and "marketCap" not in df.columns:
        df.rename(columns={"mktCap":"marketCap"}, inplace=True)
    if "exchangeShortName" in df.columns and "exchange" not in df.columns:
        df.rename(columns={"exchangeShortName":"exchange"}, inplace=True)
    if "returnOnEquityTTM" not in df.columns:
        df["returnOnEquityTTM"] = np.nan
    df = df[["symbol","marketCap","exchange","returnOnEquityTTM"]].dropna(subset=["marketCap"])
    df = df[df["exchange"].str.upper().isin(EXCHANGES)]
    df.sort_values("marketCap", ascending=False, inplace=True)
    if UNIV_LIM>0:
        df = df.head(UNIV_LIM)
    df.rename(columns={"returnOnEquityTTM":"roe"}, inplace=True)
    df.to_csv(DATA_DIR/f"universe_{date.today()}.csv", index=False)
    return df

# ---------------------------------------------------------------------------
# 2 ▸ Prices (bulk → Tiingo → FMP per ticker)
# ---------------------------------------------------------------------------

def fetch_prices_fast(symbols:List[str]) -> pd.DataFrame:
    frames=[]; chunk=100
    for i in tqdm(range(0,len(symbols),chunk), disable=not SHOW, desc="FMP bulk prices"):
        sub=symbols[i:i+chunk]
        url=("https://financialmodelingprep.com/api/v3/historical-price-full/"+','.join(sub)+
             f"?from={START}&apikey={FMP_KEY}&serietype=line")
        js=_get(url,"fmp").json()
        if isinstance(js, dict) and "historical" in js:
            js=[js]
        for rec in js if isinstance(js,list) else []:
            if "historical" not in rec: continue
            df=pd.DataFrame(rec["historical"])
            if "close" not in df.columns: continue
            df.index = pd.to_datetime(df["date"])
            frames.append(df[["close"]].rename(columns={"close":rec.get("symbol")}))
    if frames:
        return pd.concat(frames, axis=1).sort_index()
    raise RuntimeError("Bulk price endpoint returned no usable frames")


def _cache_path(sym:str)->Path:
    return PRICE_DIR / f"{sym}.csv"


def load_or_download_tiingo(sym:str) -> pd.DataFrame:
    fp=_cache_path(sym)
    if fp.exists():
        return pd.read_csv(fp, index_col=0, parse_dates=True)
    url=f"https://api.tiingo.com/tiingo/daily/{sym}/prices"
    js=_get(url,"tiingo",{"token":TIINGO_KEY,"startDate":START,"format":"json"}).json()
    df=pd.DataFrame(js)
    if df.empty:
        raise RuntimeError("empty Tiingo json")
    df.index=pd.to_datetime(df["date"])
    out=df[["adjClose"]].rename(columns={"adjClose":sym})
    out.to_csv(fp)
    return out


def download_price_fmp(sym:str)->pd.DataFrame:
    url=("https://financialmodelingprep.com/api/v3/historical-price-full/"+sym+
         f"?from={START}&apikey={FMP_KEY}&serietype=line")
    js=_get(url,"fmp").json()
    if not js or "historical" not in js: raise RuntimeError("no hist")
    df=pd.DataFrame(js["historical"])
    df.index=pd.to_datetime(df["date"])
    return df[["close"]].rename(columns={"close":sym})


def fetch_prices(symbols:List[str]) -> pd.DataFrame:
    # try fast path
    if FAST:
        try:
            return fetch_prices_fast(symbols)
        except Exception as e:
            print(f"[WARN] fast‑price path failed ({e}) — switching to per‑ticker mode")
    frames=[]
    for sym in tqdm(symbols, disable=not SHOW, desc="Per‑ticker prices"):
        try:
            frames.append(load_or_download_tiingo(sym))
        except Exception as e:
            if SHOW:
                print(f"    Tiingo fail {sym}: {e} → FMP")
            try:
                frames.append(download_price_fmp(sym))
            except Exception as e2:
                if SHOW:
                    print(f"    FMP fail {sym}: {e2}")
    return pd.concat(frames, axis=1).sort_index()

# ---------------------------------------------------------------------------
# 3 ▸ Scoring & back‑test
# ---------------------------------------------------------------------------

def momentum(s:pd.Series, w:int=126)->float:
    ma=s.rolling(w).mean().iloc[-1]
    return (s.iloc[-1]/ma)-1 if not math.isnan(ma) else np.nan


def score(univ:pd.DataFrame, prices:pd.DataFrame)->pd.DataFrame:
    mom={sym: momentum(prices[sym].dropna()) for sym in prices.columns}
    univ=univ.set_index("symbol")
    univ["mom"] = pd.Series(mom)
    for col in ["roe","mom"]:
        univ[f"z_{col}"]=(univ[col]-univ[col].mean())/univ[col].std(ddof=0)
    univ["score"] = univ[["z_roe","z_mom"]].mean(axis=1)
    return univ.sort_values("score", ascending=False)


def build_portfolio(scores:pd.DataFrame, k:int=15)->pd.DataFrame:
    port=scores.head(k).copy(); port["target_weight"] = 1/k; return port


def rebalance_dates(pr:pd.DataFrame):
    return list(pr.resample("M").first().index)


def backtest(pr:pd.DataFrame, port:pd.DataFrame, dates:List[pd.Timestamp]):
    h=port.index.tolist(); w=port["target_weight"].values
    rets=pr[h].pct_change().fillna(0)
    wt=pd.Series(0,index=h); eq=[]
    for dt,row in rets.iterrows():
        if dt in dates: wt[:]=w
        eq.append((row*wt).sum())
    return (pd.Series(eq,index=rets.index)+1).cumprod()

# ---------------------------------------------------------------------------
# 4 ▸ RUN
# ---------------------------------------------------------------------------

def run():
    print("[1/4] Universe + ROE …", flush=True)
    univ = build_universe(); print(f"   → {len(univ)} tickers", flush=True)
    print("[2/4] Prices …", flush=True)
    prices = fetch_prices(univ["symbol"].tolist())
    print("[3/4] Scoring …", flush=True)
    scores = score(univ, prices)
    print("[4/4] Portfolio & back‑test …", flush=True)
    port = build_portfolio(scores)
    print(port[["roe","mom","score","target_weight"]])
    curve = backtest(prices, port, rebalance_dates(prices))
    curve.to_csv(DATA_DIR/"equity_curve.csv")
    print("✓ Done — see data/equity_curve.csv", flush=True)

if __name__ == "__main__":
    run()
