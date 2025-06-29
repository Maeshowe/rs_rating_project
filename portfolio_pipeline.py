# portfolio_pipeline.py
"""
USA-Equities Portfolio Pipeline **v2.0 — Turbo Edition**
=======================================================
⚡ **Gyorsítások**
-----------------
1. **Egyetlen FMP Stock‑Screener hívás** hozza:
   * `symbol`, `marketCap`, `exchange`, **`returnOnEquityTTM`** (ROE) – így teljesen elhagyjuk a ticker‑enkénti fundamentum‑loopot.
2. Új `--universe-size N` flaggel már a kezdetektől toplistára szűkíthetsz (legnagyobb mcap szerint), pl. `--universe-size 500`.
3. `--fast-prices` továbbra is egy darab bulk EOD‑kérést jelent (FMP), tehát a **nasdaq‑1B** szűréssel a teljes pipeline <60 s‐en belül lefut.

```bash
pip install python-dotenv requests pandas numpy tqdm urllib3
python portfolio_pipeline.py --progress --exchange NASDAQ --mincap 1000000000 \
                             --universe-size 500 --fast-prices
```
"""
from __future__ import annotations

import os, io, time, math, argparse
from datetime import date
from pathlib import Path
from typing import List, Dict

import requests, pandas as pd, numpy as np
from dotenv import load_dotenv
from urllib3.util import Retry
from requests.adapters import HTTPAdapter
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# CLI flags
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="USA equity portfolio pipeline — turbo")
parser.add_argument("--progress", action="store_true", help="show progress bars")
parser.add_argument("--exchange", default="NYSE,NASDAQ", help="comma list, e.g. NASDAQ")
parser.add_argument("--mincap", type=int, default=1_000_000_000, help="min market‑cap USD")
parser.add_argument("--universe-size", type=int, default=1000, help="keep N largest by mcap (0=all)")
parser.add_argument("--fast-prices", action="store_true", help="use FMP bulk EOD api")
parser.add_argument("--start", default="2018-01-01", help="price history start date")
args = parser.parse_args()
SHOW = args.progress
EXCHANGES = [e.strip().upper() for e in args.exchange.split(',')]
MIN_CAP = args.mincap
UNIV_LIMIT = args.universe_size
FAST_PRICES = args.fast_prices
START_DATE = args.start

# ---------------------------------------------------------------------------
# Environment + resilient HTTP
# ---------------------------------------------------------------------------
load_dotenv()
ALPHA_KEY = os.getenv("ALPHA_VANTAGE_KEY")
FMP_KEY   = os.getenv("FMP_KEY")
TIINGO_KEY= os.getenv("TIINGO_KEY")

OUT_DIR = Path("data"); OUT_DIR.mkdir(exist_ok=True)

def _ascii(s:str)->str:
    """Strip non‑ASCII characters (avoids 'latin-1' encode errors in headers)."""
    return s.encode("ascii", errors="ignore").decode("ascii")

S = requests.Session();
S.headers.update({"User-Agent": _ascii("port-pipe/2.0")})
S.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1, status_forcelist=[502,503,504])))
RATE = {"fmp":1, "tiingo":1}

def _get(url:str,label:str,params:dict|None=None):
    time.sleep(RATE.get(label,0))
    r=S.get(url,params=params,timeout=40); r.raise_for_status(); return r

# ---------------------------------------------------------------------------
# 1 ▸ Universe + ROE in one shot
# ---------------------------------------------------------------------------

def build_universe() -> pd.DataFrame:
    """Return DataFrame with symbol, marketCap, exchange, roe.

    Fast path: single FMP stock‑screener call (requires paid plan for ROE).
    Fallback: Alpha Vantage CSV + FMP profile batches → still only ~500 calls
    when universe‑size limited.
    """
    exch_query = "&".join(f"exchange={e.lower()}" for e in EXCHANGES)
    screener_url = (
        "https://financialmodelingprep.com/api/v3/stock-screener?" + exch_query +
        f"&marketCapMoreThan={MIN_CAP}&apikey={FMP_KEY}&limit=0"
    )

    try:
        js = _get(screener_url, "fmp").json()
    except Exception as e:
        print(f"[WARN] Screener request failed ({e}) — falling back.")
        js = []

    if isinstance(js, list) and js:
        df = pd.json_normalize(js)
        # some free tiers omit ROE; fill with NaN if missing
        if "returnOnEquityTTM" not in df.columns:
            df["returnOnEquityTTM"] = np.nan
        df = df[["symbol", "marketCap", "exchange", "returnOnEquityTTM"]]
        df.rename(columns={"returnOnEquityTTM": "roe"}, inplace=True)
        df.dropna(subset=["marketCap"], inplace=True)
    else:
        print("[INFO] Using fallback: Alpha Vantage LISTING_STATUS + FMP profile")
        # Alpha Vantage ticker list
        av_csv = _get(
            "https://www.alphavantage.co/query", "alpha", {
                "function": "LISTING_STATUS", "state": "active", "apikey": ALPHA_KEY
            }).text
        csv_df = pd.read_csv(io.StringIO(av_csv))
        exch_col = "exchangeShortName" if "exchangeShortName" in csv_df.columns else "exchange"
        csv_df.rename(columns={exch_col: "exchange"}, inplace=True)
        tickers = csv_df[csv_df["exchange"].isin(EXCHANGES)]["symbol"].astype(str).tolist()
        # profile batches
        rows: List[dict] = []
        for i in tqdm(range(0, len(tickers), 100), disable=not SHOW, desc="FMP profile fallback"):
            batch = tickers[i:i+100]
            prof_url = f"https://financialmodelingprep.com/api/v3/profile/{','.join(batch)}"
            try:
                rows.extend(_get(prof_url, "fmp", {"apikey": FMP_KEY}).json())
            except Exception:
                continue
        prof_df = pd.json_normalize(rows)
        # Harmonise column names
        if "mktCap" in prof_df.columns and "marketCap" not in prof_df.columns:
            prof_df.rename(columns={"mktCap": "marketCap"}, inplace=True)
        if "exchangeShortName" in prof_df.columns and "exchange" not in prof_df.columns:
            prof_df.rename(columns={"exchangeShortName": "exchange"}, inplace=True)
        # ensure required columns
        required_cols = {"symbol", "marketCap", "exchange"}
        if not required_cols.issubset(prof_df.columns):
            missing = required_cols - set(prof_df.columns)
            raise RuntimeError(f"profile fallback missing {missing} — check API/plan")
        if "returnOnEquityTTM" not in prof_df.columns:
            prof_df["returnOnEquityTTM"] = np.nan
        df = prof_df[["symbol", "marketCap", "exchange", "returnOnEquityTTM"]][["symbol", "marketCap", "exchange", "returnOnEquityTTM"]]
        df.rename(columns={"returnOnEquityTTM": "roe"}, inplace=True)
        df.dropna(subset=["marketCap"], inplace=True)
        df = df[df["marketCap"] >= MIN_CAP]

    # final sorting & limit
    df.sort_values("marketCap", ascending=False, inplace=True)
    if UNIV_LIMIT > 0:
        df = df.head(UNIV_LIMIT)

    df.to_csv(OUT_DIR / f"universe_{date.today()}.csv", index=False)
    return df

# ---------------------------------------------------------------------------
# 2 ▸ Prices (bulk or Tiingo)
# ---------------------------------------------------------------------------

def fetch_prices_fast(symbols: List[str]) -> pd.DataFrame:
    """Bulk close prices via FMP. If the bulk endpoint fails, fall back to per‑ticker."""
    frames = []
    chunk = 100  # FMP bulk endpoint works reliably up to ~100 tickers
    for i in tqdm(range(0, len(symbols), chunk), disable=not SHOW, desc="FMP bulk prices"):
        sub = symbols[i:i+chunk]
        syms = ','.join(sub)
        url = (
            "https://financialmodelingprep.com/api/v3/historical-price-full/" + syms +
            f"?from={START_DATE}&apikey={FMP_KEY}&serietype=line"
        )
        js = _get(url, "fmp").json()
        # When only one symbol is requested, API returns dict not list
        if isinstance(js, dict) and "historical" in js:
            js = [js]
        if isinstance(js, list):
            for rec in js:
                if not isinstance(rec, dict) or "historical" not in rec:
                    continue
                df = pd.DataFrame(rec["historical"])
                if "close" not in df.columns:
                    continue
                df.index = pd.to_datetime(df["date"])
                frames.append(df[["close"]].rename(columns={"close": rec.get("symbol", syms)}))
    if not frames:
        raise RuntimeError("Bulk price endpoint returned no data")
    return pd.concat(frames, axis=1).sort_index()(frames,axis=1).sort_index()


def fetch_price_tiingo(sym:str)->pd.DataFrame:
    url=f"https://api.tiingo.com/tiingo/daily/{sym}/prices"
    js=_get(url,"tiingo",{"token":TIINGO_KEY,"startDate":START_DATE,"format":"json"}).json()
    df=pd.DataFrame(js); df.index=pd.to_datetime(df["date"])
    return df[["adjClose"]].rename(columns={"adjClose":sym})


def fetch_prices(symbols: List[str]) -> pd.DataFrame:
    if FAST_PRICES:
        try:
            return fetch_prices_fast(symbols)
        except Exception as e:
            print(f"[WARN] fast-price path failed ({e}) — falling back to Tiingo per ticker")
    frames = [fetch_price_tiingo(s) for s in tqdm(symbols, disable=not SHOW, desc="Tiingo prices")]
    return pd.concat(frames, axis=1).sort_index()(frames,axis=1).sort_index()

# ---------------------------------------------------------------------------
# 3 ▸ Scoring (ROE from screener + momentum)
# ---------------------------------------------------------------------------

def momentum(series:pd.Series, w:int=126)->float:
    sma=series.rolling(w).mean().iloc[-1]
    return (series.iloc[-1]/sma)-1 if not math.isnan(sma) else np.nan


def score(univ:pd.DataFrame, prices:pd.DataFrame)->pd.DataFrame:
    mom={s:momentum(prices[s].dropna()) for s in prices.columns}
    univ=univ.set_index("symbol")
    univ["mom"]=pd.Series(mom)
    for col in ["roe","mom"]:
        univ[f"z_{col}"]=(univ[col]-univ[col].mean())/univ[col].std(ddof=0)
    univ["score"] = univ[["z_roe","z_mom"]].mean(axis=1)
    return univ.sort_values("score",ascending=False)

# ---------------------------------------------------------------------------
# 4 ▸ Portfolio & Back‑test
# ---------------------------------------------------------------------------

def build_portfolio(scores:pd.DataFrame, top:int=15)->pd.DataFrame:
    port=scores.head(top).copy(); port["target_weight"]=1/top; return port


def rebalance_dates(prices:pd.DataFrame):
    return list(prices.resample("M").first().index)


def backtest(prices:pd.DataFrame,port:pd.DataFrame,dates:List[pd.Timestamp]):
    holds=port.index.tolist(); w=port["target_weight"].values
    rets=prices[holds].pct_change().fillna(0)
    weights=pd.Series(0,index=holds); curve=[]
    for dt,row in rets.iterrows():
        if dt in dates: weights[:]=w
        curve.append((row*weights).sum())
    return (pd.Series(curve,index=rets.index)+1).cumprod()

# ---------------------------------------------------------------------------
# 5 ▸ Run pipeline
# ---------------------------------------------------------------------------

def run():
    print("[1/4] Universe + ROE …", flush=True)
    univ=build_universe(); print(f" → {len(univ)} tickers")
    print("[2/4] Prices …", flush=True)
    prices=fetch_prices(univ["symbol"].tolist())
    print("[3/4] Scoring …", flush=True)
    scores=score(univ,prices)
    print("[4/4] Portfolio & back‑test …", flush=True)
    port=build_portfolio(scores); print(port[["roe","mom","score","target_weight"]])
    curve=backtest(prices,port,rebalance_dates(prices))
    curve.to_csv(OUT_DIR/"equity_curve.csv")
    print("✓ Done — results in data/equity_curve.csv", flush=True)

if __name__=="__main__":
    run()
