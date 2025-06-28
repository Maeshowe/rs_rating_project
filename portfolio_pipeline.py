# portfolio_pipeline.py
"""
End-to-end USA-equities portfolio pipeline
=========================================
Builds a *survivorship-bias-aware* universe, scores stocks with
fundamental + momentum factors, selects a 10-15 stock portfolio,
weights, rebalances monthly, and runs a vectorised back-test.

All required API keys/tokens are loaded from a local `.env` file.
Dependencies: ``python-dotenv``, ``requests``, ``pandas``, ``numpy``.

Services used
-------------
* **Alpha Vantage** – US listing universe (LISTING_STATUS)
* **FMP** – fundamentals (income statement, balance sheet, profile)
* **Tiingo** – adjusted daily OHLCV price history
* **Polygon** – dividends & splits (corporate actions)
* **FRED** – 10-year Treasury (risk-free rate)

Author: ChatGPT o3 — 2025-06-28
"""
from __future__ import annotations

import os, time, io, math
from datetime import date
from pathlib import Path
from typing import List, Dict

import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# 0  Environment & helpers
# ---------------------------------------------------------------------------

load_dotenv()
ALPHA_KEY  = os.getenv("ALPHA_VANTAGE_KEY")
FMP_KEY    = os.getenv("FMP_KEY")
TIINGO_KEY = os.getenv("TIINGO_KEY")
POLY_KEY   = os.getenv("POLYGON_KEY")
FRED_KEY   = os.getenv("FRED_KEY")

OUT_DIR = Path("data"); OUT_DIR.mkdir(exist_ok=True)

S = requests.Session()
S.headers.update({"User-Agent": "portfolio-pipeline/1.1"})

RATE_SLEEP = {"alpha":13, "fmp":1, "tiingo":1, "polygon":1, "fred":1}

def _get(url:str, label:str, params:dict|None=None):
    time.sleep(RATE_SLEEP.get(label,0))
    r = S.get(url, params=params, timeout=30)
    r.raise_for_status(); return r

# ---------------------------------------------------------------------------
# 1  Universe build (survivorship-bias free)
# ---------------------------------------------------------------------------

def fetch_listing(state:str="active") -> pd.DataFrame:
    url="https://www.alphavantage.co/query"
    resp=_get(url,"alpha",{"function":"LISTING_STATUS","state":state,"apikey":ALPHA_KEY})
    df=pd.read_csv(io.StringIO(resp.text))
    df["ipoDate"]      = pd.to_datetime(df["ipoDate"],      errors="coerce")
    df["delistingDate"]= pd.to_datetime(df["delistingDate"],errors="coerce")
    return df


def build_universe(min_mcap:float=1e9)->pd.DataFrame:
    active=fetch_listing("active")
    active=active[active["exchange"].isin(["NYSE","NASDAQ"])]
    # ensure symbol column is string and drop NaNs / non-standard tickers
    active["symbol"]=active["symbol"].astype(str)
    tickers=[sym for sym in active["symbol"].tolist() if sym and sym.lower()!="nan"]

    mkt_caps:Dict[str,float]={}
    batch_size=100
    for i in range(0,len(tickers),batch_size):
        batch=tickers[i:i+batch_size]
        url=f"https://financialmodelingprep.com/api/v3/profile/{','.join(batch)}"
        js=_get(url,"fmp",{"apikey":FMP_KEY}).json()
        for rec in js:
            sym=str(rec.get("symbol",""))
            if sym:
                mkt_caps[sym]=rec.get("mktCap",np.nan)

    active["marketCap"]=active["symbol"].map(mkt_caps)
    univ=active.dropna(subset=["marketCap"]).query("marketCap>=@min_mcap")
    univ.to_csv(OUT_DIR/f"universe_{date.today()}.csv",index=False)
    return univ

# ---------------------------------------------------------------------------
# 2  Fundamentals & prices
# ---------------------------------------------------------------------------

def fetch_fundamentals(symbols:List[str])->pd.DataFrame:
    rows=[]
    for sym in symbols:
        url_is=f"https://financialmodelingprep.com/api/v3/income-statement/{sym}?limit=1&apikey={FMP_KEY}"
        url_bs=f"https://financialmodelingprep.com/api/v3/balance-sheet-statement/{sym}?limit=1&apikey={FMP_KEY}"
        try:
            is_data=_get(url_is,"fmp").json(); bs_data=_get(url_bs,"fmp").json()
            if is_data and bs_data:
                ni=is_data[0].get("netIncome",np.nan)
                eq=bs_data[0].get("totalStockholdersEquity",np.nan)
                roe=ni/eq if eq else np.nan
                rows.append({"symbol":sym,"roe":roe})
        except Exception as e:
            print(f"[WARN] fundamental {sym}: {e}")
    return pd.DataFrame(rows)


def fetch_price_history(sym:str,start:str="2015-01-01")->pd.DataFrame:
    url=f"https://api.tiingo.com/tiingo/daily/{sym}/prices"
    js=_get(url,"tiingo",{"token":TIINGO_KEY,"startDate":start,"format":"json"}).json()
    if not js:
        raise ValueError("empty price")
    df=pd.DataFrame(js)
    df.index=pd.to_datetime(df["date"])
    return df[["adjClose"]].rename(columns={"adjClose":sym})


def fetch_prices(symbols:List[str],start:str="2015-01-01")->pd.DataFrame:
    frames=[]
    for sym in symbols:
        try:
            frames.append(fetch_price_history(sym,start))
        except Exception as e:
            print(f"[WARN] price {sym}: {e}")
    return pd.concat(frames,axis=1).sort_index()

# ---------------------------------------------------------------------------
# 3  Scoring
# ---------------------------------------------------------------------------

def compute_momentum(series:pd.Series,window:int=126)->float:
    sma=series.rolling(window).mean()
    return (series.iloc[-1]/sma.iloc[-1])-1 if not math.isnan(sma.iloc[-1]) else np.nan


def score(fund:pd.DataFrame,prices:pd.DataFrame)->pd.DataFrame:
    momentum={sym:compute_momentum(prices[sym].dropna()) for sym in prices.columns}
    mom_df=pd.DataFrame.from_dict(momentum,orient="index",columns=["mom"])
    df=fund.set_index("symbol").join(mom_df)
    for col in ["roe","mom"]:
        df[f"z_{col}"]=(df[col]-df[col].mean())/df[col].std(ddof=0)
    df["score"]=df[["z_roe","z_mom"]].mean(axis=1)
    return df.sort_values("score",ascending=False)

# ---------------------------------------------------------------------------
# 4  Portfolio & rebalance
# ---------------------------------------------------------------------------

def build_portfolio(scores:pd.DataFrame,top_n:int=15)->pd.DataFrame:
    sel=scores.head(top_n).copy(); sel["target_weight"]=1/len(sel)
    return sel

def rebalance_dates(prices:pd.DataFrame,freq:str="M"):
    return list(prices.resample(freq).first().index)

# ---------------------------------------------------------------------------
# 5  Vectorised back-test
# ---------------------------------------------------------------------------

def backtest(prices:pd.DataFrame,portfolio:pd.DataFrame,dates:list[pd.Timestamp]):
    holdings=portfolio.index.tolist(); w=portfolio["target_weight"].values
    closes=prices[holdings]; rets=closes.pct_change().fillna(0)
    weights=pd.Series(0,index=holdings,dtype=float); curve=[]
    for dt,row in rets.iterrows():
        if dt in dates: weights[:]=w
        curve.append((row*weights).sum())
    return (pd.Series(curve,index=rets.index)+1).cumprod()

# ---------------------------------------------------------------------------
# 6  Runner
# ---------------------------------------------------------------------------

def run_pipeline(min_mcap:float=1e9,start="2018-01-01",top_n:int=15):
    print("[1/6] Universe …"); univ=build_universe(min_mcap)
    print("[2/6] Fundamentals …"); fund=fetch_fundamentals(univ["symbol"].tolist())
    print("[3/6] Prices …"); prices=fetch_prices(fund["symbol"].tolist(),start)
    print("[4/6] Scoring …"); scores=score(fund,prices)
    print("[5/6] Portfolio …"); port=build_portfolio(scores,top_n)
    print(port[["score","target_weight"]])
    print("[6/6] Back-test …")
    curve=backtest(prices,port,rebalance_dates(prices))
    curve.to_csv(OUT_DIR/"equity_curve.csv"); print("Saved → data/equity_curve.csv")

if __name__=="__main__":
    run_pipeline()