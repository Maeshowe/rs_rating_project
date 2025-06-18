"""
fetch.py
~~~~~~~~
Letölt legalább 252 kereskedési napnyi záróárfolyamot a Polygon.io-ról,
majd DataFrame-ként adja vissza. 2025-ös Python-on (3.13+) kompatibilis,
timezone-aware dátumokkal és beépített Retry-mechanizmussal.

Függ a következőktől:
  • requests
  • pandas
  • python-dotenv  (API-kulcs .env-ből)

Környezeti változó:
  POLYGON_API_KEY=<pk_...>
"""

from __future__ import annotations

import os
import datetime as dt
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ────────────────────────────────────────────────────────────────────
# .env betöltése (projekt gyökeréből)
# ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

API_KEY = os.getenv("POLYGON_API_KEY")
if not API_KEY:
    raise EnvironmentError("❌  POLYGON_API_KEY nincs beállítva a .env fájlban!")

# ────────────────────────────────────────────────────────────────────
# Konstansok
# ────────────────────────────────────────────────────────────────────
BASE = (
    "https://api.polygon.io/v2/aggs/ticker/{sym}/range/1/day/"
    "{start}/{end}?unadjusted=false&apiKey={key}"
)
DEFAULT_WINDOW_DAYS = 400  # kicsit > 1 év, hogy mindig legyen 252 trading nap

# ────────────────────────────────────────────────────────────────────
# Requests session Retry-vel
# ────────────────────────────────────────────────────────────────────
sess = requests.Session()
sess.mount(
    "https://",
    HTTPAdapter(
        max_retries=Retry(
            total=5,                # 5 próbálkozás
            backoff_factor=1.2,     # exponenciális (1.2, 2.4, 4.8… másodpercek)
            status_forcelist=[429, 500, 502, 503, 504],
            raise_on_status=False,
        )
    ),
)

# ────────────────────────────────────────────────────────────────────
# Segédfüggvények
# ────────────────────────────────────────────────────────────────────
def _date_range(days: int = DEFAULT_WINDOW_DAYS) -> tuple[str, str]:
    end = dt.date.today()
    start = end - dt.timedelta(days=days)
    return start.isoformat(), end.isoformat()


# ────────────────────────────────────────────────────────────────────
# Publikus API
# ────────────────────────────────────────────────────────────────────
def fetch_daily_close(sym: str, window_days: int = DEFAULT_WINDOW_DAYS) -> pd.DataFrame:
    """
    Parameters
    ----------
    sym : str
        Ticker (pl. 'AAPL' vagy 'I:SPX')
    window_days : int
        Hány naptári napra kérjük az adatot (alapértelmezett 400)

    Returns
    -------
    pd.DataFrame
        Üres DataFrame, ha hálózati/API-hiba vagy nincs 'results' kulcs.
        Ellenkező esetben ISO-dátum (UTC-aware) + záróár.
    """
    start, end = _date_range(window_days)
    url = BASE.format(sym=sym, start=start, end=end, key=API_KEY)

    try:
        r = sess.get(url, timeout=20)
        r.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"[Fetch] ⚠  {sym}: network/API error → {e}")
        return pd.DataFrame()

    payload = r.json()
    if "results" not in payload:
        print(f"[Fetch] ⚠  {sym}: 'results' missing in response")
        return pd.DataFrame()

    records = [
        {
            "date": dt.datetime.fromtimestamp(item["t"] / 1_000, tz=dt.UTC),
            "close": item["c"],
        }
        for item in payload["results"]
    ]
    return (
        pd.DataFrame(records)
        .sort_values("date")
        .reset_index(drop=True)
    )


# ────────────────────────────────────────────────────────────────────
# Parancssori gyorsteszt
# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Polygon napi záróár letöltő")
    parser.add_argument("symbol", help="Ticker, pl. AAPL vagy I:SPX")
    parser.add_argument("--days", type=int, default=400, help="Naptári napok")
    args = parser.parse_args()

    df_out = fetch_daily_close(args.symbol, args.days)
    print(df_out.tail())