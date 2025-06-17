"""
fetch.py – Polygon napi záróár letöltés timezone-aware dátummal (Python ≥3.12)
"""
import os, datetime as dt
from pathlib import Path
import pandas as pd, requests
from dotenv import load_dotenv
from tqdm import tqdm  # opcionális

# .env betöltése
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

API_KEY = os.getenv("POLYGON_API_KEY")
if not API_KEY:
    raise EnvironmentError("POLYGON_API_KEY hiányzik (.env)!")

BASE = (
    "https://api.polygon.io/v2/aggs/ticker/{sym}/range/1/day/"
    "{start}/{end}?unadjusted=false&apiKey={key}"
)
DEFAULT_WINDOW_DAYS = 420

def _date_range(days=DEFAULT_WINDOW_DAYS):
    end = dt.date.today()
    start = end - dt.timedelta(days=days)
    return start.isoformat(), end.isoformat()

def fetch_daily_close(sym: str, window_days: int = DEFAULT_WINDOW_DAYS) -> pd.DataFrame:
    start, end = _date_range(window_days)
    url = BASE.format(sym=sym, start=start, end=end, key=API_KEY)
    r = requests.get(url, timeout=20)
    r.raise_for_status()

    payload = r.json()
    if "results" not in payload:
        raise ValueError(f"Váratlan Polygon-válasz: {payload}")

    # ⚠️  Deprecation-safe átalakítás
    records = [
        {
            "date": dt.datetime.fromtimestamp(item["t"] / 1_000, tz=dt.UTC),
            "close": item["c"],
        }
        for item in payload["results"]
    ]
    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    return df

# Gyors CLI-teszt
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("symbol"), parser.add_argument("--days", type=int, default=400)
    args = parser.parse_args()
    print(fetch_daily_close(args.symbol, args.days).tail())