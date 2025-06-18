"""
main.py
~~~~~~~~
Pipeline a napi RS-rating és RS-line frissítéséhez.

Lépések:
1) Lekéri a napi záróárakat (Polygon) minden tickerre
2) Kiszámítja az RS-faktort és a 1–99 rangpontot szektoronként
3) Ment:  data/rs_snapshot.json  +  data/rs_snapshot.csv
4) Minden feldolgozott tickerre RS-line JSON-t ír:  data/rs_line_<T>.json
"""

from __future__ import annotations

import json
import datetime as dt
import concurrent.futures as fut
from pathlib import Path

import pandas as pd

from fetch import fetch_daily_close
from calculate import calculate_rs_factor, percentile_rank
from export import dump
from rs_line import save_rs_line_json

# ────────────────────────────────────────────────────────────────────
# Beállítások
# ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SECTORS = json.loads((PROJECT_ROOT / "config" / "sectors.json").read_text())
AS_OF   = dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()  # ISO-UTC, pl. 2025-06-18T19:17:00+00:00
MAX_WORKERS = 6
DATA_DIR = PROJECT_ROOT / "data"
# ────────────────────────────────────────────────────────────────────


def process_sector(sector: str, tickers: list[str]) -> list[dict]:
    """RS-faktor + RS-rating számítás egy szektorra; üres DF esetén skip."""
    factors: dict[str, float] = {}
    for t in tickers:
        df = fetch_daily_close(t)
        if df.empty or len(df) < 252:               # IPO vagy hálózati hiba
            continue
        factors[t] = calculate_rs_factor(df)

    # ha egy ticker sem maradt, térjen vissza üres listával
    if not factors:
        return []

    rf = pd.Series(factors, name="rs_factor")
    rr = percentile_rank(rf)

    return [
        {
            "sector":    sector,
            "ticker":    t,
            "rs_factor": round(rf[t], 2),
            "rs_rating": int(rr[t]),
            "as_of":     AS_OF,
        }
        for t in rf.index
    ]


def main() -> None:
    rows: list[dict] = []

    # 1) RS-rating számítás párhuzamosan
    with fut.ThreadPoolExecutor(MAX_WORKERS) as ex:
        tasks = [ex.submit(process_sector, s, ts) for s, ts in SECTORS.items()]
        for task in fut.as_completed(tasks):
            rows.extend(task.result())

    # 2) Export JSON + CSV
    dump(pd.DataFrame(rows))

    # 3) RS-line JSON minden feldolgozott tickerre
    tickers = {r["ticker"] for r in rows}
    for t in tickers:
        save_rs_line_json(t, DATA_DIR)   # (üres DF esetén skip + console-warning)


if __name__ == "__main__":
    main()