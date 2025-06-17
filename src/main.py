"""
RS-rating pipeline – Python 3.11+

1. Lekéri a napi záróárakat (Polygon)
2. Kiszámítja az RS-faktort
3. Szektoronként 1–99 rangsor
4. JSON/CSV export – minden rekord tartalmazza az `as_of` UTC-időbélyeget
"""

import json
import datetime as dt
from pathlib import Path
import concurrent.futures as fut

import pandas as pd
from fetch import fetch_daily_close
from calculate import calculate_rs_factor, percentile_rank
from export import dump

# ---------------------------------------------------------------------------
SECTORS = json.loads(Path("config/sectors.json").read_text())
AS_OF = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"  # » 2025-06-17T19:17:00Z
MAX_WORKERS = 6
# ---------------------------------------------------------------------------


def process_sector(sector: str, tickers: list[str]) -> list[dict]:
    """Visszaadja az adott szektor rekordjait (dict-ek)."""
    factors: dict[str, float] = {}
    for t in tickers:
        df = fetch_daily_close(t)
        if len(df) >= 252:
            factors[t] = calculate_rs_factor(df)

    rf = pd.Series(factors, name="rs_factor")
    rr = percentile_rank(rf)

    return [
        {
            "sector": sector,
            "ticker": t,
            "rs_factor": round(rf[t], 2),
            "rs_rating": int(rr[t]),
            "as_of": AS_OF,  # ✨ minden rekord ugyanazt az UTC-időt kapja
        }
        for t in rf.index
    ]


def main():
    rows: list[dict] = []

    with fut.ThreadPoolExecutor(MAX_WORKERS) as ex:
        futures = [
            ex.submit(process_sector, sector, tickers)
            for sector, tickers in SECTORS.items()
        ]
        for f in fut.as_completed(futures):
            rows.extend(f.result())

    # DataFrame-ből export (JSON+CSV)
    dump(pd.DataFrame(rows))


if __name__ == "__main__":
    main()