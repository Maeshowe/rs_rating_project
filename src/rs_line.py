from __future__ import annotations

"""
RS Line – Polygon alapú, XAI-kompatibilis kalkuláció
---------------------------------------------------
• RS_line(t) = Close_stock(t) / Close_index(t)  (normalised to 100)
• Index: S&P 500 (ticker: I:SPX); ha nincs adat, SPY ETF-re esünk vissza.
• A részvény- és indexsorozatot naptári nap szerint illesztjük
  `merge_asof`-pal, ±2 nap toleranciával, így az ünnep/UTC-eltolódás
  nem okoz üres metszetet.

Kimenet:  data/rs_line_<TICKER>.json   –  [{date:"2024-06-19", rs_line:102.7}, …]
"""

import datetime as dt
from pathlib import Path
from typing import Optional

import pandas as pd

from fetch import fetch_daily_close

# ───────────────────────────────────────────────────────────────
IDX_TICKER   = "I:SPX"   # elsődleges benchmark
ETF_FALLBACK = "SPY"     # ha index üres
WINDOW_DAYS  = 400       # ≈16 hónap => ≥252 kereskedési nap
TOLERANCE    = "2D"      # merge_asof tolerancia
# ───────────────────────────────────────────────────────────────


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    """
    Előkészíti a DataFrame-et a merge-hez:
      • ha üres → változatlanul visszaadja
      • 'day' oszlop (datetime64[ns], éjfélre normalizálva, UTC)
      • csak ['day', 'close'] marad
    """
    if df.empty:
        return df
    out = df.copy()
    out["day"] = pd.to_datetime(out["date"], utc=True).dt.normalize()
    return out[["day", "close"]]


def compute_rs_line(sym: str, win: int = WINDOW_DAYS) -> Optional[pd.DataFrame]:
    """None, ha nincs közös nap±2 között."""
    df_s = _prep(fetch_daily_close(sym, win))
    df_i = _prep(fetch_daily_close(IDX_TICKER, win))
    if df_i.empty:
        df_i = _prep(fetch_daily_close(ETF_FALLBACK, win))

    if df_s.empty or df_i.empty:
        return None

    df_s.sort_values("day", inplace=True)
    df_i.sort_values("day", inplace=True)

    df = pd.merge_asof(
        df_s,
        df_i,
        on="day",
        tolerance=pd.Timedelta(TOLERANCE),
        direction="nearest",
        suffixes=("_stk", "_idx"),
    ).dropna()

    if df.empty:
        return None

    rel = df["close_stk"] / df["close_idx"]
    rs_norm = rel / rel.iloc[0] * 100
    return pd.DataFrame({"date": df["day"], "rs_line": rs_norm})


def save_rs_line_json(sym: str, out_dir: Path) -> None:
    """Írja a data/rs_line_<sym>.json fájlt (ha van adat)."""
    rs = compute_rs_line(sym)
    if rs is None:
        print(f"[RS-Line] ⚠  No overlapping data for {sym}, skipped")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"rs_line_{sym}.json").write_text(
    rs.to_json(orient="records", date_format="iso", date_unit="s")
)