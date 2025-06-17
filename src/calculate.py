# src/calculate.py
import pandas as pd

def roc(df: pd.DataFrame, days: int) -> float:
    """n-napos százalékos változás."""
    return (df["close"].iloc[-1] / df["close"].iloc[-days] - 1) * 100

def calculate_rs_factor(df: pd.DataFrame) -> float:
    """IBD-súlyozás: 0.4*63d + 0.2*126d + 0.2*189d + 0.2*252d."""
    return 0.4*roc(df,63) + 0.2*roc(df,126) + 0.2*roc(df,189) + 0.2*roc(df,252)

def percentile_rank(series: pd.Series) -> pd.Series:
    """Szektoron belüli 1–99 rangsor (`rank(pct=True)` → 1-99 skála)."""
    return (series.rank(pct=True) * 98 + 1).round().astype(int)