# src/export.py
import json
import pandas as pd
import logging
from pathlib import Path

# Projekt root (az src mappa fölötti könyvtár)
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
LOGS_DIR = ROOT / "logs"

# Mappák létrehozása, ha hiányoznak
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Logging beállítás
logging.basicConfig(
    filename=str(LOGS_DIR / "run.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

def dump(df: pd.DataFrame):
    # Export JSON és CSV fájlba
    df.to_json(DATA_DIR / "rs_snapshot.json", orient="records", indent=2)
    df.to_csv(DATA_DIR / "rs_snapshot.csv", index=False)
    logging.info("Export → data/ (rows=%s)", len(df))