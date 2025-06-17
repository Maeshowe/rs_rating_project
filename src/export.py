# src/export.py
import json, pandas as pd, logging
from pathlib import Path

Path("data").mkdir(exist_ok=True)
logging.basicConfig(filename="logs/run.log",
                    level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

def dump(df: pd.DataFrame):
    df.to_json("data/rs_snapshot.json", orient="records", indent=2)
    df.to_csv("data/rs_snapshot.csv", index=False)
    logging.info("Export → data/ (rows=%s)", len(df))