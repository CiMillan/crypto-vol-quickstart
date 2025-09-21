import pandas as pd
from pathlib import Path

def write_parquet(df: pd.DataFrame, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=True)

def read_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)
