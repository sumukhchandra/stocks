"""
Parquet Store: Thread-safe high-speed columnar data storage.
"""

import os
import pandas as pd
from typing import Optional


class ParquetStore:
    def __init__(self, base_dir: Optional[str] = None):
        if base_dir is None:
            # Default to project data directory
            root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
            base_dir = os.path.join(root, "data", "processed")
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def save_dataset(self, df: pd.DataFrame, filename: str = "master_labeled_dataset.parquet") -> str:
        filepath = os.path.join(self.base_dir, filename)
        df.to_parquet(filepath, index=False, engine="pyarrow", compression="snappy")
        return filepath

    def load_dataset(self, filename: str = "master_labeled_dataset.parquet") -> pd.DataFrame:
        filepath = os.path.join(self.base_dir, filename)
        if not os.path.exists(filepath):
            return pd.DataFrame()
        return pd.read_parquet(filepath, engine="pyarrow")

    def exists(self, filename: str = "master_labeled_dataset.parquet") -> bool:
        return os.path.exists(os.path.join(self.base_dir, filename))
