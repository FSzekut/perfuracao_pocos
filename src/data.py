from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class RegionDataset:
    name: str
    features: pd.DataFrame
    target: pd.Series
    row_count: int
    feature_names: list[str]


def load_region_data(path: str | Path) -> RegionDataset:
    csv_path = Path(path)
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates().drop(columns=["id"], errors="ignore")

    return RegionDataset(
        name=csv_path.stem,
        features=df.drop(columns=["product"]),
        target=df["product"],
        row_count=len(df),
        feature_names=df.drop(columns=["product"]).columns.tolist(),
    )
