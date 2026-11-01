"""
Data loading helpers for the ASRS pipeline.

The cleaned data (train.csv / val.csv / test.csv) produced by
prepare_dataset.py carries the fixed header ``acn,text,label,label_id`` and is
already stratified and leakage-free. These helpers load it back into pandas and
HuggingFace structures, keeping a single deterministic path across training and
evaluation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config


def load_split(csv_path: Path) -> pd.DataFrame:
    """Load one cleaned split as a DataFrame and validate its columns."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing data file: {path}. Run `scripts/01_prepare.py` or place "
            "the cleaned CSVs (train.csv/val.csv/test.csv) in the data/ folder."
        )
    df = pd.read_csv(path, dtype={"acn": object})
    needed = {"acn", "text", "label", "label_id"}
    if not needed.issubset(df.columns):
        raise ValueError(f"{path.name} must contain columns {sorted(needed)}")
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
    df["label_id"] = df["label_id"].astype(int)
    return df


def load_all_splits(data_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Return {train, val, test} DataFrames in one call."""
    base = Path(data_dir) if data_dir else config.DATA_DIR
    return {
        "train": load_split(base / "train.csv"),
        "val": load_split(base / "val.csv"),
        "test": load_split(base / "test.csv"),
    }


def class_counts(df: pd.DataFrame) -> pd.Series:
    """Return per-class sample counts (indexed by label_id)."""
    return df["label_id"].value_counts().sort_index()


def class_weights_from(df: pd.DataFrame, smoothing: float = 1.0,
                       power: float = 1.0) -> np.ndarray:
    """
    Compute per-class loss weights as inverse relative frequency.

    ``power`` sharpens the weighting: >1 up-weights rare classes (and
    down-weights the majority class) relative to the standard inverse
    frequency, which helps when the majority class acts as a "catch-all".

    Weights are length ``len(LABELS)``; classes absent from the frame get the
    average weight so the tensor stays a valid CrossEntropyLoss weight vector.
    """
    counts = class_counts(df)
    rel = counts / float(counts.sum())
    weights = (1.0 / (rel + smoothing)) ** power
    weights = weights / weights.sum() * len(rel)
    full = np.ones(len(config.LABELS), dtype=np.float64) * float(weights.mean())
    for label_id in counts.index:
        full[int(label_id)] = weights[int(label_id)]
    return full


def to_hf_dataset(df: pd.DataFrame):
    """Convert a pandas DataFrame to a HuggingFace Dataset (optional import)."""
    from datasets import Dataset

    return Dataset.from_pandas(df[["text", "label", "label_id"]])


def tokenise_batch(tokenizer, batch: dict, max_len: int | None = None) -> dict:
    """Tokenise a HuggingFace batch for the sequence classifier."""
    max_len = max_len or config.MAX_LEN
    out = tokenizer(
        batch["text"], truncation=True, padding="max_length", max_length=max_len
    )
    out["labels"] = batch["label_id"]
    return out