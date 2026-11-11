"""Load the job-posting table and normalise locations."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.config import FEATURE_COLS, TARGET_COL, data_path

# The original course regex used "\,\s", which triggers a Python
# escape warning. A raw string is the correct form. We also accept
# optional extra spaces before the two-letter state code.
_STATE_AT_END = re.compile(r",\s*([A-Z]{2})$")


def normalize_location(location: object) -> object:
    """Map 'Houston, TX' -> 'TX'; leave other values unchanged.

    The dataset mixes 'City, ST', bare state names, and a few country
    labels. Reducing US city-state strings to the state code cuts the
    cardinality of the one-hot feature without inventing geography.
    """
    if not isinstance(location, str):
        return location
    text = location.strip()
    match = _STATE_AT_END.search(text)
    return match.group(1) if match else text


def load_job_postings(path: Path | None = None) -> pd.DataFrame:
    """Read the ODS/CSV file, drop the single empty description, clean location."""
    path = Path(path) if path is not None else data_path()
    suffix = "".join(path.suffixes).lower()
    if suffix.endswith(".ods"):
        frame = pd.read_excel(path, engine="odf", dtype=str)
    elif suffix.endswith(".csv") or suffix.endswith(".csv.gz"):
        frame = pd.read_csv(path, dtype=str)
    else:
        raise ValueError(f"Unsupported data file: {path}")

    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    frame = frame[FEATURE_COLS + [TARGET_COL]].copy()
    before = len(frame)
    frame = frame.dropna(axis=0).reset_index(drop=True)
    dropped = before - len(frame)
    if dropped:
        print(f"Dropped {dropped} row(s) with missing values ({before} -> {len(frame)}).")

    frame["location"] = frame["location"].map(normalize_location)
    return frame


def split_xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return frame[FEATURE_COLS], frame[TARGET_COL]
