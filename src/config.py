"""Paths and shared experiment settings."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_SPLITS = 3
TARGET_COL = "career_level"
FEATURE_COLS = ["title", "location", "description", "function", "industry"]

# Prefer an explicit path, then a local copy, then the course materials
# on this machine (not committed).
_COURSE_ODS = Path(
    "/workspace/dl/dữ liệu/final_project (khoahochatde.com) (ods)"
    "/final_project (khoahochatde.com).ods"
)

LOCAL_CANDIDATES = [
    DATA_DIR / "final_project.ods",
    DATA_DIR / "job_postings.csv.gz",
    DATA_DIR / "job_postings.csv",
    _COURSE_ODS,
]


def data_path() -> Path:
    env = os.environ.get("JOB_DATA_PATH")
    if env:
        path = Path(env).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"JOB_DATA_PATH does not exist: {path}")
        return path
    for path in LOCAL_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find the job-posting dataset. "
        "Put final_project.ods in data/ or set JOB_DATA_PATH. "
        "See data/README.md."
    )
