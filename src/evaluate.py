"""Test-set metrics and confusion-matrix plot."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)

# Short labels for the confusion-matrix plot only.
DISPLAY_NAMES = {
    "senior_specialist_or_project_manager": "Senior specialist",
    "manager_team_leader": "Manager / team lead",
    "bereichsleiter": "Bereichsleiter",
    "director_business_unit_leader": "Director / BU lead",
    "specialist": "Specialist",
    "managing_director_small_medium_company": "MD (SME)",
}


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def report_text(y_true, y_pred, labels) -> str:
    return classification_report(
        y_true, y_pred, labels=labels, digits=3, zero_division=0
    )


def plot_confusion_matrix(y_true, y_pred, labels: list[str], path: Path, title: str) -> None:
    display_labels = [DISPLAY_NAMES.get(label, label) for label in labels]
    fig, ax = plt.subplots(figsize=(8.5, 7))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=labels,
        display_labels=display_labels,
        cmap="Blues",
        colorbar=True,
        ax=ax,
        xticks_rotation=35,
    )
    ax.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def metrics_table(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    preferred = [
        "experiment",
        "model",
        "handling",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "best_params",
        "cv_macro_f1",
        "fit_seconds",
    ]
    cols = [c for c in preferred if c in frame.columns] + [
        c for c in frame.columns if c not in preferred
    ]
    return frame[cols]
