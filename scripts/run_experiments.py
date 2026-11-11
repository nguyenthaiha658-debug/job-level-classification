#!/usr/bin/env python3
"""Train every experiment, write a results table and the best confusion matrix."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE, RESULTS_DIR, TARGET_COL, TEST_SIZE
from src.data import load_job_postings, split_xy
from src.evaluate import (
    compute_metrics,
    metrics_table,
    plot_confusion_matrix,
    report_text,
)
from src.train import experiment_specs, fit_experiment


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    reports_dir = RESULTS_DIR / "classification_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    frame = load_job_postings()
    x, y = split_xy(frame)
    labels = list(y.value_counts().index)  # most frequent first, stable for plots

    print("Class distribution:")
    print(y.value_counts().to_string())
    print()

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"Train: {len(x_train)}  Test: {len(x_test)}")
    print(f"Train class counts:\n{y_train.value_counts().to_string()}\n")

    rows = []
    predictions = {}

    for spec in experiment_specs():
        name = spec["name"]
        print("=" * 72)
        print(f"Running {name}  ({spec['model']}, {spec['handling']})")
        t0 = time.perf_counter()
        search = fit_experiment(spec, x_train, y_train)
        elapsed = time.perf_counter() - t0
        y_pred = search.predict(x_test)
        metrics = compute_metrics(y_test, y_pred)
        report = report_text(y_test, y_pred, labels=labels)
        (reports_dir / f"{name}.txt").write_text(report, encoding="utf-8")
        predictions[name] = y_pred
        best_params = {k: (int(v) if hasattr(v, "item") else v) for k, v in search.best_params_.items()}
        row = {
            "experiment": name,
            "model": spec["model"],
            "handling": spec["handling"],
            "accuracy": round(metrics["accuracy"], 4),
            "macro_f1": round(metrics["macro_f1"], 4),
            "weighted_f1": round(metrics["weighted_f1"], 4),
            "best_params": json.dumps(best_params, sort_keys=True),
            "cv_macro_f1": round(float(search.best_score_), 4),
            "fit_seconds": round(elapsed, 1),
        }
        rows.append(row)
        print(report)
        print(
            f"test accuracy={row['accuracy']:.4f}  "
            f"macro_f1={row['macro_f1']:.4f}  "
            f"weighted_f1={row['weighted_f1']:.4f}  "
            f"cv_macro_f1={row['cv_macro_f1']:.4f}  "
            f"params={row['best_params']}  "
            f"({row['fit_seconds']}s)"
        )

    table = metrics_table(rows)
    csv_path = RESULTS_DIR / "metrics.csv"
    table.to_csv(csv_path, index=False)
    print("\n" + "=" * 72)
    print(table.to_string(index=False))
    print(f"\nWrote {csv_path}")

    best = max(rows, key=lambda r: (r["macro_f1"], r["weighted_f1"], r["accuracy"]))
    best_name = best["experiment"]
    cm_path = RESULTS_DIR / "confusion_matrix.png"
    plot_confusion_matrix(
        y_test,
        predictions[best_name],
        labels=labels,
        path=cm_path,
        title=f"Confusion matrix — {best_name} (macro F1 = {best['macro_f1']:.3f})",
    )
    print(f"Best by test macro-F1: {best_name} -> {cm_path}")

    summary = {
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "class_counts": {k: int(v) for k, v in y.value_counts().items()},
        "best_experiment": best_name,
        "metrics": rows,
    }
    (RESULTS_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
