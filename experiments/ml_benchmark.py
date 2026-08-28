"""Compare a Random Forest classifier with the interpretable baseline."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

FEATURES = [
    "mean", "abs_mean", "std", "rms", "peak", "slope",
    "process_sensor_correlation", "variance_ratio",
]


def load_dataset(path: str = "results/classification_benchmark.csv") -> tuple[np.ndarray, np.ndarray]:
    with Path(path).open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    X = np.array([[float(row[name]) for name in FEATURES] for row in rows], dtype=float)
    y = np.array([row["fault"] for row in rows])
    return X, y


def main(seed: int = 2026) -> None:
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=seed,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)

    labels = sorted(np.unique(y))
    accuracy = accuracy_score(y_test, predicted)
    precision = precision_score(y_test, predicted, average="macro", zero_division=0)
    recall = recall_score(y_test, predicted, average="macro", zero_division=0)
    f1 = f1_score(y_test, predicted, average="macro", zero_division=0)

    print("\nRANDOM FOREST BENCHMARK")
    print("=" * 60)
    print(f"Train samples: {len(y_train)}")
    print(f"Test samples:  {len(y_test)}")
    print(f"Accuracy:      {accuracy:.1%}")
    print(f"Macro precision: {precision:.1%}")
    print(f"Macro recall:    {recall:.1%}")
    print(f"Macro F1:        {f1:.1%}")
    print("\nClassification report:")
    print(classification_report(y_test, predicted, labels=labels, zero_division=0))
    print("Confusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(y_test, predicted, labels=labels))
    print("\nFeature importance:")
    for name, importance in sorted(zip(FEATURES, model.feature_importances_), key=lambda p: p[1], reverse=True):
        print(f"{name:<30} {importance:.3f}")

    out = Path("results/ml_benchmark.txt")
    out.write_text(
        f"Random Forest benchmark\nAccuracy: {accuracy:.6f}\nMacro precision: {precision:.6f}\n"
        f"Macro recall: {recall:.6f}\nMacro F1: {f1:.6f}\n",
        encoding="utf-8",
    )
    print(f"\nSaved summary to {out}")


if __name__ == "__main__":
    main()
