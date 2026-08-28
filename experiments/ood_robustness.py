"""Out-of-distribution robustness benchmark for sensor-fault diagnosis.

The Random Forest is trained on the existing in-distribution dataset, then
frozen. It is evaluated on deliberately shifted synthetic conditions whose
fault magnitudes/noise/process dynamics are outside the training ranges.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src.fault_classifier import classify_features
from src.feature_extraction import extract_features
from src.signal_generator import SensorScenario, simulate_sensor

FEATURES = [
    "mean", "abs_mean", "std", "rms", "peak", "slope",
    "process_sensor_correlation", "variance_ratio",
]
FAULTS = ("healthy", "noisy", "bias", "drift", "stuck")


def load_training_data(path: str = "results/classification_benchmark.csv"):
    with Path(path).open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    X = np.array([[float(r[name]) for name in FEATURES] for r in rows], dtype=float)
    y = np.array([r["fault"] for r in rows])
    return X, y


def shifted_scenario(fault: str, rng: np.random.Generator) -> SensorScenario:
    """Create conditions outside the ranges used by the training generator."""
    base = float(rng.uniform(0.22, 0.35))
    if fault == "healthy":
        return SensorScenario(fault, noise_std=base)
    if fault == "noisy":
        return SensorScenario(fault, noise_std=float(rng.uniform(1.15, 1.60)))
    if fault == "bias":
        return SensorScenario(fault, noise_std=base, bias=float(rng.uniform(3.2, 4.8)))
    if fault == "drift":
        return SensorScenario(fault, noise_std=base, drift_rate=float(rng.uniform(0.050, 0.080)))
    if fault == "stuck":
        return SensorScenario(fault, noise_std=base, stuck_at=float(rng.uniform(28.0, 32.0)))
    raise ValueError(fault)


def build_ood_dataset(trials: int = 100, seed: int = 9090):
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, 120.0, 0.1)
    onset = 60.0
    start = int(onset / 0.1)
    rows = []
    for fault in FAULTS:
        for trial in range(1, trials + 1):
            scenario = shifted_scenario(fault, rng)
            truth, measured = simulate_sensor(
                t, scenario, seed=int(rng.integers(0, 2**32 - 1)),
                fault_start_time=None if fault == "healthy" else onset,
            )
            residual = measured - truth
            window = slice(start, None) if fault != "healthy" else slice(-len(t[start:]), None)
            features = extract_features(
                residual[window], measured=measured[window], reference=truth[window]
            )
            rows.append({"fault": fault, "trial": trial, **features})
    return rows


def main(seed: int = 2026) -> None:
    X_train, y_train = load_training_data()
    model = RandomForestClassifier(
        n_estimators=300, random_state=seed, class_weight="balanced", n_jobs=-1
    )
    model.fit(X_train, y_train)

    rows = build_ood_dataset()
    X_ood = np.array([[float(r[name]) for name in FEATURES] for r in rows], dtype=float)
    y_ood = np.array([r["fault"] for r in rows])
    rf_pred = model.predict(X_ood)

    rule_pred = np.array([
        classify_features({name: float(r[name]) for name in FEATURES}) for r in rows
    ], dtype=object)
    rule_labels = np.array([p[0] for p in rule_pred])

    rf_acc = accuracy_score(y_ood, rf_pred)
    rf_f1 = f1_score(y_ood, rf_pred, average="macro")
    rule_acc = accuracy_score(y_ood, rule_labels)
    rule_f1 = f1_score(y_ood, rule_labels, average="macro")
    labels = sorted(np.unique(y_ood))

    print("\nOUT-OF-DISTRIBUTION ROBUSTNESS BENCHMARK")
    print("=" * 72)
    print("Training: existing in-distribution synthetic dataset")
    print("Testing:  shifted fault magnitudes and noise levels")
    print("Trials:   100 per class (500 total)")
    print("\nModel comparison")
    print(f"{'Model':<24}{'Accuracy':>14}{'Macro F1':>14}")
    print("-" * 52)
    print(f"{'Random Forest':<24}{rf_acc:>13.1%}{rf_f1:>14.1%}")
    print(f"{'Rule-based classifier':<24}{rule_acc:>13.1%}{rule_f1:>14.1%}")
    print("\nRandom Forest confusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(y_ood, rf_pred, labels=labels))
    print("\nRule-based confusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(y_ood, rule_labels, labels=labels))

    out = Path("results/ood_robustness.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["fault", "trial", "rf_pred", "rule_pred", *FEATURES])
        writer.writeheader()
        for row, rf, rule in zip(rows, rf_pred, rule_labels):
            writer.writerow({"fault": row["fault"], "trial": row["trial"], "rf_pred": rf, "rule_pred": rule, **{k: row[k] for k in FEATURES}})

    summary = Path("results/ood_robustness.txt")
    summary.write_text(
        "Out-of-distribution robustness benchmark\n"
        f"Random Forest accuracy: {rf_acc:.6f}\n"
        f"Random Forest macro F1: {rf_f1:.6f}\n"
        f"Rule-based accuracy: {rule_acc:.6f}\n"
        f"Rule-based macro F1: {rule_f1:.6f}\n",
        encoding="utf-8",
    )
    print(f"\nSaved trial data to {out}")
    print(f"Saved summary to {summary}")


if __name__ == "__main__":
    main()
