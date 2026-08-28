"""Cross-process generalization benchmark.

Train a Random Forest on features generated from the nominal process and test
it on a deliberately different process dynamic. The fault classes and sensor
fault mechanisms remain the same; only the underlying process waveform changes.
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

FEATURES = ["mean", "abs_mean", "std", "rms", "peak", "slope", "process_sensor_correlation", "variance_ratio"]
FAULTS = ("healthy", "noisy", "bias", "drift", "stuck")


def shifted_process(t: np.ndarray) -> np.ndarray:
    """Unseen process dynamics used only for evaluation."""
    return 30.0 + 3.8 * np.sin(2 * np.pi * 0.035 * t + 0.4) + 1.2 * np.sin(2 * np.pi * 0.11 * t)


def make_scenario(fault: str, rng: np.random.Generator) -> SensorScenario:
    base = float(rng.uniform(0.08, 0.20))
    if fault == "healthy":
        return SensorScenario(fault, noise_std=base)
    if fault == "noisy":
        return SensorScenario(fault, noise_std=float(rng.uniform(0.55, 1.10)))
    if fault == "bias":
        return SensorScenario(fault, noise_std=base, bias=float(rng.uniform(1.2, 3.0)))
    if fault == "drift":
        return SensorScenario(fault, noise_std=base, drift_rate=float(rng.uniform(0.015, 0.045)))
    if fault == "stuck":
        return SensorScenario(fault, noise_std=base, stuck_at=float(rng.uniform(23.0, 27.0)))
    raise ValueError(fault)


def simulate_shifted(t: np.ndarray, scenario: SensorScenario, seed: int, onset: float):
    rng = np.random.default_rng(seed)
    truth = shifted_process(t)
    measured = truth + rng.normal(0.0, scenario.noise_std, size=t.shape)
    active = t >= onset
    if scenario.bias:
        measured[active] += scenario.bias
    if scenario.drift_rate:
        measured[active] += scenario.drift_rate * (t[active] - onset)
    if scenario.stuck_at is not None:
        measured[active] = scenario.stuck_at
    return truth, measured


def load_training_data(path: str = "results/classification_benchmark.csv"):
    with Path(path).open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    X = np.array([[float(r[name]) for name in FEATURES] for r in rows])
    y = np.array([r["fault"] for r in rows])
    return X, y


def main(trials: int = 100, seed: int = 4242) -> None:
    X_train, y_train = load_training_data()
    model = RandomForestClassifier(n_estimators=300, random_state=seed, class_weight="balanced", n_jobs=-1)
    model.fit(X_train, y_train)

    rng = np.random.default_rng(seed)
    t = np.arange(0.0, 120.0, 0.1)
    onset = 60.0
    start = int(onset / 0.1)
    rows = []
    for fault in FAULTS:
        for trial in range(1, trials + 1):
            scenario = make_scenario(fault, rng)
            truth, measured = simulate_shifted(t, scenario, int(rng.integers(0, 2**32 - 1)), onset)
            residual = measured - truth
            window = slice(start, None) if fault != "healthy" else slice(-len(t[start:]), None)
            features = extract_features(residual[window], measured=measured[window], reference=truth[window])
            rows.append({"fault": fault, "trial": trial, **features})

    X_test = np.array([[float(r[name]) for name in FEATURES] for r in rows])
    y_test = np.array([r["fault"] for r in rows])
    rf_pred = model.predict(X_test)
    rule_pred = np.array([classify_features({name: float(r[name]) for name in FEATURES})[0] for r in rows])
    labels = sorted(np.unique(y_test))

    rf_acc = accuracy_score(y_test, rf_pred)
    rf_f1 = f1_score(y_test, rf_pred, average="macro")
    rule_acc = accuracy_score(y_test, rule_pred)
    rule_f1 = f1_score(y_test, rule_pred, average="macro")

    print("\nCROSS-PROCESS GENERALIZATION BENCHMARK")
    print("=" * 72)
    print("Training: nominal process dynamics")
    print("Testing:  unseen process waveform with same fault mechanisms")
    print(f"Trials:   {trials} per class ({trials * len(FAULTS)} total)")
    print("\nModel comparison")
    print(f"{'Model':<24}{'Accuracy':>14}{'Macro F1':>14}")
    print("-" * 52)
    print(f"{'Random Forest':<24}{rf_acc:>13.1%}{rf_f1:>14.1%}")
    print(f"{'Rule-based classifier':<24}{rule_acc:>13.1%}{rule_f1:>14.1%}")
    print("\nRandom Forest confusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(y_test, rf_pred, labels=labels))
    print("\nRule-based confusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(y_test, rule_pred, labels=labels))

    out = Path("results/cross_process_generalization.txt")
    out.write_text(
        "Cross-process generalization benchmark\n"
        f"Random Forest accuracy: {rf_acc:.6f}\n"
        f"Random Forest macro F1: {rf_f1:.6f}\n"
        f"Rule-based accuracy: {rule_acc:.6f}\n"
        f"Rule-based macro F1: {rule_f1:.6f}\n",
        encoding="utf-8",
    )
    print(f"\nSaved summary to {out}")


if __name__ == "__main__":
    main()
