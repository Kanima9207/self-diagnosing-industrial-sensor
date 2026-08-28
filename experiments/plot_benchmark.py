"""Generate publication-style benchmark plots from CSV outputs."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def read_rows(path: str) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    out = Path("results")
    out.mkdir(exist_ok=True)

    rows = read_rows("results/randomized_benchmark.csv")
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["fault"]].append(row)

    faults = list(groups)
    detection = []
    false_alarm = []
    delay = []
    for fault in faults:
        values = groups[fault]
        detection.append(sum(float(r["detection_delay_s"] != "") for r in values) / len(values) * 100)
        false_alarm.append(sum(float(r["false_alarm_rate"]) for r in values) / len(values) * 100)
        delays = [float(r["detection_delay_s"]) for r in values if r["detection_delay_s"]]
        delay.append(sum(delays) / len(delays) if delays else 0.0)

    for values, ylabel, filename, title in [
        (detection, "Detection rate (%)", "detection_rate.png", "CUSUM Detection Rate"),
        (false_alarm, "False-alarm rate (%)", "false_alarm_rate.png", "Healthy-interval False Alarm Rate"),
        (delay, "Mean detection delay (s)", "detection_delay.png", "Mean Detection Delay"),
    ]:
        plt.figure(figsize=(9, 5))
        plt.bar(faults, values)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out / filename, dpi=180)
        plt.close()

    classification = read_rows("results/classification_benchmark.csv")
    cls_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in classification:
        cls_groups[row["fault"]].append(row)
    cls_faults = list(cls_groups)
    accuracy = [100 * sum(int(r["correct"]) for r in cls_groups[f]) / len(cls_groups[f]) for f in cls_faults]

    plt.figure(figsize=(9, 5))
    plt.bar(cls_faults, accuracy)
    plt.ylim(0, 105)
    plt.ylabel("Classification accuracy (%)")
    plt.title("Interpretable Fault Classification Accuracy")
    plt.tight_layout()
    plt.savefig(out / "classification_accuracy.png", dpi=180)
    plt.close()

    print(f"Saved benchmark plots to {out}")


if __name__ == "__main__":
    main()
