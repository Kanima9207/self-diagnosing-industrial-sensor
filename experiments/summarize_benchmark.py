"""Summarize randomized benchmark results by fault class."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


def summarize(path: str = "results/randomized_benchmark.csv") -> None:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    with Path(path).open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            groups[row["fault"]].append(row)

    print("\nRANDOMIZED BENCHMARK SUMMARY")
    print("=" * 76)
    print(f"{'Fault':<10}{'Trials':>8}{'CUSUM alarm':>16}{'False alarm':>16}{'Mean delay':>16}")
    print("-" * 76)

    for fault, rows in groups.items():
        alarm = sum(float(r["cusum_alarm_rate"]) for r in rows) / len(rows)
        far = sum(float(r["false_alarm_rate"]) for r in rows) / len(rows)
        delays = [float(r["detection_delay_s"]) for r in rows if r["detection_delay_s"]]
        delay_text = f"{sum(delays) / len(delays):.2f} s" if delays else "N/A"
        print(f"{fault:<10}{len(rows):>8}{alarm:>15.1%}{far:>15.1%}{delay_text:>16}")


if __name__ == "__main__":
    summarize()
