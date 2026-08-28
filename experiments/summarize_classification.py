"""Print classification accuracy by fault class."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


def main(path: str = "results/classification_benchmark.csv") -> None:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    with Path(path).open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            groups[row["fault"]].append(row)

    print("\nFAULT CLASSIFICATION SUMMARY")
    print("=" * 56)
    print(f"{'Fault':<12}{'Trials':>10}{'Accuracy':>14}{'Mean confidence':>20}")
    print("-" * 56)
    total_correct = 0
    total = 0
    for fault, rows in groups.items():
        correct = sum(int(r["correct"]) for r in rows)
        confidence = sum(float(r["confidence"]) for r in rows) / len(rows)
        total_correct += correct
        total += len(rows)
        print(f"{fault:<12}{len(rows):>10}{correct / len(rows):>13.1%}{confidence:>19.1%}")
    print("-" * 56)
    print(f"{'Overall':<12}{total:>10}{total_correct / total:>13.1%}")


if __name__ == "__main__":
    main()
