# Randomized Benchmark

The benchmark runs 100 independent trials for each of five sensor conditions, for 500 trials total.

## Randomization

Each trial varies the sensor noise level and, where applicable, fault magnitude. A fixed master seed (`2026`) makes the benchmark reproducible while still generating independent trial conditions.

## Recorded metrics

- CUSUM alarm rate
- False-alarm rate during the healthy/reference interval
- Detection delay when a persistent alarm is observed
- Actual randomized fault parameters

## Run

From the repository root:

```powershell
python -m experiments.randomized_benchmark
python -m experiments.summarize_benchmark
```

The raw trial-level results are written to `results/randomized_benchmark.csv`.

## Important validation note

The current Phase 3 benchmark infrastructure randomizes fault parameters, but the simulator still applies a selected scenario over the complete record. Consequently, detection-delay values are not yet a final claim of fault-onset performance. The next validation step will add explicit fault onset at a known time (for example 60 s), then compute true detection delay, false alarms, and missed detections against that ground truth.
