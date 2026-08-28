# Out-of-Distribution Robustness Benchmark

This experiment tests whether the diagnostic models generalize beyond the synthetic conditions used to construct the training dataset.

## Protocol

The Random Forest is trained only on `results/classification_benchmark.csv` and is then frozen. A separate 500-trial dataset is generated with deliberately shifted conditions:

- Healthy noise: 0.22–0.35 instead of 0.08–0.20
- Noisy sensor noise: 1.15–1.60 instead of 0.55–1.10
- Bias: 3.2–4.8 instead of 1.2–3.0
- Drift rate: 0.050–0.080 instead of 0.015–0.045
- Stuck value: 28–32 instead of 23–27

The same controlled 60 s fault onset and post-onset feature window are used. The benchmark compares the frozen Random Forest with the interpretable rule-based classifier.

## Run

```powershell
python -m experiments.ood_robustness
```

Outputs:

- `results/ood_robustness.csv` — trial-level predictions and features
- `results/ood_robustness.txt` — accuracy and macro-F1 summary

## Interpretation

This is a synthetic distribution-shift test, not real-world validation. A performance drop is useful evidence of model sensitivity to distribution shift; stable performance provides evidence of robustness within the simulated fault family.
