# Self-Diagnosing Industrial Sensor

> A research-oriented instrumentation and machine-learning system for detecting and classifying industrial sensor faults from process measurements.

**Status: Portfolio-ready software prototype**

## Overview

Industrial sensors can degrade through increased noise, bias, drift, or frozen outputs. This project builds an end-to-end diagnostic pipeline that separates the process signal from sensor behavior, detects abnormality with statistical signal processing, extracts interpretable health features, and compares a transparent rule-based classifier with a Random Forest model.

The system is deliberately evaluated under randomized trials, shifted fault severity, and an unseen process waveform rather than relying on a single hand-picked example.

## Key Results

| Experiment | Random Forest | Rule-based |
|---|---:|---:|
| In-distribution classification (500 trials) | **100.0%** | **100.0%** |
| OOD fault-severity benchmark (500 trials) | **100.0%** | 80.2% |
| Cross-process generalization (500 trials) | **100.0%** | **100.0%** |

CUSUM randomized benchmark:

| Fault | Detection rate | False alarm rate | Mean delay |
|---|---:|---:|---:|
| Healthy | 0.1% alarm | 0.1% | N/A |
| Noisy | 97.9% | 0.1% | 0.06 s |
| Bias | 99.9% | 0.2% | 0.04 s |
| Drift | 91.9% | 0.1% | 4.88 s |
| Stuck | 99.3% | 0.1% | 0.37 s |

> Results are from the project's synthetic simulator and should not be interpreted as validation on real industrial hardware.

## Fault Classes

1. Healthy sensor
2. Increased-noise sensor
3. Biased sensor
4. Gradually drifting sensor
5. Stuck/frozen sensor

## Architecture

```text
                 PROCESS MODEL
                      |
                      v
               SENSOR SIMULATOR
                      |
                FAULT INJECTION
                      |
                      v
              SIGNAL PROCESSING
                      |
                      v
                   RESIDUAL
                      |
             +--------+--------+
             |                 |
             v                 v
           CUSUM       FEATURE EXTRACTION
             |                 |
             |       +---------+----------+
             |       |         |          |
             |      Level     Trend   Sensor response
             |       |         |          |
             +-------+---------+----------+
                     |
             +-------+--------+
             |                |
             v                v
       RULE-BASED       RANDOM FOREST
       CLASSIFIER          MODEL
             |                |
             +-------+--------+
                     |
                     v
             DIAGNOSTIC RESULT
                     |
                     v
            STREAMLIT DASHBOARD
```

## Diagnostic Features

The classifier uses interpretable signal and sensor-response features:

- Mean and absolute mean residual
- Standard deviation
- RMS and peak residual
- Residual slope
- Process-to-sensor correlation
- Sensor/process variance ratio

The correlation and variance-ratio features are particularly useful for identifying a sensor that has stopped responding to process variation.

## Repository Structure

```text
self-diagnosing-industrial-sensor/
├── app.py
├── README.md
├── requirements.txt
├── requirements-dashboard.txt
├── .gitignore
├── src/
│   ├── signal_generator.py
│   ├── signal_processing.py
│   ├── fault_detection.py
│   ├── feature_extraction.py
│   ├── fault_classifier.py
│   └── main.py
├── experiments/
│   ├── healthy.py
│   ├── noisy.py
│   ├── bias.py
│   ├── drift.py
│   ├── stuck_sensor.py
│   ├── benchmark_faults.py
│   ├── classification_benchmark.py
│   ├── ml_benchmark.py
│   ├── plot_benchmark.py
│   ├── ood_robustness.py
│   └── cross_process_generalization.py
├── tests/
│   └── test_signal_generator.py
├── results/
│   ├── diagnostic_metrics.csv
│   ├── randomized_benchmark.csv
│   ├── classification_benchmark.csv
│   ├── ood_robustness.csv
│   ├── ml_benchmark.txt
│   └── cross_process_generalization.txt
└── docs/
    ├── methodology.md
    ├── dashboard.md
    ├── ood_robustness.md
    └── cross_process_generalization.md
```

## Reproducible Experiments

From the repository root:

```powershell
pytest
python -m experiments.benchmark_faults
python -m experiments.classification_benchmark
python -m experiments.ml_benchmark
python -m experiments.plot_benchmark
python -m experiments.ood_robustness
python -m experiments.cross_process_generalization
```

The experiments write quantitative results and plots to `results/`.

## Interactive Dashboard

Install the dashboard dependencies:

```powershell
pip install -r requirements-dashboard.txt
```

Launch:

```powershell
streamlit run app.py
```

The dashboard provides fault selection, severity/noise controls, raw and residual signal plots, extracted diagnostic features, and an automatic diagnosis.

## Validation Philosophy

The project intentionally separates three questions:

1. **Can abnormal sensor behavior be detected?** — CUSUM benchmark.
2. **Can different fault mechanisms be distinguished?** — rule-based and Random Forest classification.
3. **Does the classifier generalize beyond its training conditions?** — fault-severity shift and cross-process tests.

The results are synthetic validation. Future work should include real sensor datasets, hardware-in-the-loop testing, uncertainty calibration, and deployment on an embedded target.

## Engineering Stack

**Python · NumPy · SciPy · Matplotlib · scikit-learn · pytest · Streamlit · Git/GitHub**

## Future Work

- Real industrial sensor dataset validation
- Hardware-in-the-loop experiments
- Online/streaming inference
- Model uncertainty and calibrated confidence
- Embedded deployment on a microcontroller
- Multisensor fault diagnosis

## Author

**Dibyo** — Instrumentation Engineering
