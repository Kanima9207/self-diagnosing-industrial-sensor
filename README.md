# Self-Diagnosing Industrial Sensor

> A research-oriented instrumentation system that estimates sensor health while measuring a process variable.

## Why this project?

Industrial measurement systems can fail gradually through noise increase, bias, drift, or frozen outputs. This project develops an interpretable diagnostic pipeline that detects those conditions before adding machine-learning models.

## Core Model

The simulated measurement is represented as:

**y(t) = x(t) + n(t) + b(t) + d(t)**

- `x(t)` — true process signal
- `n(t)` — measurement noise
- `b(t)` — sensor bias
- `d(t)` — sensor drift

## Fault Classes

1. Healthy sensor
2. Increased-noise sensor
3. Biased sensor
4. Gradually drifting sensor
5. Stuck/frozen sensor

## System Architecture

```text
Process Model
     |
     v
Sensor Simulator ----> Fault Injection
     |                      |
     +----------+-----------+
                v
       Signal Processing
                |
                v
       Feature Extraction
                |
                v
        Fault Detection
                |
                v
        Sensor Health
                |
                v
       Results / Dashboard
```

## Repository Structure

```text
self-diagnosing-industrial-sensor/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── signal_generator.py
│   ├── signal_processing.py
│   ├── fault_detection.py
│   └── main.py
├── experiments/
│   ├── healthy.py
│   ├── noisy.py
│   ├── bias.py
│   ├── drift.py
│   └── stuck_sensor.py
├── tests/
│   └── test_signal_generator.py
├── results/
│   └── .gitkeep
└── docs/
    └── methodology.md
```

## Development Roadmap

- [x] Project architecture and documentation
- [x] Synthetic sensor/process model
- [x] Fault injection framework
- [x] Signal-processing baseline
- [x] Statistical fault detection
- [ ] Quantitative benchmark across fault classes
- [ ] ML anomaly detection comparison
- [ ] Real sensor acquisition
- [ ] Microcontroller implementation
- [ ] Real-time dashboard

## Evaluation Metrics

The final system will report:

- Detection accuracy
- False alarm rate
- Detection delay
- Noise reduction / residual error
- Drift sensitivity
- Computational cost

## Engineering Philosophy

The project deliberately starts with interpretable signal-processing and statistical diagnostics. Machine learning will be introduced only as a benchmark against a transparent engineering baseline.

## Status

**Phase 1 — Software prototype**

## Author

Instrumentation Engineering project by **Dibyo**.
