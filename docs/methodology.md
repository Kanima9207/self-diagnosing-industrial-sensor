# Methodology

## 1. Process model

The baseline process is a slowly varying signal composed of low-frequency sinusoidal components. It represents a generic industrial process variable rather than a specific physical plant.

## 2. Sensor model

The measurement model is:

`y(t) = x(t) + n(t) + b(t) + d(t)`

Noise, bias, and drift are independently configurable. A stuck fault replaces the measurement with a constant value.

## 3. Signal processing

The first baseline uses a moving mean and moving standard deviation. These provide local estimates of signal level and variation without requiring a black-box model.

## 4. Fault logic

- **Stuck:** local standard deviation approaches zero.
- **Bias:** sustained residual magnitude exceeds a threshold.
- **Drift:** local residual slope exceeds a threshold.
- **Noise:** local variability exceeds a threshold.

Thresholds are intentionally explicit so they can be tuned and benchmarked during experiments.

## 5. Validation plan

Each fault will be evaluated using:

- detection rate
- false alarm rate
- detection delay
- sensitivity to fault magnitude
- robustness to random noise

The baseline will later be compared against CUSUM and machine-learning anomaly detectors.
