# Cross-Process Generalization Benchmark

This experiment evaluates whether the sensor-fault classifier learns fault signatures rather than memorizing the nominal process waveform.

## Protocol

The Random Forest is trained using the existing feature dataset generated from the nominal process. It is then frozen and tested on a separate 500-trial dataset generated from an unseen process waveform:

`30 + 3.8 sin(2π·0.035t + 0.4) + 1.2 sin(2π·0.11t)`

The fault mechanisms remain the same: healthy, noisy, bias, drift, and stuck. Fault onset remains fixed at 60 s, and classification uses the post-onset feature window.

The benchmark compares Random Forest and the interpretable rule-based classifier using accuracy, macro F1, and confusion matrices.

## Run

```powershell
python -m experiments.cross_process_generalization
```

## Interpretation

This is still synthetic validation. Strong performance indicates robustness to a change in the underlying process dynamics, but it does not replace evaluation on real industrial sensor data or hardware-in-the-loop experiments.
