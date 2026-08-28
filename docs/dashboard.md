# Interactive Dashboard

The project includes a Streamlit dashboard for demonstrating the synthetic sensor-diagnosis pipeline interactively.

## Run locally

Pull the latest repository changes, then install the dashboard dependencies:

```powershell
pip install -r requirements-dashboard.txt
```

Start the dashboard:

```powershell
streamlit run app.py
```

The dashboard provides controls for fault type, fault severity, noise, and random seed. It displays the true process, measured sensor signal, residual, extracted diagnostic features, and the rule-based diagnosis with a heuristic confidence score.

## Scope

The dashboard uses the same synthetic signal model as the experiments. It is a demonstration interface and should not be treated as a safety-critical industrial diagnostic system without validation on real sensor data and appropriate hardware/software assurance.
