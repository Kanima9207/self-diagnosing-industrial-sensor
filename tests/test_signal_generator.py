import numpy as np

from src.signal_generator import SCENARIOS, simulate_sensor, true_process


def test_true_process_shape():
    t = np.arange(0, 10, 0.1)
    assert true_process(t).shape == t.shape


def test_stuck_sensor_is_constant():
    t = np.arange(0, 10, 0.1)
    _, measured = simulate_sensor(t, SCENARIOS["stuck"])
    assert np.allclose(measured, measured[0])


def test_bias_increases_mean_residual():
    t = np.arange(0, 120, 0.1)
    truth, measured = simulate_sensor(t, SCENARIOS["bias"])
    assert np.mean(measured - truth) > 1.5
