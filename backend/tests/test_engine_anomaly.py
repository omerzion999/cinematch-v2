import numpy as np
import pytest

from app.engine import anomaly


@pytest.fixture(autouse=True)
def reset_anomaly_state():
    """Reset anomaly module global state before and after each test."""
    anomaly._threshold = None
    yield
    anomaly._threshold = None


def test_calibrate_sets_threshold_to_percentile():
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    threshold = anomaly.calibrate(scores, percentile=10.0)
    assert threshold == np.percentile(scores, 10.0)
    assert anomaly.get_threshold() == threshold


def test_is_anomalous_below_threshold():
    anomaly.calibrate(np.array([0.5, 0.6, 0.7, 0.8, 0.9]), percentile=20.0)
    threshold = anomaly.get_threshold()
    assert anomaly.is_anomalous(threshold - 0.01) is True
    assert anomaly.is_anomalous(threshold + 0.01) is False


def test_is_anomalous_uses_explicit_threshold_override():
    assert anomaly.is_anomalous(0.1, threshold=0.5) is True
    assert anomaly.is_anomalous(0.6, threshold=0.5) is False
