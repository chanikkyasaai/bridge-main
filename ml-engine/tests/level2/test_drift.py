import numpy as np
import pytest
from mlengine.core.drift_detection import BehavioralDriftMonitor, DriftDetectionResult
from datetime import datetime
from collections import deque

@pytest.mark.parametrize("drift", [True, False])
def test_behavioral_drift_monitor(drift):
    monitor = BehavioralDriftMonitor(window_size=50)
    user_id = "test_user"
    # Create baseline and recent vectors
    baseline = np.random.normal(0, 1, (50, 16))
    if drift:
        recent = np.random.normal(3, 1, (50, 16))  # Shift mean for drift
    else:
        recent = np.random.normal(0, 1, (50, 16))  # No drift
    # Manually set up user profile
    monitor.user_profiles[user_id] = type('UserProfile', (), {
        'user_id': user_id,
        'base_vectors': baseline,
        'recent_vectors': recent,
        'cluster_centers': np.zeros((1, 16)),
        'cluster_labels': ['normal'],
        'feature_distributions': {},
        'drift_history': [],
        'last_updated': datetime.utcnow(),
        'creation_date': datetime.utcnow(),
        'total_samples': 100,
        'stability_score': 1.0
    })()
    # Patch the monitor's recent_vectors and baseline_vectors directly
    monitor.recent_vectors[user_id] = deque(recent, maxlen=50)
    monitor.baseline_vectors[user_id] = baseline
    result = monitor.detect_drift(user_id)
    assert isinstance(result, DriftDetectionResult)
    assert result.user_id == user_id
    assert isinstance(result.drift_detected, (bool, np.bool_))
    assert result.drift_type in ["gradual", "sudden", "concept", "none", "insufficient_data", "error"]
    assert 0.0 <= result.drift_magnitude <= 1.0
    assert isinstance(result.confidence, float)
    assert isinstance(result.affected_features, list)
    assert isinstance(result.recommendation, str)
    assert isinstance(result.timestamp, datetime)
    assert isinstance(result.metadata, dict)
    if drift:
        if not result.drift_detected:
            print(f"DEBUG: drift=True but result.drift_detected={result.drift_detected} (type={type(result.drift_detected)})")
        assert result.drift_detected
    else:
        if result.drift_detected:
            print(f"DEBUG: drift=False but result.drift_detected={result.drift_detected} (type={type(result.drift_detected)})")
        assert not result.drift_detected 