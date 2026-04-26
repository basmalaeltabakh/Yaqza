from src.mlops.drift import detect_drift


def test_detect_drift_defaults_to_false():
    assert detect_drift([]) is False
