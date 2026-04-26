import pytest

from src.preprocessing.features import extract_features


def test_extract_features_basic():
    data = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ]
    feats = extract_features(data)
    assert isinstance(feats, list)
    assert len(feats) == 3
    assert feats[0] == pytest.approx((1.0 + 4.0 + 7.0) / 3)
