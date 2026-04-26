import pytest

from src.models.lstm_rul import build_lstm_rul_model


def test_build_lstm_rul_model_returns_dict():
    cfg = {"layers": 2}
    model = build_lstm_rul_model(config=cfg)
    assert isinstance(model, dict)
    assert model["model"] == "lstm_rul"
    assert model["config"] == cfg
