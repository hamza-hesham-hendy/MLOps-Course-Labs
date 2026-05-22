"""
Tests for the Churn Prediction API.

Run with:
    pytest tests/ -v
    pytest tests/ -v --cov=app --cov=main --cov-report=term-missing
"""

import importlib

import pytest
from litestar.testing import TestClient

from app import model_utils as model_utils_module
from app.model_utils import predict_churn
from main import app


def test_predict_churn_direct() -> None:
    sample_features = [0.0] * 11
    prediction = predict_churn(sample_features)
    assert prediction in (0, 1)


def test_predict_churn_edge_case() -> None:
    sample_features = [2.0, 2.0, 2.0, 2.0, 3.0, 1.0, 1.0, 3.0, 0.0, 1.0, 1.0]
    prediction = predict_churn(sample_features)
    assert isinstance(prediction, int)
    assert prediction in (0, 1)


def test_get_root() -> None:
    with TestClient(app=app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the churn prediction API"}


def test_get_health() -> None:
    with TestClient(app=app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_post_predict() -> None:
    payload = {
        "credit_score": 0.0,
        "age": 0.0,
        "tenure": 0.0,
        "balance": 0.0,
        "num_of_products": 0.0,
        "has_credit_card": 0.0,
        "is_active_member": 0.0,
        "estimated_salary": 0.0,
        "geography_germany": 0.0,
        "geography_spain": 0.0,
        "gender_male": 1.0,
    }
    with TestClient(app=app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 201
    assert response.json()["prediction"] in (0, 1)


def test_post_predict_invalid_input_returns_400() -> None:
    payload = {"credit_score": "invalid"}
    with TestClient(app=app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 400


def test_model_load_missing_file_raises(monkeypatch) -> None:
    def raise_missing_model(path):
        raise FileNotFoundError("missing model")

    monkeypatch.setattr(model_utils_module.joblib, "load", raise_missing_model)

    with pytest.raises(FileNotFoundError, match="Model file not found"):
        importlib.reload(model_utils_module)
