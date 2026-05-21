from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

MODEL_PATH = Path("artifacts/model_latest.joblib")

SAMPLE = pd.DataFrame(
    [
        {
            "price": 89.90,
            "freight_value": 12.50,
            "payment_installments": 1,
            "payment_value": 102.40,
            "order_item_id": 1,
            "order_purchase_timestamp": datetime(2018, 6, 10, 9, 0),
            "seller_state": "SP",
            "customer_state": "SP",
        }
    ]
)


@pytest.fixture(scope="module")
def model():
    if not MODEL_PATH.exists():
        pytest.skip("Modèle absent — lancer train.py d'abord")

    import joblib

    return joblib.load(MODEL_PATH)


def test_prediction_is_positive(model):
    prediction = model.predict(SAMPLE)[0]

    assert prediction > 0


def test_prediction_under_60_days(model):
    prediction = model.predict(SAMPLE)[0]

    assert prediction < 60
