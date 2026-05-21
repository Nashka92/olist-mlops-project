import numpy as np
import pytest

from src.ml_olist.common.features import add_olist_features
from src.ml_olist.training.data import load_olist_data, split_data
from src.ml_olist.training.features import build_preprocessing_pipeline


@pytest.fixture(scope="module")
def splits():
    df = load_olist_data()
    return split_data(df)


def test_derived_features_present(splits):
    X_train, _, _, _ = splits

    output = add_olist_features(X_train)

    expected_columns = [
        "freight_ratio",
        "log_price",
        "is_multi_item",
        "purchase_dow",
        "purchase_month",
    ]

    for column in expected_columns:
        assert column in output.columns


def test_no_nan_after_preprocessing(splits):
    X_train, X_test, _, _ = splits

    pipe = build_preprocessing_pipeline()

    assert not np.isnan(pipe.fit_transform(X_train)).any()
    assert not np.isnan(pipe.transform(X_test)).any()


def test_shape_preserved(splits):
    X_train, _, _, _ = splits

    pipe = build_preprocessing_pipeline()
    output = pipe.fit_transform(X_train)

    assert output.shape[0] == X_train.shape[0]


def test_log_price_positive(splits):
    X_train, _, _, _ = splits

    output = add_olist_features(X_train)

    assert (output["log_price"] >= 0).all()
