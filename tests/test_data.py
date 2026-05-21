import pandas as pd
import pytest

from src.ml_olist.training.data import load_olist_data, split_data


@pytest.fixture(scope="module")
def df():
    return load_olist_data()


def test_load_returns_non_empty(df):
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 1000


def test_target_positive(df):
    assert (df["delivery_time_days"] > 0).all()


def test_target_under_60_days(df):
    assert (df["delivery_time_days"] < 60).all()


def test_split_sizes(df):
    _, X_test, _, _ = split_data(df)

    ratio_test = len(X_test) / len(df)

    assert abs(ratio_test - 0.2) < 0.02
