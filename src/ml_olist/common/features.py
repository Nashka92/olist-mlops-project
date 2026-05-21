import numpy as np
import pandas as pd


def add_olist_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les features métier utilisées par le modèle Olist.

    Ces features sont utilisées à l'entraînement et à la prédiction.
    """

    df = df.copy()

    total = df["price"] + df["freight_value"] + 1e-8

    df["freight_ratio"] = df["freight_value"] / total
    df["log_price"] = np.log1p(df["price"])
    df["is_multi_item"] = (df["order_item_id"] > 1).astype(int)
    df["purchase_dow"] = df["order_purchase_timestamp"].dt.dayofweek
    df["purchase_month"] = df["order_purchase_timestamp"].dt.month

    return df
