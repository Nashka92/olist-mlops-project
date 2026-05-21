from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.ml_olist.common.features import add_olist_features

NUM_FEATURES = [
    "price",
    "freight_value",
    "payment_installments",
    "payment_value",
    "order_item_id",
    "freight_ratio",
    "log_price",
    "is_multi_item",
    "purchase_dow",
    "purchase_month",
]

CAT_FEATURES = [
    "seller_state",
    "customer_state",
]


def build_preprocessing_pipeline() -> Pipeline:
    """
    Construit le pipeline de preprocessing Olist.

    Étapes :
    1. Ajout des features métier
    2. Traitement des colonnes numériques
    3. Traitement des colonnes catégorielles
    """

    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("num", numeric_pipe, NUM_FEATURES),
            ("cat", cat_pipe, CAT_FEATURES),
        ],
        remainder="drop",
    )

    return Pipeline(
        [
            ("features", FunctionTransformer(add_olist_features, validate=False)),
            ("preprocessing", preprocessor),
        ]
    )
