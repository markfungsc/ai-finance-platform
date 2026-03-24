from database.queries import fetch_features
from ml.features import FEATURE_COLUMNS, TARGET_COLUMN


def load_dataset(symbol: str):
    df = fetch_features(symbol)
    df[TARGET_COLUMN] = df["return_1d"].shift(-1)
    df = df.dropna()
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    return X, y, df
