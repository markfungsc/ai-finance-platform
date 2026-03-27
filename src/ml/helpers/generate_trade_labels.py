import pandas as pd

from constants import MAX_HOLD_DAYS, SL_PCT, TP_PCT
from ml.features import TARGET_COLUMN


def generate_trade_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    labels = []

    for i in range(len(df)):
        entry_price = df.iloc[i]["close"]

        tp_price = entry_price * (1 + TP_PCT)
        sl_price = entry_price * (1 - SL_PCT)
        label = 0

        for j in range(1, MAX_HOLD_DAYS + 1):
            if i + j >= len(df):
                label = None
                break

            future_high = df.iloc[i + j]["high"]
            future_low = df.iloc[i + j]["low"]

            if future_high >= tp_price:
                label = 1
                break

            if future_low <= sl_price:
                label = 0
                break

        labels.append(label)

    df[TARGET_COLUMN] = labels

    # remove rows without future data
    df = df.dropna(subset=[TARGET_COLUMN])

    return df
