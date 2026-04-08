import numpy as np
import pandas as pd

from database.queries import fetch_features, fetch_features_z
from ml.helpers.attach_market_context import attach_market_context
from ml.helpers.merge_features import merge_features_with_target
from ml.models.save_loads import load_model
from ml.sentiment.attach import attach_sentiment_features


def generate_predictions(model, X: pd.DataFrame) -> np.ndarray:
    """
    Generate predicted returns using the trained model.

    Args:
        model: trained sklearn model
        X: feature matrix (normalized / z-score features)

    Returns:
        predictions: np.ndarray of predicted returns
    """
    return model.predict(X)


def generate_signals(predictions: np.ndarray) -> np.ndarray:
    """
    Convert predicted returns into trading signals.

    Args:
        predictions: np.ndarray of predicted returns

    Returns:
        signals: 1 → BUY, -1 → SELL, 0 → neutral
    """
    return np.where(predictions > 0.02, 1, np.where(predictions < -0.02, -1, 0))


def predict_for_symbol(
    model_path: str,
    symbol: str,
    n_display: int = 50,
    *,
    debug_merge: bool = False,
):
    """
    Load model and features for a symbol, generate predictions, signals, and display table.

    Args:
        model_path: path to saved model
        symbol: stock symbol to predict
        target_shift: shift for the target return (e.g., 1 for 1-day, 5 for 5-day)
        n_display: number of rows to print
        debug_merge: print last 10 rows (timestamp, symbol, return_1d/5d, return_1d_z/5d_z) per stage
    """
    # Load the trained model
    model = load_model(model_path)

    # Load normalized features from feature store
    df = fetch_features(symbol)
    df_z = fetch_features_z(symbol)

    # attach market context and optional sentiment z-scores
    df_z_context = attach_market_context(df_z)
    df_z_context = attach_sentiment_features(df_z_context)

    X, y_actual, df_merged = merge_features_with_target(
        df, df_z_context, debug=debug_merge
    )

    # Generate predictions
    predictions = generate_predictions(model, X)

    # Generate signals
    signals = generate_signals(predictions)

    # Build output table
    out_df = pd.DataFrame(
        {
            "timestamp": df_merged["timestamp"],
            "predicted_return": predictions,
            "signal": signals,
            "actual_return": y_actual,
        }
    )

    print(f"\n=== Predictions & Signals for {symbol} ===")
    print(out_df.tail(n_display))

    return out_df


# ----------------------------------------
# Run as script
# ----------------------------------------
if __name__ == "__main__":
    symbol = "AAPL"
    model_path = f"models/random_forest_{symbol}.pkl"

    predict_for_symbol(model_path, symbol, debug_merge=True)
