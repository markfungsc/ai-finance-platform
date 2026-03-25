import numpy as np

from ml.backtest.runner import run_backtest
from ml.dataset import load_dataset
from ml.models.random_forest import train_random_forest


def run(symbol: str, model_fn):
    X, y, df = load_dataset(symbol)
    print("dataset size:", len(X))

    results = run_backtest(
        symbol,
        X,
        y,
        model_fn=model_fn,
        train_size=2000,
        test_size=250,
        step_size=250,
    )
    return results


if __name__ == "__main__":
    # symbols = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]
    symbols = ["AAPL"]
    for symbol in symbols:
        results = run(symbol, train_random_forest)
        print(f"symbol: {symbol} | {results}")
        mae_scores = [r["mae"] for r in results]
        dir_scores = [r["directional_accuracy"] for r in results]

        print("\nAverage MAE:", np.mean(mae_scores))
        print("Average Directional Accuracy:", np.mean(dir_scores))
