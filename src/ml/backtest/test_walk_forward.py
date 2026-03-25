from ml.backtest.walk_forward import walk_forward_split
from ml.dataset import load_dataset


def test_walk_forward(symbol: str):
    # Build dataset
    X, y, _df = load_dataset(symbol)

    print("dataset size:", len(X))

    splits = walk_forward_split(
        X,
        y,
        train_size=2000,
        test_size=250,
        step_size=250,
    )

    for i, (X_train, X_test, _y_train, _y_test) in enumerate(splits):
        print(f"Split {i} | train={len(X_train)} test={len(X_test)}")


if __name__ == "__main__":
    # symbols = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]
    symbols = ["AAPL"]
    for symbol in symbols:
        test_walk_forward(symbol)
