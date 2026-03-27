from ml.backtest.runner import run_backtest
from ml.dataset import load_dataset
from ml.models.registry import MODEL_REGISTRY


def run(symbol: str, model_fn, model_name):
    X, y, df_merged = load_dataset(symbol)
    print("dataset size:", len(X))
    print("y value counts:", y.value_counts())

    i = 1000
    row = df_merged.iloc[i]

    entry = row["close"]
    tp = entry * 1.08
    sl = entry * 0.96

    print("Entry:", entry)
    print("TP:", tp)
    print("SL:", sl)

    print(df_merged.iloc[i + 1 : i + 11][["high", "low"]])
    print("Label:", row["trade_success"])

    results = run_backtest(
        symbol,
        X,
        y,
        model_fn=model_fn,
        model_name=model_name,
        df_merged=df_merged,
        train_size=2000,
        test_size=250,
        step_size=250,
    )
    return results


if __name__ == "__main__":
    # symbols = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]
    symbols = ["AAPL"]
    model_names = ["random_forest", "logistic_regression"]
    for symbol in symbols:
        for model_name in model_names:
            results, summary, model = run(
                symbol, MODEL_REGISTRY[model_name], model_name
            )
            print(f"symbol: {symbol} | {model_name} | {summary}")
