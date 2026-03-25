from ml.backtest.walk_forward import walk_forward_split
from ml.evaluate import evaluate_model
from ml.models.save_loads import save_model


def run_backtest(
    symbol: str,
    X,
    y,
    model_fn,
    train_size=2000,
    test_size=250,
    step_size=250,
):

    results = []

    splits = walk_forward_split(
        X,
        y,
        train_size=train_size,
        test_size=test_size,
        step_size=step_size,
    )

    final_model = None

    for i, (X_train, X_test, y_train, y_test) in enumerate(splits):
        # train model
        model = model_fn(symbol, X_train, y_train)
        # predict
        preds = model.predict(X_test)

        # evaluate
        metrics = evaluate_model(preds, y_test, verbose=False)
        metrics["split"] = i

        results.append(metrics)
        final_model = model

        print(f"split {i} | {metrics}")

    print(f"Model trained for {symbol}")
    save_model(final_model, f"models/random_forest_{symbol}.pkl")
    print(f"Model saved to models/random_forest_{symbol}.pkl")

    return results
