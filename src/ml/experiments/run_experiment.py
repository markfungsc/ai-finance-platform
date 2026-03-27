from pathlib import Path

from constants import THRESHOLD, TRAIN_SYMBOLS
from ml.analysis.feature_importance import get_feature_importance
from ml.backtest.runner import run_backtest
from ml.dataset import load_train_dataset
from ml.experiments.artifacts import save_split_artifacts
from ml.experiments.logger import log_experiment
from ml.experiments.mlflow_logger import log_experiment_mlflow
from ml.models.registry import MODEL_REGISTRY


def run_experiment(model_name: str = "logistic_regression"):
    X, y, df_merged = load_train_dataset()
    run_symbol = "pooled"

    _results, summary, model = run_backtest(
        symbol=run_symbol,
        X=X,
        y=y,
        model_fn=MODEL_REGISTRY[model_name],
        model_name=model_name,
        df_merged=df_merged,
        threshold=THRESHOLD,
    )

    if hasattr(model, "feature_importances_"):
        get_feature_importance(model, X.columns)

    experiment = {
        "model": model_name,
        "symbol": run_symbol,
        "n_symbols": len(TRAIN_SYMBOLS),
        "symbols": ",".join(TRAIN_SYMBOLS),
        "mae": summary["avg_mae"].round(4),
        "directional_accuracy": summary["avg_directional_accuracy"].round(4),
        "cum_strategy_return": summary["avg_strategy_cum_return"],
        "cum_market_return": summary["avg_market_return"],
        "cum_market_return_pooled_eqw": summary.get("avg_market_return_pooled_eqw"),
        "directional_accuracy_strategy": summary["avg_strategy_directional_accuracy"],
        "strategy_total_trades": summary["strategy_total_trades"],
        "strategy_total_hits": summary["strategy_total_hits"],
        "features": X.shape[1],
    }

    log_experiment(experiment)

    artifacts_root = Path("experiments/artifacts") / f"{run_symbol}_{model_name}"
    artifact_paths = save_split_artifacts(summary["split_details"], artifacts_root)

    # mlflow logging
    log_experiment_mlflow(experiment, model=model, artifact_paths=artifact_paths)

    print("\nExperiment Summary")
    print(experiment)

    return experiment


if __name__ == "__main__":
    model_names = ["random_forest"]
    for model_name in model_names:
        run_experiment(model_name)
