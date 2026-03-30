import warnings

import mlflow
import mlflow.sklearn


def log_experiment_mlflow(
    experiment: dict,
    *,
    model=None,
    artifact_paths: list[str] | None = None,
):
    """Log params, metrics, model, and optional artifacts to MLflow."""
    with mlflow.start_run(run_name=f"{experiment['symbol']}_{experiment['model']}"):
        mlflow.log_param("model", experiment["model"])
        mlflow.log_param("symbol", experiment["symbol"])
        mlflow.log_param("features", experiment["features"])

        mlflow.log_metric("mae", float(experiment["mae"]))
        mlflow.log_metric(
            "directional_accuracy", float(experiment["directional_accuracy"])
        )
        mlflow.log_metric(
            "cum_strategy_return", float(experiment["cum_strategy_return"])
        )
        mlflow.log_metric("cum_market_return", float(experiment["cum_market_return"]))
        mlflow.log_metric(
            "directional_accuracy_strategy",
            float(experiment["directional_accuracy_strategy"]),
        )
        mlflow.log_metric(
            "strategy_total_trades", float(experiment["strategy_total_trades"])
        )
        mlflow.log_metric(
            "strategy_total_hits", float(experiment["strategy_total_hits"])
        )

        _opt_metrics = [
            ("avg_win_rate", "avg_win_rate"),
            ("avg_profit_factor", "avg_profit_factor"),
            ("avg_max_drawdown", "avg_max_drawdown"),
            ("best_threshold", "best_threshold"),
            ("opt_avg_cum_return", "opt_avg_cum_return"),
            ("opt_avg_cum_market_return", "opt_avg_cum_market_return"),
            ("opt_avg_profit_factor", "opt_avg_profit_factor"),
            ("opt_avg_win_rate", "opt_avg_win_rate"),
            ("opt_total_trades", "opt_total_trades"),
        ]
        for key, ml_name in _opt_metrics:
            val = experiment.get(key)
            if val is not None:
                mlflow.log_metric(ml_name, float(val))

        if model is not None:
            # MLflow warns about pickle deserialization; skops is optional extra.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Saving scikit-learn models",
                    category=UserWarning,
                )
                mlflow.sklearn.log_model(model, name="model")

        for artifact in artifact_paths or []:
            mlflow.log_artifact(artifact)
