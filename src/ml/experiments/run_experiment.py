import json
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional plotting dependency
    plt = None

from constants import EXPERIMENT_STRATEGY_SLUG, THRESHOLD
from log_config import get_logger
from ml.analysis.feature_importance import get_feature_importance
from ml.backtest.runner import run_backtest
from ml.dataset import get_pooled_dataset_symbols, load_train_dataset
from ml.experiments.artifacts import save_split_artifacts
from ml.experiments.logger import log_experiment
from ml.experiments.mlflow_logger import log_experiment_mlflow
from ml.models.registry import MODEL_REGISTRY

logger = get_logger(__name__)


def run_experiment(model_name: str = "logistic_regression"):
    pooled_symbols = get_pooled_dataset_symbols()
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

    th_opt = summary.get("threshold_optimization") or {}
    best_agg = th_opt.get("best_aggregate") or {}
    best_threshold = best_agg.get("threshold")
    threshold_grid = th_opt.get("grid") or []

    experiment = {
        "model": model_name,
        "symbol": run_symbol,
        "n_symbols": len(pooled_symbols),
        "symbols": ",".join(pooled_symbols),
        "mae": summary["avg_mae"].round(4),
        "directional_accuracy": summary["avg_directional_accuracy"].round(4),
        "cum_strategy_return": summary["avg_strategy_cum_return"],
        "cum_market_return": summary["avg_market_return"],
        "cum_market_return_pooled_eqw": summary.get("avg_market_return_pooled_eqw"),
        "directional_accuracy_strategy": summary["avg_strategy_directional_accuracy"],
        "strategy_total_trades": summary["strategy_total_trades"],
        "strategy_total_hits": summary["strategy_total_hits"],
        "avg_win_rate": summary["avg_win_rate"],
        "avg_profit_factor": summary["avg_profit_factor"],
        "avg_max_drawdown": summary["avg_max_drawdown"],
        "best_threshold": best_agg.get("threshold"),
        "opt_avg_cum_return": best_agg.get("avg_cum_return"),
        "opt_avg_cum_market_return": best_agg.get("avg_cum_market_return"),
        "opt_avg_profit_factor": best_agg.get("avg_profit_factor"),
        "opt_avg_win_rate": best_agg.get("avg_win_rate"),
        "opt_total_trades": best_agg.get("total_trades"),
        "features": X.shape[1],
    }

    # Persist the optimized threshold so the API/scanner can use it for
    # consistent trade/no-trade decisions without re-running backtests.
    if best_threshold is not None:
        best_thr_path = (
            Path("models")
            / EXPERIMENT_STRATEGY_SLUG
            / f"{model_name}_{run_symbol}_best_threshold.json"
        )
        best_thr_path.parent.mkdir(parents=True, exist_ok=True)
        best_thr_path.write_text(
            json.dumps({"best_threshold": float(best_threshold)}, indent=2),
            encoding="utf-8",
        )
        logger.info("Saved best_threshold=%s to %s", best_threshold, best_thr_path)
    # Persist the full threshold grid so UI / API clients can visualize it
    # and avoid re-running expensive backtests.
    grid_payload = {
        "best_threshold": float(best_threshold) if best_threshold is not None else None,
        "grid": threshold_grid,
    }
    grid_path = (
        Path("models")
        / EXPERIMENT_STRATEGY_SLUG
        / f"{model_name}_{run_symbol}_threshold_grid.json"
    )
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    grid_path.write_text(json.dumps(grid_payload, indent=2), encoding="utf-8")
    logger.info("Saved threshold_grid to %s", grid_path)

    log_experiment(experiment)

    model_slug = model_name.replace(" ", "_")
    artifacts_root = (
        Path(f"experiments/artifacts/{EXPERIMENT_STRATEGY_SLUG}")
        / f"{run_symbol}_{model_slug}"
    )
    artifact_paths = save_split_artifacts(summary["split_details"], artifacts_root)

    grid_copy = artifacts_root / "threshold_grid.json"
    grid_copy.write_text(json.dumps(grid_payload, indent=2), encoding="utf-8")
    artifact_paths.append(str(grid_copy))

    # Optional visualization: threshold objective landscape across the grid.
    grid = summary.get("threshold_optimization", {}).get("grid", []) or []
    if plt is not None and grid:
        try:
            grid_sorted = sorted(grid, key=lambda r: float(r.get("threshold", 0.0)))
            thresholds = [float(r["threshold"]) for r in grid_sorted]

            avg_cum_return = [float(r.get("avg_cum_return", 1.0)) for r in grid_sorted]
            avg_profit_factor = [
                float(r.get("avg_profit_factor", 0.0)) for r in grid_sorted
            ]
            avg_win_rate = [float(r.get("avg_win_rate", 0.0)) for r in grid_sorted]
            avg_max_drawdown = [
                float(r.get("avg_max_drawdown", 0.0)) for r in grid_sorted
            ]

            best_threshold = summary.get("threshold_optimization", {}).get(
                "best_threshold"
            )
            best_threshold = (
                float(best_threshold) if best_threshold is not None else None
            )

            # Rescale metrics to make them visible on the same plot.
            win_rate_pct = [w * 100.0 for w in avg_win_rate]
            max_drawdown_pct = [d * 100.0 for d in avg_max_drawdown]

            fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
            ax1, ax2, ax3 = axes

            ax1.plot(thresholds, avg_cum_return, marker="o")
            ax1.set_ylabel("avg_cum_return")
            ax1.grid(True, alpha=0.25)

            ax2.plot(thresholds, avg_profit_factor, marker="s", linestyle="--")
            ax2.set_ylabel("avg_profit_factor")
            ax2.grid(True, alpha=0.25)

            ax3.plot(thresholds, win_rate_pct, marker="^", linestyle=":")
            ax3.plot(thresholds, max_drawdown_pct, marker="x", linestyle="-.")
            ax3.set_ylabel("win_rate% and max_drawdown%")
            ax3.grid(True, alpha=0.25)

            if best_threshold is not None:
                for ax in axes:
                    ax.axvline(
                        best_threshold,
                        color="red",
                        linestyle="--",
                        alpha=0.7,
                    )

            ax3.set_xlabel("threshold")
            fig.suptitle("Threshold optimization metrics", y=0.98)
            fig.tight_layout()

            plot_path = artifacts_root / "threshold_grid_metrics.png"
            fig.savefig(plot_path)
            plt.close(fig)
            artifact_paths.append(str(plot_path))
            logger.info("Saved threshold grid plot to %s", plot_path)
        except Exception:
            # Plotting should never break experiment runs.
            logger.exception("Failed to generate threshold grid plot")

    # mlflow logging
    log_experiment_mlflow(experiment, model=model, artifact_paths=artifact_paths)

    logger.info("Experiment Summary %s", experiment)

    return experiment


if __name__ == "__main__":
    model_names = ["random_forest"]
    for model_name in model_names:
        run_experiment(model_name)
