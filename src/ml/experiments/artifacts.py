from pathlib import Path

import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional plotting dependency
    plt = None


def save_split_artifacts(split_details: list[dict], output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths: list[str] = []

    for detail in split_details:
        split = detail["split"]
        market_label = detail.get("market_return_label", "single_symbol_close")
        split_dir = output_dir / f"split_{split:03d}"
        split_dir.mkdir(parents=True, exist_ok=True)

        x_train_path = split_dir / "X_train_head.csv"
        y_train_path = split_dir / "y_train_head.csv"
        x_test_path = split_dir / "X_test_head.csv"
        y_test_path = split_dir / "y_test_head.csv"
        probs_path = split_dir / "predicted_probabilities.csv"
        backtest_path = split_dir / "backtest.csv"
        probs_hist_path = split_dir / "probs_hist.png"
        cumret_path = split_dir / "cum_returns.png"
        cumret_by_symbol_path = split_dir / "cum_returns_by_symbol.png"
        probs_hist_by_symbol_path = split_dir / "probs_hist_by_symbol.png"

        detail["X_train_head"].to_csv(x_train_path, index=False)
        detail["y_train_head"].to_csv(y_train_path, index=False)
        detail["X_test_head"].to_csv(x_test_path, index=False)
        detail["y_test_head"].to_csv(y_test_path, index=False)
        pd.DataFrame({"prob_trade_success": detail["probs"]}).to_csv(
            probs_path, index=False
        )
        detail["df_backtest"].to_csv(backtest_path, index=False)

        if plt is not None:
            plt.figure(figsize=(8, 4))
            plt.hist(detail["probs"], bins=50)
            plt.title(f"Split {split}: Predicted Probability Distribution")
            plt.xlabel("prob_trade_success")
            plt.ylabel("count")
            plt.tight_layout()
            plt.savefig(probs_hist_path)
            plt.close()

            plt.figure(figsize=(8, 4))
            plt.plot(detail["df_backtest"]["cum_strategy_return"], label="Strategy")
            market_series_col = (
                "cum_market_return_pooled_eqw"
                if (
                    market_label == "pooled_equal_weight"
                    and "cum_market_return_pooled_eqw" in detail["df_backtest"].columns
                )
                else "cum_market_return"
            )
            market_plot_label = (
                "Market (pooled equal-weight)"
                if market_label == "pooled_equal_weight"
                else "Market"
            )
            plt.plot(detail["df_backtest"][market_series_col], label=market_plot_label)
            plt.title(f"Split {split}: Cumulative Returns ({market_label})")
            plt.xlabel("row")
            plt.ylabel("cumulative return")
            plt.legend()
            plt.tight_layout()
            plt.savefig(cumret_path)
            plt.close()

            df_backtest = detail["df_backtest"]
            pooled_ready = (
                market_label == "pooled_equal_weight"
                and "symbol" in df_backtest.columns
                and "close" in df_backtest.columns
                and "strategy_return" in df_backtest.columns
                and "prob_trade_success" in df_backtest.columns
            )
            if pooled_ready:
                by_symbol = df_backtest.copy()
                sort_cols = (
                    ["symbol", "timestamp"]
                    if "timestamp" in by_symbol.columns
                    else ["symbol"]
                )
                by_symbol = by_symbol.sort_values(sort_cols).reset_index(drop=True)
                symbols = sorted(by_symbol["symbol"].dropna().astype(str).unique())

                plt.figure(figsize=(11, 6))
                for sym in symbols:
                    grp = by_symbol[by_symbol["symbol"].astype(str) == sym].copy()
                    grp["cum_strategy_return_symbol"] = (
                        1.0 + grp["strategy_return"]
                    ).cumprod()
                    grp["market_return_symbol"] = grp["close"].pct_change().fillna(0.0)
                    grp["cum_market_return_symbol"] = (
                        1.0 + grp["market_return_symbol"]
                    ).cumprod()
                    x = grp["timestamp"] if "timestamp" in grp.columns else grp.index
                    plt.plot(
                        x, grp["cum_strategy_return_symbol"], label=f"{sym} strategy"
                    )
                    plt.plot(
                        x,
                        grp["cum_market_return_symbol"],
                        linestyle="--",
                        alpha=0.75,
                        label=f"{sym} market",
                    )

                plt.title(f"Split {split}: Cumulative Returns by Symbol")
                plt.xlabel("timestamp" if "timestamp" in by_symbol.columns else "row")
                plt.ylabel("cumulative return")
                plt.legend(ncol=2, fontsize=8)
                plt.tight_layout()
                plt.savefig(cumret_by_symbol_path)
                plt.close()

                plt.figure(figsize=(11, 6))
                for sym in symbols:
                    grp = by_symbol[by_symbol["symbol"].astype(str) == sym]
                    plt.hist(
                        grp["prob_trade_success"],
                        bins=40,
                        alpha=0.35,
                        label=str(sym),
                    )
                plt.title(f"Split {split}: Predicted Probabilities by Symbol")
                plt.xlabel("prob_trade_success")
                plt.ylabel("count")
                plt.legend()
                plt.tight_layout()
                plt.savefig(probs_hist_by_symbol_path)
                plt.close()

        artifact_paths.extend(
            [
                str(x_train_path),
                str(y_train_path),
                str(x_test_path),
                str(y_test_path),
                str(probs_path),
                str(backtest_path),
            ]
        )
        if plt is not None:
            artifact_paths.extend([str(probs_hist_path), str(cumret_path)])
            if cumret_by_symbol_path.exists():
                artifact_paths.append(str(cumret_by_symbol_path))
            if probs_hist_by_symbol_path.exists():
                artifact_paths.append(str(probs_hist_by_symbol_path))

    return artifact_paths
