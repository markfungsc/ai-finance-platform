from pathlib import Path

import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional plotting dependency
    plt = None


POOLED_PORTFOLIO_MARKET_LABELS = frozenset(
    {"pooled_equal_weight", "pooled_avg_buyhold"}
)


def _dedupe_pooled_timestamp_for_plot(df: pd.DataFrame) -> pd.DataFrame:
    """
    For pooled runs, strategy/market equity values repeat for each symbol row at the
    same timestamp. When plotting the pooled portfolio curve, use one point per
    timestamp.
    """
    if "timestamp" not in df.columns:
        return df
    return (
        df.sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .copy()
    )


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
        dfb = detail["df_backtest"]
        if "symbol" in dfb.columns and "timestamp" in dfb.columns:
            dfb = dfb.sort_values(["symbol", "timestamp"], kind="mergesort")
        elif "timestamp" in dfb.columns:
            dfb = dfb.sort_values(["timestamp"], kind="mergesort")
        dfb.to_csv(backtest_path, index=False)

        thr_path = split_dir / "backtest_threshold.txt"
        bt = detail.get("backtest_threshold")
        if bt is not None:
            thr_path.write_text(f"{float(bt)}\n", encoding="utf-8")

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

            df_backtest = detail["df_backtest"]
            pooled_ready = (
                market_label in POOLED_PORTFOLIO_MARKET_LABELS
                and "timestamp" in df_backtest.columns
                and "cum_strategy_return" in df_backtest.columns
            )
            if pooled_ready:
                plot_df = _dedupe_pooled_timestamp_for_plot(df_backtest)
                x = plot_df["timestamp"]
            else:
                plot_df = df_backtest
                x = plot_df.index

            plt.plot(x, plot_df["cum_strategy_return"], label="Strategy")
            market_series_col = (
                "cum_market_return_pooled_eqw"
                if (
                    market_label in POOLED_PORTFOLIO_MARKET_LABELS
                    and "cum_market_return_pooled_eqw" in df_backtest.columns
                )
                else "cum_market_return"
            )
            market_plot_label = (
                "Market (pooled avg buy-hold)"
                if market_label == "pooled_avg_buyhold"
                else (
                    "Market (pooled equal-weight)"
                    if market_label == "pooled_equal_weight"
                    else "Market"
                )
            )
            if pooled_ready:
                plt.plot(x, plot_df[market_series_col], label=market_plot_label)
            else:
                plt.plot(plot_df[market_series_col], label=market_plot_label)
            plt.title(f"Split {split}: Cumulative Returns ({market_label})")
            plt.xlabel("timestamp" if pooled_ready else "row")
            plt.ylabel("cumulative return")
            plt.legend()
            plt.tight_layout()
            plt.savefig(cumret_path)
            plt.close()

            pooled_ready = (
                market_label in POOLED_PORTFOLIO_MARKET_LABELS
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
                    m = by_symbol["symbol"].astype(str) == sym
                    sub = by_symbol.loc[m].copy()
                    c0 = float(sub["close"].iloc[0])
                    if c0 > 1e-12:
                        cum_mkt_series = sub["close"].astype(float) / c0
                    else:
                        cum_mkt_series = pd.Series(1.0, index=sub.index)
                    market_ret = cum_mkt_series.pct_change().fillna(0.0)
                    extra = pd.DataFrame(
                        {
                            "cum_strategy_return_symbol": (
                                1.0 + sub["strategy_return"]
                            ).cumprod(),
                            "market_return_symbol": market_ret,
                            "cum_market_return_symbol": cum_mkt_series,
                        },
                        index=sub.index,
                    )
                    grp = pd.concat([sub, extra], axis=1)
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
        if bt is not None:
            artifact_paths.append(str(thr_path))
        if plt is not None:
            artifact_paths.extend([str(probs_hist_path), str(cumret_path)])
            if cumret_by_symbol_path.exists():
                artifact_paths.append(str(cumret_by_symbol_path))
            if probs_hist_by_symbol_path.exists():
                artifact_paths.append(str(probs_hist_by_symbol_path))

    return artifact_paths
