import os

# Default pipeline universe when INGESTION_UNIVERSE is unset or ``subscriptions``
# (see ``universe.resolve``). For S&P 500 + context symbols, fetch tickers then set
# ``INGESTION_UNIVERSE=sp500`` (Makefile: ``universe-fetch-sp500``, ``ingestion-sp500``).
SUBSCRIPTIONS = [
    "AAPL",
    "MSFT",
    "GOOG",
    "AMZN",
    "TSLA",
    "NVDA",
    "META",
    "QQQ",
    "SPY",
    "^VIX",
]
MARKET_CONTEXT_SYMBOLS = {"QQQ", "SPY", "^VIX"}
TRAIN_SYMBOLS = [s for s in SUBSCRIPTIONS if s not in MARKET_CONTEXT_SYMBOLS]

TP_PCT = 0.08
SL_PCT = 0.04
MAX_HOLD_DAYS = 15
# signal threshold
# EV = P(win) * TP - (1 - P(win)) * SL
# trade only if P(win)*0.08 - (1-P(win))*0.04 > 0 => P(win) > 0.33
THRESHOLD = 0.33

# Threshold grid selection in walk-forward (see ml.backtest.threshold_optimization)
THRESHOLD_OBJECTIVE = "calmar_proxy"
THRESHOLD_SELECTION_LAMBDA_DD = 1.0
THRESHOLD_CALMAR_EPS = 1e-6
THRESHOLD_MIN_AVG_PROFIT_FACTOR = None  # e.g. 1.0 to require avg PF >= 1
THRESHOLD_MAX_MEAN_ABS_DRAWDOWN = None  # e.g. 0.35 caps mean |max_drawdown|
# ~21 trading days/month * 2; scales min trade count for threshold grid eligibility
THRESHOLD_TRADING_DAYS_PER_TWO_MONTHS = 42

# ``single_objective``: maximize selection_score (legacy).
# ``multi_top_k``: each metric rank (within pool) must be <= K; relax K until non-empty
# or fall back to best mean rank (see ml.backtest.threshold_optimization).
THRESHOLD_SELECTION_MODE = (
    os.environ.get("THRESHOLD_SELECTION_MODE", "single_objective").strip().lower()
)
THRESHOLD_MULTI_TOP_K_START = int(os.environ.get("THRESHOLD_MULTI_TOP_K_START", "3"))
THRESHOLD_MULTI_TOP_K_MAX = int(os.environ.get("THRESHOLD_MULTI_TOP_K_MAX", "16"))
_DEFAULT_THRESHOLD_MULTI_METRICS = (
    "avg_cum_return,avg_profit_factor,avg_win_rate,"
    "avg_directional_accuracy_strategy,avg_max_drawdown,avg_mae_at_threshold"
)
THRESHOLD_MULTI_METRICS = os.environ.get(
    "THRESHOLD_MULTI_METRICS", _DEFAULT_THRESHOLD_MULTI_METRICS
).strip()

EXPERIMENT_STRATEGY_SLUG = "swing-trade"
