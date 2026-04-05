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

EXPERIMENT_STRATEGY_SLUG = "swing-trade"
