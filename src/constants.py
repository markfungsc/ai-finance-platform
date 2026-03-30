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
MAX_HOLD_DAYS = 10
# signal threshold
# EV = P(win) * TP - (1 - P(win)) * SL
# trade only if P(win)*0.08 - (1-P(win))*0.04 > 0 => P(win) > 0.33
THRESHOLD = 0.33

EXPERIMENT_STRATEGY_SLUG = "swing-trade"
