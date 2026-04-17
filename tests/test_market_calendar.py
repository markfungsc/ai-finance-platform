import pandas as pd

from market.calendar import last_completed_xnys_close_utc


def test_last_completed_xnys_close_skips_us_market_holiday():
    # 2026-07-03 is observed market holiday for Independence Day.
    now = pd.Timestamp("2026-07-03T20:00:00Z")
    close = last_completed_xnys_close_utc(now)
    assert close is not None
    assert close.date().isoformat() == "2026-07-02"


def test_last_completed_xnys_close_handles_half_day_session():
    # Day after Thanksgiving is typically a half-day session.
    now = pd.Timestamp("2026-11-27T20:00:00Z")
    close = last_completed_xnys_close_utc(now)
    assert close is not None
    assert close.date().isoformat() == "2026-11-27"
