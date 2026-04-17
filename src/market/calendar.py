"""Market session calendar utilities."""

from __future__ import annotations

import pandas as pd


def last_completed_xnys_close_utc(
    now_utc: pd.Timestamp | None = None,
) -> pd.Timestamp | None:
    """
    Return the last completed XNYS session close in UTC.

    Returns ``None`` when the exchange-calendar dependency is unavailable or no
    close can be derived for the lookback window.
    """
    try:
        import exchange_calendars as xcals
    except Exception:
        return None

    now = now_utc or pd.Timestamp.now(tz="UTC")
    now = pd.Timestamp(now)
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    else:
        now = now.tz_convert("UTC")

    cal = xcals.get_calendar("XNYS")
    start_date = (now - pd.Timedelta(days=14)).date().isoformat()
    end_date = (now + pd.Timedelta(days=1)).date().isoformat()
    sessions = cal.sessions_in_range(start_date, end_date)
    if len(sessions) == 0:
        return None

    closes: list[pd.Timestamp] = []
    for session in sessions:
        close_ts = pd.Timestamp(cal.session_close(session))
        if close_ts.tzinfo is None:
            close_ts = close_ts.tz_localize("UTC")
        else:
            close_ts = close_ts.tz_convert("UTC")
        closes.append(close_ts)

    completed = [ts for ts in closes if ts <= now]
    if not completed:
        return None
    return max(completed)
