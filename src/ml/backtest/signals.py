from constants import THRESHOLD


def generate_trading_signal(pred_return: float, threshold: float = THRESHOLD) -> int:
    """
    Convert predicted return into trading signal.
    1 -> long, -1 -> short, 0 -> do nothing
    """
    if pred_return > threshold:
        return 1
    elif pred_return < -threshold:
        return -1
    else:
        return 0
