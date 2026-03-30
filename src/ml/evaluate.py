import numpy as np
from sklearn.metrics import mean_absolute_error

from log_config import get_logger

logger = get_logger(__name__)


def evaluate_model(predictions, actual, *, verbose: bool = True) -> dict[str, float]:
    mae = mean_absolute_error(actual, predictions)

    # Check if the direction of the predictions and actual are the same, i.e. up or down.
    direction_correct = np.sign(predictions) == np.sign(actual)
    directional_accuracy = direction_correct.mean()

    if verbose:
        logger.info("MAE: %.6f", mae)
        logger.info("Directional Accuracy: %.2f%%", directional_accuracy * 100.0)

    return {
        "mae": float(mae),
        "directional_accuracy": float(directional_accuracy),
    }
