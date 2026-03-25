import numpy as np
from sklearn.metrics import mean_absolute_error


def evaluate_model(predictions, actual, *, verbose: bool = True) -> dict[str, float]:
    mae = mean_absolute_error(actual, predictions)

    # Check if the direction of the predictions and actual are the same, i.e. up or down.
    direction_correct = np.sign(predictions) == np.sign(actual)
    directional_accuracy = direction_correct.mean()

    if verbose:
        print(f"MAE: {mae:.6f}")
        print(f"Directional Accuracy: {directional_accuracy:.2%}")

    return {
        "mae": float(mae),
        "directional_accuracy": float(directional_accuracy),
    }
