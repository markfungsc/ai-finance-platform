import numpy as np
from sklearn.metrics import mean_absolute_error


def evaluate(predictions, actual):
    mae = mean_absolute_error(actual, predictions)

    # Check if the direction of the predictions and actual are the same, i.e. up or down.
    direction_correct = np.sign(predictions) == np.sign(actual)
    directional_accuracy = direction_correct.mean()

    print(f"MAE: {mae:.6f}")
    print(f"Directional Accuracy: {directional_accuracy:.2%}")
