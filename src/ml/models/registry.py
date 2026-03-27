from ml.models.logistic_regression import train_logistic_regression
from ml.models.random_forest import train_random_forest

MODEL_REGISTRY = {
    "logistic_regression": train_logistic_regression,
    "random_forest": train_random_forest,
}
