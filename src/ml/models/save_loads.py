from pathlib import Path

import joblib

from log_config import get_logger

logger = get_logger(__name__)


def save_model(model, path: str):
    """Save a trained model to a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)  # ensure directory exists
    joblib.dump(model, path)
    logger.info("Model saved to %s", path)


def save_feature_columns(feature_cols, path: str) -> None:
    """Persist the ordered list of feature column names used at training time."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(list(feature_cols), path)
    logger.info("Feature columns saved to %s", path)


def load_model(path: str):
    """Load a previously saved model."""
    return joblib.load(path)


def load_feature_columns(path: str) -> list[str]:
    """Load the ordered feature column names saved at training time."""
    cols = joblib.load(path)
    return list(cols)


def save_scaler(scaler, path: str) -> None:
    """Persist a fitted sklearn scaler (e.g. StandardScaler) for inference."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)
    logger.info("Scaler saved to %s", path)


def load_scaler(path: str):
    """Load a previously saved scaler."""
    return joblib.load(path)
