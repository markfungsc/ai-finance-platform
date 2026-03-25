from pathlib import Path

import joblib


def save_model(model, path: str):
    """Save a trained model to a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)  # ensure directory exists
    joblib.dump(model, path)
    print(f"[INFO] Model saved to {path}")


def load_model(path: str):
    """Load a previously saved model."""
    return joblib.load(path)
