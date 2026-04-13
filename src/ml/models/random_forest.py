import os

from sklearn.ensemble import RandomForestClassifier


def _rf_n_jobs() -> int:
    raw = os.environ.get("SKLEARN_RF_N_JOBS", "2").strip()
    return int(raw)


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=200,  # number of trees in the forest
        max_depth=6,  # maximum depth of the tree
        random_state=42,  # random seed for reproducibility
        n_jobs=_rf_n_jobs(),
    )
    model.fit(X_train, y_train)

    return model
