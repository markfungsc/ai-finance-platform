from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

RESULTS_PATH = Path("experiments/results.csv")


def log_experiment(result: dict):
    result["timestamp"] = datetime.now(timezone.utc)
    df = pd.DataFrame([result])

    if RESULTS_PATH.exists():
        df_existing = pd.read_csv(RESULTS_PATH)
        df = pd.concat([df_existing, df], ignore_index=True)

    RESULTS_PATH.parent.mkdir(exist_ok=True)

    df.to_csv(RESULTS_PATH, index=False)
