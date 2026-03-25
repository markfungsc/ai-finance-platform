import pandas as pd


def walk_forward_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_size: int,
    test_size: int,
    step_size: int,
):
    """
    Walk-forward time series split generator.

    Parameters
    ----------
    X : features dataframe
    y : target series
    train_size : number of rows in training window
    test_size : number of rows in testing window
    step_size : how much to move forward each iteration
    """

    start = 0
    n = len(X)

    while True:
        train_end = start + train_size
        test_end = train_end + test_size

        if test_end > n:
            break

        X_train = X.iloc[start:train_end]
        y_train = y.iloc[start:train_end]

        X_test = X.iloc[train_end:test_end]
        y_test = y.iloc[train_end:test_end]

        yield X_train, X_test, y_train, y_test

        start += step_size
