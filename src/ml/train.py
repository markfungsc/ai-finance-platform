from sklearn.linear_model import LinearRegression

from ml.dataset import load_dataset


def train_model(symbol="AAPL"):
    X, y, df = load_dataset(symbol)

    split_index = int(len(X) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    model = LinearRegression()

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    return model, predictions, y_test
