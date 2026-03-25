from sklearn.ensemble import RandomForestRegressor


def train_random_forest(symbol: str, X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=200,  # number of trees in the forest
        max_depth=6,  # maximum depth of the tree
        random_state=42,  # random seed for reproducibility
        n_jobs=-1,  # use all available CPU cores
    )
    model.fit(X_train, y_train)
    get_feature_importance(model, X_train.columns)

    return model


def get_feature_importance(model, feature_names):
    importance = model.feature_importances_

    pairs = sorted(
        zip(feature_names, importance),
        key=lambda x: x[1],
        reverse=True,
    )

    for name, score in pairs:
        print(f"{name:20} {score:.4f}")
