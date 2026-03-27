from sklearn.ensemble import RandomForestClassifier


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=200,  # number of trees in the forest
        max_depth=6,  # maximum depth of the tree
        random_state=42,  # random seed for reproducibility
        n_jobs=-1,  # use all available CPU cores
    )
    model.fit(X_train, y_train)

    return model
