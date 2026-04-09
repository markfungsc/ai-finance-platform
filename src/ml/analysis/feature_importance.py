from ml.analysis.explanations import global_feature_importance


def get_feature_importance(model, feature_names, top_n=30):
    df = global_feature_importance(model, list(feature_names), top_n=top_n)
    pairs = list(zip(df["feature"].tolist(), df["importance"].tolist()))

    print("\nFeature Importance\n")

    for name, score in pairs:
        print(f"{name:20} {score:.4f}")

    return pairs
