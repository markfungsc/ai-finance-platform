def get_feature_importance(model, feature_names, top_n=30):
    importance = model.feature_importances_

    pairs = sorted(
        zip(feature_names, importance),
        key=lambda x: x[1],
        reverse=True,
    )

    print("\nFeature Importance\n")

    for name, score in pairs[:top_n]:
        print(f"{name:20} {score:.4f}")

    return pairs
