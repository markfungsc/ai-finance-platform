from database.queries import fetch_features, fetch_features_z
from ml.helpers.merge_features import merge_features_with_target


def load_dataset(symbol: str, *, debug_merge: bool = False):
    df = fetch_features(symbol)
    df_z = fetch_features_z(symbol)

    X, y, df_merged = merge_features_with_target(
        df, df_z, target_shift=5, debug=debug_merge
    )

    return X, y, df_merged
