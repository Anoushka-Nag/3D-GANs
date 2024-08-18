import pandas as pd


def split_train_test(
        df: pd.DataFrame,
        column: str = "split",
        train_name: str = "train",
        test_name: str = "test"
):
    train_idx = df[column] == train_name
    test_idx = df[column] == test_name

    train_df = df[train_idx]
    test_df = df[test_idx]

    return train_df, test_df


def get_uniques(
        df: pd.DataFrame,
        column: str = "class"
):
    uniques = df[column].unique()
    return sorted(list(uniques))


def get_class_df(
        df: pd.DataFrame,
        class_column: str = "class",
        class_name: str = "class"
) -> pd.DataFrame:
    return df[df[class_column] == class_name]


def get_time_elapsed(start_time, end_time):
    elapsed = end_time - start_time
    minutes = elapsed // 60
    secs = elapsed % 60
    return minutes, secs
