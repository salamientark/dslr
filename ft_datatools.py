import pandas as pd


def get_numerical_features(df: pd.DataFrame) -> list:
    """Get the numerical features name only from a dataframe.

    Parameters:
      df (pd.DataFrame): Dataframe.

    Returns:
      list: List of numerical features.
    """
    columns = df.columns.tolist()
    filtered_features = []
    for col in columns:
        if (col == 'Index'):
            continue
        for elem in df[col].tolist():
            if elem is None or pd.isna(elem):
                continue
            if isinstance(elem, (int, float)):
                filtered_features.append(col)
            break
    return filtered_features


def filter_col(col: list) -> list:
    """Filter a column to keep only numerical values.

    Parameters:
      col (list): Column to filter.

    Returns:
      list: Filtered column.
    """
    i = 0
    lim = len(col)
    while i < lim:
        if col[i] is None or pd.isna(col[i]):
            del col[i]
            lim -= 1
            continue
        i += 1
    return col
