import pandas as pd


def get_numerical_features(
        df: pd.DataFrame,
        exclude: list = []
        ) -> list:
    """Get the numerical features name only from a dataframe.

    Parameters:
      df (pd.DataFrame): Dataframe.
      exclude (list) (optional): List of columns to exclude.

    Returns:
      list: List of numerical features.
    """
    columns = df.columns.tolist()
    filtered_features = []
    for col in columns:
        if col in exclude:
            continue
        for elem in df[col].tolist():
            if elem is None or pd.isna(elem):
                continue
            if isinstance(elem, (int, float)):
                filtered_features.append(col)
            break
    return filtered_features


def get_class_list(df: pd.DataFrame, col: str) -> list:
    """Get the list of unique values in the specified column

    Parameters:
      df (pd.DataFrame): Dataframe.
      col (str): col column name.

    Returns:
      list: List of unique values in the col column. Empty if no class
    """
    try:
        target_col = df[col]
        result = []
        for elem in target_col:
            if pd.isna(elem):
                continue
            if elem not in result:
                result.append(elem)
        return result
    except KeyError as e:
        raise Exception(f"Column '{col}' not found in the dataframe.") from e


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


def remove_missing(df: pd.DataFrame, exclude: list[str] = []) -> pd.DataFrame:
    """Remove rows with missing values in the dataframe.

    Parameters:
      df (pd.DataFrame): Dataframe.
      exclude (list) (optional): List of columns to exclude.

    Returns:
      pd.DataFrame: Dataframe without missing values.
    """
    cleaned_df = df.copy()
    for i, row in cleaned_df.iterrows():
        filtered_row = row.drop(exclude)
        if filtered_row.isnull().any():
            cleaned_df = cleaned_df.drop(i)
    return cleaned_df


def classify(df: pd.DataFrame, target_col: str, features: list
             ) -> dict[str, pd.DataFrame]:
    """Classify the dataframe into multiple dataframes based on the target

    Parameters:
      df (pd.DataFrame): Dataframe.
      target (str): Target column name.
      features (list): List of numerical features name.

    Returns:
      dict[pd.DataFrame]: Dictionary of dataframes classified by class.
    """
    target = get_class_list(df, target_col)
    res = {key: [] for key in target}
    for i, row in df.iterrows():
        numeric_row = row.drop(target_col)
        res[row[target_col]].append(numeric_row)
    for key in res:
        res[key] = pd.DataFrame(res[key], columns=features)
    return res


def select_columns(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """Select only the specified columns from the dataframe.

    Parameters:
      df (pd.DataFrame): Dataframe.
      features (list): List of numerical features name.

    Returns:
      pd.DataFrame: Dataframe with only the specified columns.
    """
    return df[features]
