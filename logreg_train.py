import pandas as pd
import numpy as np
import ft_datatools as ftdt
import ft_math as ftm
from tqdm import tqdm


# MODEL LEARNING PARAM
TARGET = "Hogwarts House"  # Class column
# Features to use for classification
FEATURES = ["Astronomy", "Herbology", "Ancient Runes"]
ALPHA = 0.1  # Learning rate
ITERATION = 1000


def gradient_descent(thetas: np.ndarray,
                     features: pd.DataFrame,
                     target: np.ndarray,
                     alpha: float
                     ) -> np.ndarray:
    """Calculate new thetas values using gradient descent

    Parameters:
      thetas (np.ndarray): Current thetas values
      features (pd.DataFrame): Features values
      target (np.ndarray): Target values (0 or 1)
      alpha (float): Learning rate

    Returns:
      float: New theta value
    """
    new_thetas = thetas.copy()
    sums = np.zeros(thetas.shape)
    for i, row in features.iterrows():
        prediction = ftdt.sigmoid(thetas, row.values)
        error = prediction - target[i]
        for j in range(len(thetas)):
            sums[j] += error * row.values[j]
    for j in range(len(thetas)):
        sums[j] /= len(features)
        new_thetas[j] -= alpha * sums[j]
    return new_thetas


def convert_classes_to_nbr(class_name: str, data: pd.Series) -> pd.Series:
    """Convert class names to numerical values

    Parameters:
    class_name (str): Class name
    data (pd.Series): Series with class names

    Returns:
    pd.Series: Series with numerical values
    """
    converted_col = (data == class_name).astype(int)
    return converted_col.astype(int)


def train(
        target_col: pd.Series,
        data: pd.DataFrame,
        classes: list,
        thetas: dict
        ) -> dict:
    """Train model and update thetas values

    Parameters:
    target_col (pd.Series): Target column
    data (pd.DataFrame): Features data
    classes (list): List of class names
    thetas (dict): Thetas values for each class

    Returns:
    dict: Updated thetas values
    """
    # Split target col from other data
    new_thetas = thetas.copy()
    for elem in classes:
        true_result = convert_classes_to_nbr(elem, target_col)
        new_thetas[elem] = gradient_descent(
                thetas[elem], data, true_result, ALPHA)
    return new_thetas


def init_thetas(classes: list, feature_nbr: int) -> dict:
    """Initialize thetas dictionary with zeros

    Parameters:
    classes (list): List of class names
    feature_nbr (int): Number of features
    """
    thetas = {}
    for elem in classes:
        thetas[elem] = np.zeros(feature_nbr)
    return thetas


def unstandardized_thetas(thetas: dict, df: pd.DataFrame) -> dict:
    """Convert standardized thetas to unstandardized thetas

    Parameters:
      thetas (dict): Standardized thetas
      df (pd.DataFrame): Dataframe with original data

    Returns:
      dict: Unstandardized thetas
    """
    means = {}
    std = {}
    for feature in FEATURES:
        means[feature] = ftm.ft_mean(df[feature].to_list())
        std[feature] = ftm.ft_std(df[feature].to_list())
    unstandardized = {}
    for cls, theta in thetas.items():
        unstandardized[cls] = theta.copy()
        for i in range(1, len(theta)):
            unstandardized[cls][i] = theta[i] / std[FEATURES[i - 1]]
            unstandardized[cls][0] -= (
                theta[i] * means[FEATURES[i - 1]]) / std[FEATURES[i - 1]]
    return unstandardized


def save_thetas(thetas: dict) -> None:
    """Save thetas to a file

    Parameters:
    thetas (dict): Thetas to save
    """
    with open("thetas.csv", "w") as f:
        f.write("Class,Bias," + ",".join(FEATURES) + "\n")
        for cls, theta in thetas.items():
            f.write(cls + "," + ",".join([str(t) for t in theta]) + "\n")


def main():
    """Train model and save the thetas in a file."""
    try:
        df = pd.read_csv("dataset_train.csv")  # Load data
        filtered_df = df[FEATURES + [TARGET]]
        cleaned_df = ftdt.remove_missing(filtered_df)
        standardized_df = ftdt.standardize_df(cleaned_df, FEATURES)
        classes = ftdt.get_class_list(standardized_df, TARGET)
        thetas = init_thetas(classes, len(FEATURES) + 1)
        data = standardized_df.drop(columns=[TARGET])
        data.insert(0, 'x0', 1)  # Add x0 col filled with 1
        for _ in tqdm(range(ITERATION)):
            thetas = train(standardized_df[TARGET], data, classes, thetas)
        unstandardized = unstandardized_thetas(thetas, cleaned_df)
        print(unstandardized)
        save_thetas(unstandardized)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
