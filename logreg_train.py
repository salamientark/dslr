import pandas as pd
import numpy as np
import ft_math as ftm
import ft_datatools as ftdt
from tqdm import tqdm


# MODEL LEARNING PARAM
TARGET = "Hogwarts House"  # Class column
# Features to use for classification
FEATURES = ["Astronomy", "Herbology", "Ancient Runes"]
ALPHA = 0.1  # Learning rate
ITERATION = 100


def batch_gradient_descent(thetas: np.ndarray,
                           features: pd.DataFrame,
                           target: np.ndarray,
                           alpha: float,
                           hypothesis=None
                           ) -> np.ndarray:
    """Calculate new thetas values using gradient descent

    Parameters:
      thetas (np.ndarray): Current thetas values
      features (pd.DataFrame): Features values
      target (np.ndarray): Target values (0 or 1)
      alpha (float): Learning rate
      hypothesis (function) (optionnal): Hypothesis function to use if not
                                         provided score_function will be used

    Returns:
      float: New theta value
    """
    new_thetas = thetas.copy()
    sums = np.zeros(thetas.shape)
    f = ftdt.score_function if hypothesis is None else hypothesis
    for i, row in features.iterrows():
        prediction = f(thetas, row.values)
        error = prediction - target[i]
        for j in range(len(thetas)):
            sums[j] += error * row.values[j]
    for j in range(len(thetas)):
        sums[j] /= len(features)
        new_thetas[j] -= alpha * sums[j]
    return new_thetas


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
        true_result = ftdt.convert_classes_to_nbr(elem, target_col)
        new_thetas[elem] = batch_gradient_descent(
                thetas[elem], data, true_result, ALPHA)
    return new_thetas


def main():
    """Train model and save the thetas in a file."""
    try:
        df = pd.read_csv("dataset_train.csv")  # Load data
        filtered_df = df[FEATURES + [TARGET]]
        cleaned_df = ftdt.replace_nan(filtered_df, columns=FEATURES)
        standardized_df = ftdt.standardize_df(cleaned_df, FEATURES)
        classes = ftdt.get_class_list(standardized_df, TARGET)
        thetas = ftdt.init_thetas(classes, len(FEATURES) + 1)
        data = standardized_df.drop(columns=[TARGET])
        data.insert(0, 'x0', 1)  # Add x0 col filled with 1
        print("Training model...")
        for _ in tqdm(range(ITERATION)):
            thetas = train(standardized_df[TARGET], data, classes, thetas)
        unstandardized = ftdt.unstandardized_thetas(thetas, cleaned_df, FEATURES)
        ftdt.save_thetas(unstandardized, FEATURES)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
