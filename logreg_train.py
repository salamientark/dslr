import pandas as pd
import ft_datatools as ftdt
from tqdm import tqdm


# MODEL LEARNING PARAM
TARGET = "Hogwarts House"  # Class column
# Features to use for classification
FEATURES = ["Astronomy", "Herbology", "Ancient Runes"]
ALPHA = 0.1  # Learning rate
ITERATION = 100


def train(
        target_col: pd.Series,
        data: pd.DataFrame,
        classes: list,
        thetas: dict,
        alpha: float = 0.1,
        algo=ftdt.batch_gradient_descent,
        hypothesis=ftdt.score_function,
        batch_size: int = 32
        ) -> dict:
    """Train model and update thetas values

    Parameters:
    target_col (pd.Series): Target column
    data (pd.DataFrame): Features data
    classes (list): List of class names
    thetas (dict): Thetas values for each class
    alpha (float, optional): Learning rate. Defaults to 0.1.
    algo (function, optional): Algorithm to use for training. Defaults to
                               None, which uses batch_gradient_descent.
    hypothesis (function, optional): Hypothesis function to use. Defaults to
                                     None, which uses score_function.
    batch_size (int, optional): Size of the batch for mini-batch gradient

    Returns:
    dict: Updated thetas values
    """
    # Split target col from other data
    new_thetas = thetas.copy()
    for elem in classes:
        true_result = ftdt.convert_classes_to_nbr(elem, target_col)
        if algo == ftdt.mini_batch_gradient_descent:
            new_thetas[elem] = algo(thetas[elem], data, true_result, alpha,
                                    hypothesis=hypothesis,
                                    batch_size=batch_size)
        else:
            new_thetas[elem] = algo(thetas[elem], data, true_result, alpha,
                                    hypothesis)
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
            thetas = train(standardized_df[TARGET], data, classes, thetas,
                           algo=ftdt.batch_gradient_descent,
                           hypothesis=ftdt.sigmoid)
            # thetas = train(standardized_df[TARGET], data, classes, thetas,
            #                algo=ftdt.mini_batch_gradient_descent,
            #                hypothesis=ftdt.sigmoid, batch_size=32)
        unstandardized = ftdt.unstandardized_thetas(thetas, cleaned_df,
                                                    FEATURES)
        ftdt.save_thetas(unstandardized, FEATURES)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
