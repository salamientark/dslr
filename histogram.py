import sys
import pandas as pd
import ft_math as ftm
import ft_datatools as ftdt
from matplotlib import pyplot as plt


def standardize_array(array,
                      mean: float | None = None,
                      std: float | None = None
                      ) -> list:
    """Standardize a list of numerical values.

    Parameters:
    array (list): List of numerical values.
    mean (float | None): Mean of the list. If None, it will be computed.
    std (float | None): Standard deviation of the list. If None, it will be
        computed.

    Returns:
    list: Standardized list.
    """
    standardized = []
    m = mean if mean is not None else ftm.ft_mean(array)
    s = std if std is not None else ftm.ft_std(array)
    for elem in array:
        standardized.append((elem - m) / s if s != 0 else 0)
    return standardized


def draw_merged_histogram(standardized: dict):
    """Draw histogram for each feature in the standardized data matrix.

    Parameters:
      standardized (dict): Standardized data matrix.
    """
    for feature, data in standardized.items():
        plt.hist(data, bins=30, alpha=0.5, label=feature)
    plt.legend(loc='upper right')
    plt.xlabel("Standardized value")
    plt.ylabel("Frequency")
    plt.title("Histogram of standardized features")
    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()
    plt.show()


def main(ac: int, av: list):
    """Describe the dataset given as parameter.

    Parameters:
      ac (int): Number of command line arguments.
      av (list): List of command line arguments.
    """
    try:
        if (ac != 2):
            raise Exception("Usage: describe.py <dataset_path>")
        df = pd.read_csv(av[1])  # Dataframe
        features = ftdt.get_numerical_features(df)
        standardized = {}  # Standardized data matrix
        for feature in features:
            col = ftdt.filter_col(df[feature].tolist())
            standardized[feature] = standardize_array(col)
        draw_merged_histogram(standardized)
        plt.show()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main(len(sys.argv), sys.argv)
