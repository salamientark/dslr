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
    """Draw merged histogram from each feature in the standardized data matrix.

    Parameters:
      standardized (dict): Standardized data matrix.
    """
    colors = {"Arithmancy": "blue", "Potions": "red",
              "Care of Magical Creatures": "green"}
    for feature, data in standardized.items():
        if feature in colors:
            plt.hist(data, bins=30, alpha=0.5, label=feature,
                     color=colors[feature])
    plt.legend(loc='upper right')
    plt.xlabel("Standardized value")
    plt.ylabel("Frequency")
    plt.title("Histogram of standardized features")
    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()
    plt.show()


def draw_unique_histogram(standardized: dict):
    """Draw unique histogram for each feature in the standardized data matrix.

    Parameters:
      standardized (dict): Standardized data matrix.
    """
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.subplots_adjust(hspace=0.5, wspace=0.5, left=0.1, right=0.9,
                        top=0.95, bottom=0.05)
    i, j = 0, 0
    for feature, data in standardized.items():
        axes[i, j].hist(data, bins=30, alpha=0.5, label=feature)
        axes[i, j].set_title(f"{feature} repartition")
        j += 1
        if j == 4:
            j = 0
            i += 1
    while j < 4:
        axes[i, j].set_visible(False)
        j += 1
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
        features = ftdt.get_numerical_features(df, exclude=['Index'])
        standardized = {}  # Standardized data matrix
        for feature in features:
            col = ftdt.filter_col(df[feature].tolist())
            standardized[feature] = standardize_array(col)
        draw_unique_histogram(standardized)
        draw_merged_histogram(standardized)
        plt.show()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main(len(sys.argv), sys.argv)
