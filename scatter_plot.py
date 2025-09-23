import sys
import pandas as pd
import ft_datatools as ftdt
from matplotlib import pyplot as plt


def draw_scatter_plot(standardized: dict):
    """Draw unique histogram for each feature in the standardized data matrix.

    Parameters:
      standardized (dict): Standardized data matrix.
    """
    for feature, data in standardized.items():
        plt.scatter(range(len(data)), data, alpha=0.5, label=feature)
    plt.title("Scatter plot of features")
    plt.legend(loc='upper right')
    plt.xlabel("Index")
    plt.ylabel("Standardized value")
    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()
    plt.ylim(-10, 10)
    plt.show()


def draw_unique_scatter_plot(standardized: dict):
    """Draw unique histogram for each feature in the standardized data matrix.

    Parameters:
      standardized (dict): Standardized data matrix.
    """
    fig, axes = plt.subplots(4, 5, figsize=(12, 12))
    fig.subplots_adjust(hspace=0.7, wspace=0.7, left=0.07,
                        right=0.93, top=0.95, bottom=0.1)
    x, y, count = 0, 0, 0
    keys = list(standardized.keys())
    for i_f, f1 in enumerate(keys):
        data1 = standardized[f1]
        for j_f in range(i_f + 1, len(keys)):
            f2 = keys[j_f]
            data2 = standardized[keys[j_f]]
            axes[x, y].scatter(data1, data2, alpha=0.5)
            axes[x, y].set_xlabel(f1, fontsize=8)
            axes[x, y].set_ylabel(f2, fontsize=8)
            y += 1
            count += 1
            if y == 5:
                y = 0
                x += 1
            if count == 20:
                manager = plt.get_current_fig_manager()
                manager.full_screen_toggle()
                plt.show()
                fig, axes = plt.subplots(4, 5, figsize=(12, 12))
                fig.subplots_adjust(hspace=0.7, wspace=0.7, left=0.07,
                                    right=0.93, top=0.95, bottom=0.1)
                x, y, count = 0, 0, 0
    while y < 5:
        axes[x, y].set_visible(False)
        y += 1
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
        matrix = {}
        for feature in features:
            col = df[feature].tolist()
            matrix[feature] = col
        draw_unique_scatter_plot(matrix)
        plt.show()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main(len(sys.argv), sys.argv)
