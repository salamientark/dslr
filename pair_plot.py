import sys
import pandas as pd
import ft_datatools as ftdt
import seaborn as sns
from matplotlib import pyplot as plt


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
        df.columns = [col[:10] for col in df.columns]
        features = ftdt.get_numerical_features(df)
        matrix = {}
        for feature in features:
            col = df[feature].tolist()
            matrix[feature] = col
        pair = sns.pairplot(df[features], plot_kws={'alpha': 0.5, 's': 5})
        for ax in pair.axes.flatten():
            ax.tick_params(labelsize=6)  # change tick label size
            ax.set_xlabel(ax.get_xlabel(), fontsize=6)  # x-label font size
            ax.set_ylabel(ax.get_ylabel(), fontsize=6)  # y-label font size
        plt.subplots_adjust(hspace=0.7, wspace=0.7, left=0.07,
                            right=0.93, top=0.95, bottom=0.1)
        manager = plt.get_current_fig_manager()
        manager.full_screen_toggle()
        plt.show()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main(len(sys.argv), sys.argv)
