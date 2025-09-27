import sys
import pandas as pd
import ft_datatools as ftdt
import seaborn as sns
from matplotlib import pyplot as plt


# COLORS
RED = '\033[91m'
GREEN = '\033[92m'
BLUE = '\033[94m'
RESET = '\033[0m'


def pairplot(df: pd.DataFrame, features: list, target_col: str | None = None):
    """Draw pairplot for each features in the dataframe.

    Parameters:
      df (pd.DataFrame): Dataframe.
      features (list): List of numerical features name.
      targets (str) (optionnal): Target col name
    """
    # Change global plot param
    plt.rcParams.update({
        'font.size': 8,
        'axes.titlesize': 10,
        'axes.labelsize': 8,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'legend.fontsize': 8,
        'legend.markerscale': 2,
    })
    if target_col is not None and len(ftdt.get_class_list(
                                      df, target_col)) > 1:
        plot = sns.pairplot(
            df, hue=target_col, vars=features,
            plot_kws={'alpha': 0.5, 's': 5})
    else:
        plot = sns.pairplot(df[features],
                            plot_kws={'alpha': 0.5, 's': 5})
    for ax in plot.axes.flatten():
        if ax is not None and ax.get_ylabel():
            ax.yaxis.label.set_rotation(45)
            current_label = ax.get_ylabel()
            if len(current_label) > 4:
                ax.set_ylabel(current_label[:4] + '...')
    plt.subplots_adjust(hspace=0.7, wspace=0.7, left=0.1,
                        right=0.9, top=0.95, bottom=0.1)
    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()
    plt.show()
    plt.rcParams.update(plt.rcParamsDefault)


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
        target = "Hogwarts House"
        cleaned_df = ftdt.select_columns(df, features + [target])
        cleaned_df = ftdt.remove_missing(cleaned_df, exclude=[target])
        print(f"Generating pairplot... {RESET}", flush=True, end="")
        pairplot(cleaned_df, features, target)
        print(f"{GREEN}Success{RESET}")
    except Exception as e:
        print(f"{RED}Error{RESET}: {e}")


if __name__ == "__main__":
    main(len(sys.argv), sys.argv)
