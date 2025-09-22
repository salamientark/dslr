import sys
import pandas as pd
import ft_math as ftm




def main(ac: int,av :list):
    """Describe the dataset given as parameter.

    Parameters:
      ac (int): Number of command line arguments.
      av (list): List of command line arguments.
    """
    try:
        if (ac != 2):
            raise Exception("Usage: describe.py <dataset_path>")
        df = pd.read_csv(av[1])  # Dataframe
        features = df.columns.tolist()
        count = len(df[features[0]].tolist())
        means = stds = mins = q1s= q2s = q3s = maxs = []
        for feature in features:
            col = df[feature].tolist()
            means.append(ftm.ft_mean(col, count=count))
            stds.append(ftm.ft_std(col, count=count))
            mins.append(ftm.ft_min(col))
            q1s.append(ftm.ft_q1(col, count=count))
            print("ok")
            q2s.append(ftm.ft_q2(col, count=count))
            q3s.append(ftm.ft_q3(col, count=count))
            maxs.append(ftm.ft_max(col))
        print("Feature\tMean\tStd\tMin\tQ1\tQ2\tQ3\tMax")
        print("-------\t----\t---\t---\t--\t--\t--\t---")
        print("-------\t----\t---\t---\t--\t--\t--\t---")
        print(f"{features[0]}\t{means[0]:.2f}\t{stds[0]:.2f}\t{mins[0]}\t{q1s[0]}\t{q2s[0]}\t{q3s[0]}\t{maxs[0]}")
        print(f"Dataframe: {type(df)}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main(len(sys.argv), sys.argv)
