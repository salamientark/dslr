import sys
import pandas as pd
import numpy as np
import ft_datatools as ftdt
from tqdm import tqdm


def predict(weights_df: pd.DataFrame, row: list) -> str:
    """Predict the class of a given row using the weights

    Parameters:
      weights (pd.DataFrame): Weights dataframe
      row (list): Row to predict

    Returns:
      str: Predicted class
    """
    prediction = None
    result = {}
    for i, cls in weights_df.iterrows():
        result[cls['Class']] = ftdt.sigmoid(
                cls.drop(['Class']), np.array([1] + row))
        if prediction is None or result[cls['Class']] > result[prediction]:
            prediction = cls['Class']
    return prediction


def main(ac: int, av: list):
    """Main function for logreg_predict.py"""
    try:
        if ac != 3:
            raise Exception("Usage: python logreg_predict.py dataset_test.csv"
                            " <weights.csv>")
        # if av[1] != "dataset_test.csv":
        #     raise Exception("dataset_test.csv file name is required as"
        #                     " the first argument.")
        df = pd.read_csv(av[1])
        weights = pd.read_csv(av[2])
        features = weights.drop(columns=['Class', 'Bias']).columns.tolist()
        data = ftdt.replace_nan(df[features])
        with open('houses.csv', 'w') as f:
            f.write("Index,Hogwarts House\n")
            for i, row in tqdm(data.iterrows(), total=len(data)):
                predicted_class = predict(weights, row.tolist())
                f.write(f"{i},{predicted_class}\n")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main(len(sys.argv), sys.argv)
