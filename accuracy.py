import os
import sys
import subprocess
import pandas as pd
from sklearn.metrics import accuracy_score


PYTHON_PATH = "/home/madlab/.env/.venv/bin/python"


def main(ac: int, av: list):
    """Main function for logreg_predict.py"""
    try:
        if ac != 3:
            raise Exception("Usage: python logreg_predict.py "
                            "<dataset_train>.csv <weights.csv>")
        # if av[1] != "dataset_test.csv":
        #     raise Exception("dataset_test.csv file name is required as"
        #                     " the first argument.")
        env_copy = os.environ.copy()
        print("Predicting values...")
        result = subprocess.run([PYTHON_PATH, "logreg_predict.py",
                                "dataset_train.csv", "thetas.csv"],
                                env=env_copy)
        if result.returncode != 0:
            raise Exception("Error while running logreg_predict.py")
        df = pd.read_csv(av[1])
        true_value = df['Hogwarts House']
        houses_df = pd.read_csv('houses.csv')
        predicted_value = houses_df['Hogwarts House']
        accuracy = accuracy_score(true_value, predicted_value)
        print(f"Accuracy: {accuracy * 100:.2f}%")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main(len(sys.argv), sys.argv)
