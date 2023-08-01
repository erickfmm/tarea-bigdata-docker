import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

def main() -> None:
    """
    Train a linear regresión to predict the wine quality.
    The source of the dataset is:
    - https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv
    """
    df = pd.read_csv("winequality-red.csv", delimiter=";")

    X_columns = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", 
                 "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]
    
    y_column = ["quality"]
    
    print("Starting train test split ...")
    X_train, X_test, y_train, y_test = \
        train_test_split(df[X_columns], df[y_column], test_size=0.33, random_state=42)
    print(f"len(X_train): {len(X_train)} | len(X_test): {len(X_test)}")
    
    print("Starting training linear regresion ...")
    reg = LinearRegression().fit(X_train, y_train)
    y_test_predicted = reg.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_test_predicted)
    print(f"mean_absolute_error: {mae}")

if __name__ == "__main__":
    main()
