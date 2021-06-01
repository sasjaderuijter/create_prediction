#!/usr/bin/env python3
import os
import sys
import yaml

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd


def run_random_forest(input_data: str) -> str:
    # First read in the data
    data = pd.read_csv(f'{input_data}.csv', sep=';')

    # Rename Columns (1-3 for Explanatory Variables, 4 for Dependent Variable)
    data.columns = ['1', '2', '3', '4']

    # Create training/testing split (90/10)
    training_data, testing_data = train_test_split(data, test_size=0.1)

    # Separate the independent and dependent variables
    y_training_data = training_data['4']
    y_testing_data = testing_data['4']
    x_training_data = training_data.drop(axis=1, columns=['4'])
    x_testing_data = testing_data.drop(axis=1, columns=['4'])

    clf = RandomForestClassifier(n_estimators=10)
    # Fit to the training data
    clf = clf.fit(x_training_data, y_training_data)
    # Predict based on the x_testing_data
    prediction = clf.predict(x_testing_data)
    # First transform 'prediction' into a Series
    prediction = pd.Series(prediction)
    # Reset the index of the y_testing_data
    y_testing_data = y_testing_data.reset_index(drop=True)
    regression_results = pd.concat([prediction, y_testing_data], axis=1).rename(
        columns={0: 'Prediction', '4': 'Actual'})

    # Then write the results data to a csv
    OUTPUT_LOCATION = 'random_forest_results.csv'
    regression_results.to_csv(OUTPUT_LOCATION, sep=';', index=False)

    return OUTPUT_LOCATION


def run_regression(input_data: str) -> str:

    # First read in the data
    data = pd.read_csv(f'{input_data}.csv', sep=';')

    # Rename Columns (1-3 for Explanatory Variables, 4 for Dependent Variable)
    data.columns = ['1', '2', '3', '4']

    # Run regression and format results
        # Create training/testing split (90/10)
    training_data, testing_data = train_test_split(data, test_size=0.1)

    # Separate the independent and dependent variables
    y_training_data = training_data['4']
    y_testing_data = testing_data['4']
    x_training_data = training_data.drop(axis=1, columns=['4'])
    x_testing_data = testing_data.drop(axis=1, columns=['4'])

    # Create Regression Object
    # Fit to training data
    regression = LogisticRegression().fit(X=x_training_data, y=y_training_data)
    # Predict based on the x_testing_data
    prediction = regression.predict(x_testing_data)

    # First transform 'prediction' into a Series
    prediction = pd.Series(prediction)
    # Reset the index of the y_testing_data
    y_testing_data = y_testing_data.reset_index(drop=True)
    regression_results = pd.concat([prediction, y_testing_data], axis=1).rename(
        columns={0: 'Prediction', '4': 'Actual'})

    # Then write the results data to a csv
    OUTPUT_LOCATION = 'regression_results.csv'
    regression_results.to_csv(OUTPUT_LOCATION, sep=';', index=False)

    return OUTPUT_LOCATION


if __name__ == "__main__":
    command = sys.argv[1]
    argument = os.environ["INPUT_LOCATION"]
    functions = {
        "run_regression": run_regression,
        "run_random_forest": run_random_forest
    }
    output = functions[command](argument)
    print(yaml.dump({"output": output}))

