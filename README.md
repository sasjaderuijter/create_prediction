# create_prediction
_Universiteit van Amsterdam - Web Services and Cloud-Based Systems: assignment 5_

This Brane package is used as the prediction modelling part of the pipeline.
It uses the output of the Brane package etl.

## Installation
Import the package as follows:
```shell
$ brane import sasjaderuijter/create_prediction
```
## Running package
Run a regression model on a trainingset, using the example dataset _titanic_trainset.csv_
```shell
run_regression("/data/titanic_trainset");
```

Run a random forest model on a trainingset, using the example dataset _titanic_trainset.csv_
```shell
run_random_forest("/data/titanic_trainset");
```
