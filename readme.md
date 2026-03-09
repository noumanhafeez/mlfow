# Mushroom Classification with MLflow Experiment Tracking

## Project Overview

## This project demonstrates how to use MLflow to manage the machine learning lifecycle including experiment tracking, hyperparameter tuning, model comparison, and model versioning.

## The goal of this project is to predict whether a mushroom is edible or poisonous using the Mushroom dataset while tracking multiple experiments using MLflow.

## This project illustrates how ML engineers organize machine learning experiments in real-world workflows.

# What is MLflow

## MLflow is an open-source platform designed to manage the complete machine learning lifecycle.

## It helps machine learning engineers track experiments, log parameters and metrics, save trained models, compare model performance, and manage model versions.

## MLflow ensures experiments are reproducible, organized, and easy to analyze.

# MLflow Core Concepts

## Experiments

### An experiment is a container that stores multiple runs for a particular machine learning problem.

### Example experiment name used in this project:

```
RandomForest_Experiment
```

### Each run inside the experiment represents one training attempt.

## Runs

### A run represents a single model training execution.

### Example run names:

```
RF_100_10
RF_150_5
```

### Each run stores parameters, metrics, artifacts, and logs.

## Parameters

### Parameters are configuration values used during model training.

### Example parameters:

```
n_estimators = 100
max_depth = 10
```

### These parameters are logged using:

```
mlflow.log_param()
```

## Metrics

### Metrics represent the performance of the trained model.

### Example metric used in this project:

```
accuracy
```

### Metrics are logged using:

```
mlflow.log_metric()
```

## Artifacts

### Artifacts are files generated during model training.

### Examples include trained models, plots, confusion matrices, and datasets.

### Artifacts are stored automatically inside the mlruns directory.

# Project Workflow

## The workflow implemented in this project follows these steps:

```
Dataset
   ↓
Data Preprocessing
   ↓
Model Training
   ↓
MLflow Experiment Tracking
   ↓
Hyperparameter Testing
   ↓
Model Logging
   ↓
Model Versioning
```

# Dataset

## This project uses the Mushroom Dataset which contains categorical features describing mushroom characteristics.

### Example features include:

## cap-shape

## cap-color

## odor

## gill-size

## stalk-shape

### Target variable:

```
class
```

### Where:

```
e = edible
p = poisonous
```

# Data Preprocessing

## The dataset contains categorical features which must be converted into numeric values for machine learning models.

## All categorical columns are encoded using LabelEncoder.

### Example transformation:

```
odor: almond → 0
odor: foul → 1
```

## This transformation allows the models to process the data effectively.

# Models Used

## Multiple machine learning models were tested in this project.

## Gaussian Naive Bayes

### Gaussian Naive Bayes is a probabilistic classifier based on Bayes theorem.

## Logistic Regression

### Logistic Regression is a linear model commonly used for binary classification problems.

## Random Forest

### Random Forest is an ensemble learning algorithm that builds multiple decision trees and combines their predictions.

# Hyperparameter Experiments

## Different hyperparameter combinations were tested to evaluate model performance.

### Example Random Forest parameters tested:

```
n_estimators = [100, 150]
max_depth = [5, 10]
```

## Each combination creates a separate MLflow run which allows easy comparison of results.

# MLflow Logging

## During each experiment run the following information is logged.

## Parameters

```
model_type
n_estimators
max_depth
```

## Metrics

```
accuracy
```

## Artifacts

```
trained model
```

# MLflow Directory Structure

## After running experiments MLflow automatically generates the following structure:

```
mlruns/

 ├── experiment_id/
 │     ├── run_id/
 │     │     ├── artifacts/
 │     │     ├── metrics/
 │     │     ├── params/
 │     │     └── meta.yaml
 │
 └── mlflow.db
```

## Explanation of directories

### mlruns stores all experiment results

### run_id uniquely identifies each run

### artifacts store models and files generated during training

### params store logged parameters

### metrics store evaluation results

### mlflow.db stores experiment metadata

# Viewing Experiments in MLflow UI

## The MLflow user interface allows visualization and comparison of experiments.

### Start MLflow UI:

```
mlflow ui
```

### Open the UI in browser:

```
http://127.0.0.1:5000
```

## The UI allows users to compare runs, inspect parameters, view metrics, and manage models.

# Model Versioning

## Models are registered in MLflow using the registered_model_name parameter.

### Example:

```
registered_model_name = "RandomForest"
```

## Each time a model is logged MLflow automatically creates a new version.

### Example model versions:

```
RandomForest
 Version 1
 Version 2
 Version 3
```

## This allows tracking improvements across different experiments.

# Key Learnings from this Project

## Implementation of MLflow experiment tracking

## Logging parameters and evaluation metrics

## Performing hyperparameter experiments

## Comparing multiple machine learning models

## Managing model artifacts

## Tracking model versions

## Creating reproducible machine learning experiments

# Future Improvements

## Adding more machine learning models such as XGBoost or LightGBM

## Logging additional evaluation metrics

## Saving confusion matrices and performance plots

## Automating hyperparameter optimization

## Deploying the best model using an API

# Technologies Used

## Python

## Scikit-learn

## MLflow

## Pandas

# Author

## Nouman Hafeez

## Machine Learning and Data Science Enthusiast
