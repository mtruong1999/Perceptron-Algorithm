# Perceptron Algorithm
## Overview
This code implements the perceptron algorithm and runs a set of classification experiments using the data found in `data/data.mat`. The input data contains 40,000 rows and 9 columns where each row represents a data instance, the first column contains the class labels, and the rest of the columns contain the attribute values. There are four classes (labeled 1, 2, 3, and 4). The 3 experiments are explained as follows:
1. First, we combine classes 1 and 2 and combine classes 3 and 4 to create a binary classification problem from 4 classes. We then run 5-fold cross validation and report the accuracy results for each fold.
2. Second, we create a similar classifier as in the first experiment but train on a PCA reduced representation of the input data (using the two principle components).
3. Third, we create a classifier to classify between labels 1 and 2 and a classifier to classify between labels 3 and 4. We then combine these two binary classifiers with the binary classifier from experiment 1 and create a decision tree where the first node decides which of the two binary classifiers to use.

The implementation of the perceptron algorithm can be found in `perceptron.py`. The experiments entry point can be found in `run_experiments.py`. Experiment results can be found in `results/results.txt`.

## Setup
1. Installation of [Python 3.7](https://www.python.org/downloads/) or later required

2. To install dependencies, in the `HW3_linear_classifier/` directory, run
```
pip install -r requirements.txt
```

## Usage
To run the program with dataset provided,
```
python run_experiments.py
```
The output text file can also be specified,
```
python run_experiments.py -o OUTPUT_FILENAME
```
