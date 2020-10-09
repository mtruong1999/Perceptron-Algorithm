"""
Author: Michael Truong
"""
import os
import time
import numpy as np
import scipy.io as sio
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from perceptron import Perceptron
from tabulate import tabulate

DATA_DIR = os.path.join('data','data.mat') # input file
ACCURACY_THRESHOLD = 0.993 # Threshold for 5-fold CV on full data
ACCURACY_THRESHOLD_PCA = 0.92 # Threshold for 5-fold CV on PCA reduced data
OUTPUT_FILE = 'results.txt' # Output file name

# Globals for 4-class classifier
EPOCHS_1_2 = 1 # Epochs for training classifier on labels 1 and 2 
EPOCHS_3_4 = 1 # Epochs for training classifier on labels 3 and 4
EPOCHS_A_B = 2 # Epochs for parent node training on A/B combined labels

def get_five_fold_accuracies(X, classes, accuracy_thresh):
    """Runs 5-fold cross validation on input data X with given classes
    and trains each fold until a given accuracy threshold is met.

    Parameters
    ----------
    X: 2-D numpy array containing the input data
    classes: 1-D numpy array containing the corresponding class
        labels for X
    accuracy_thresh: float, continue training until this threshold is
        met.
    
    Returns
    ----------
    List of lists, each list element containing fold information to be
    reported. Includes the fold number with the number of iterations
    required to meet accuracy_thresh, the fold accuracy, and the training
    time.

    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    fold_count = 0
    fold_accuracies = np.zeros(5)
    fold_infos = []
    for train_index, test_index in skf.split(X, classes):
        fold_count += 1

        # Get fold data
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = classes[train_index], classes[test_index]
        
        print("Fold {}\n---------------------------------".format(fold_count))
        start_time = time.time()
        # Get weights from training set
        fold_perceptron = Perceptron(X_train, Y_train, 
                                    X_test, Y_test, accuracy_thresh)
        iter_required, fold_accuracy = fold_perceptron.train()

        elapsed_time = time.time() - start_time
        print("\tElapsed time: {} seconds".format(elapsed_time))
        fold_accuracies[fold_count - 1] = fold_accuracy
        fold_infos.append([fold_count, iter_required, fold_accuracy, elapsed_time])
    return fold_infos
	
def run_experiment(data):
    """Runs 5-fold cross validation on the normalized data set
    and the dataset reduced by the first two principle components.

    Parameters
    ----------
    data - numpy array containing the raw data directly imported by main

    Returns
    ----------
    The fold informations (described in `get_five_fold_accuracies`) for both
    the normalized data and PCA reduced data.

    """
    # Extract our classes
    Y = data[:, 0]
    binary_classes = np.where(Y <= 2, 0, 1)

    scaler = preprocessing.StandardScaler()
    data = scaler.fit_transform(data)
    # Turn first column into all 1's for bias (first column had the classes, which is irrelevant)
    data[:,0] = 1
    fold_infos = get_five_fold_accuracies(data, binary_classes, ACCURACY_THRESHOLD)

    pca = PCA(n_components=2)
    # Fit and transform data without the first column
    transformed_data = pca.fit_transform(np.delete(data, 0, 1))
    num_rows = transformed_data.shape[0]
    # Add column of 1's for perceptron algorithm
    transformed_data = np.hstack((
            np.ones((num_rows, 1)),
            transformed_data
    ))
    fold_infos_pca = get_five_fold_accuracies(transformed_data, binary_classes, ACCURACY_THRESHOLD_PCA)
    return fold_infos, fold_infos_pca


def run_decision_tree(data):
    """Trains 3 binary classifiers and combined them into a decision
    tree. The first node test whether a value belongs to class A
    (which represents labels 1 and 2) or class B (which represents
    labels 3 and 4). If A, we classify the data instance using a model
    trained to classify for the classes 1 and 2. If B, we classify the 
    data instance using a model trained to classify for 3 and 4.

    Parameters
    -----------
    data - numpy array containing the raw data directly imported by main

    Returns
    -----------
    The classification accuracy when running the model on the full dataset.
    """
    
    # Filter out data to get labels 1 and 2 only
    data_1_2 = data[data[:,0]<=2]

    # Extract and binarize labels 
    binary_labels_1_2 = np.where(data_1_2[:,0] == 1,0, 1)
    
    scaler_1_2 = preprocessing.StandardScaler()
    data_1_2 = scaler_1_2.fit_transform(data_1_2)
    data_1_2[:,0] = 1
    perceptron_1_2 = Perceptron(data_1_2, binary_labels_1_2, epochs=EPOCHS_1_2)
    weights_1_2 = perceptron_1_2.train()

    # delete data as we dont need it anymore
    del data_1_2
    del binary_labels_1_2

    # Filter out data to get labels 3 and 4 only
    data_3_4 = data[data[:,0] > 2]

    # Extract and binarize labels 
    binary_labels_3_4 = np.where(data_3_4[:,0] == 3, 0, 1)
    
    scaler_3_4 = preprocessing.StandardScaler()
    data_3_4 = scaler_3_4.fit_transform(data_3_4)
    data_3_4[:,0] = 1
    perceptron_3_4 = Perceptron(data_3_4, binary_labels_3_4, epochs=EPOCHS_3_4)
    weights_3_4 = perceptron_3_4.train()
    del data_3_4
    del binary_labels_3_4

    # Train AB class split set
    Y = data[:,0]
    binary_classes = np.where(Y <= 2, 0, 1)
    scaler = preprocessing.StandardScaler()
    data = scaler.fit_transform(data)
    data[:,0] = 1
    perceptron = Perceptron(data, binary_classes, epochs=EPOCHS_A_B)
    weights = perceptron.train()

    # Decision tree: run classifcation on full data set
    predicted_labels = np.empty(data.shape[0])
    for i, x in enumerate(data):
        # Classify between A and B
        firstNode = Perceptron.classify(x, weights)
        
        # If class is A...
        if firstNode == 0:
            secondNode = Perceptron.classify(x, weights_1_2)
            predicted_labels[i] = 2 if secondNode == 1 else 1
        else:
            secondNode = Perceptron.classify(x, weights_3_4)
            predicted_labels[i] = 4 if secondNode == 1 else 3
    accuracy = Perceptron.get_accuracy(Y, predicted_labels)
    return accuracy


if __name__ == '__main__':
    data = sio.loadmat(DATA_DIR)['data']
    
    fold_infos, fold_infos_pca = run_experiment(data)

    accuracy = run_decision_tree(data)

    with open(OUTPUT_FILE, 'w') as f:
        f.write('1. 5-fold cross validation results on FULL dataset\n')
        f.write(tabulate(fold_infos, headers=['Fold #', 
                                            'Iterations Required',
                                            'Fold Accuracy',
                                            'Running Time (sec)']))

        f.write('\n\n2. 5-fold cross validation results on PCA reduction dataset using 2 principle components\n')
        f.write(tabulate(fold_infos_pca, headers=['Fold #',
                                                'Iterations Required',
                                                'Fold Accuracy',
                                                'Running Time (sec)']))
        
        f.write('\n\n3. Accuracy of 4-class classifier on full dataset.\n')
        f.write('---------------------------------------------------\n')
        f.write('Accuracy: {}\n'.format(accuracy))

