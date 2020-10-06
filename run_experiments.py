"""
Author: Michael Truong
"""
import os
import time
import numpy as np
import scipy.io as sio
from scipy import stats
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from perceptron import Perceptron

DATA_DIR = os.path.join('data','data.mat')
NUM_EPOCHS = 100
ACCURACY_THRESHOLD = 0.993

def get_five_fold_accuracies(X, classes, accuracy_thresh):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    fold_count = 0
    fold_accuracies = np.zeros(5)
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
        fold_weights, iter_required, fold_accuracy = fold_perceptron.train()

        print("\tElapsed time: {} seconds".format(time.time() - start_time))
        #fold_accuracies[fold_count - 1] = fold_accuracy
    
    return fold_accuracies
	
def run_experiment(data):
    # Extract our classes
    Y = data[:, 0]
    binary_classes = np.where(Y <= 2, 0, 1)

    scaler = preprocessing.StandardScaler()
    data = scaler.fit_transform(data)
    # Turn first column into all 1's for bias (first column had the classes, which is irrelevant)
    data[:,0] = 1
    get_five_fold_accuracies(data, binary_classes, ACCURACY_THRESHOLD)

    pca = PCA(n_components=2)
    # Fit and transform data without the first column
    transformed_data = pca.fit_transform(np.delete(data, 0, 1))
    num_rows = transformed_data.shape[0]
    # Add column of 1's for perceptron algorithm
    transformed_data = np.hstack((
            np.ones((num_rows, 1)),
            transformed_data
    ))
    get_five_fold_accuracies(transformed_data, binary_classes, 0.92)


def run_decision_tree(data):
	# Train on data
	# Filter out data by classes
	# train on the smaller data sets
    
    # Filter out data to get labels 1 and 2 only
    data_1_2 = data[data[:,0]<=2]
    # Extract and binarize labels 
    binary_labels_1_2 = np.where(data_1_2[:,0] == 1,0, 1)
    
    scaler_1_2 = preprocessing.StandardScaler()
    data_1_2 = scaler_1_2.fit_transform(data_1_2)
    data_1_2[:,0] = 1
    perceptron_1_2 = Perceptron(data_1_2, binary_labels_1_2, epochs=2)

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
    perceptron_3_4 = Perceptron(data_3_4, binary_labels_3_4, epochs=2)

    weights_3_4 = perceptron_3_4.train()
    del data_3_4
    del binary_labels_3_4

    # Train AB class split set
    Y = data[:,0]
    binary_classes = np.where(Y <= 2, 0, 1)
    scaler = preprocessing.StandardScaler()
    data = scaler.fit_transform(data)
    data[:,0] = 1
    perceptron = Perceptron(data, binary_classes, epochs=2)
    weights = perceptron.train()

    # Run classifcation on full data set
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
    
    run_experiment(data)

    run_decision_tree(data)

