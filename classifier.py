"""
Author: Michael Truong
"""
import os
import numpy as np
import scipy.io as sio
from scipy import stats
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold

DATA_DIR = os.path.join('data','data.mat')
NUM_EPOCHS = 100
np.random.seed(0)

def train(X, Y, num_iterations):
    """Runs the perceptron algorithm to train a set of weights. 
    Runs num_iterations through the data set.

    Parameters
    ----------
    X - numpy array containing the feature vectors
    Y - numpy array containing the correct classification of
        each feature vector in X.
    num_iterations - int indicating the number of iterations 
        through the data set X perceptron is to run for.

    Returns
    -------
    A numpy array `w` containing the trained weights (includes
    the bias term as the first element).
    """
    # Initialize wieghts with zeros
    w = np.zeros(X.shape[1])
    for _ in range(num_iterations):
        for idx,element in enumerate(X):
            if Y[idx] == 1 and w.dot(element) < 0:
                w = w + element
            elif Y[idx] == 0 and w.dot(element) >= 0:
                w = w - element
    return w

def classify(x, weights):
    """Uses a set of weights to classify a single data instance.

    Parameters
    ----------
    x - numpy array containing the attributes of the data element to
        be classified
    weights - numpy array containing the set of trained weights for
        classification

    Returns
    -------
    An integer representing the predicted class label.
    """
    return 1 if weights.dot(x) >= 0 else 0

def get_classification_results(X, W):
    """Classifies an input data set given a set of weights.

    Parameters
    ----------
    X - 2D numpy array containing the data set to be classified where
        each row is a data instance and each column represents an attribute
    W - numpy array containing the trained weights

    Returns
    -------
    A 1D numpy array containing the classified labels for X
    """
    return np.array([classify(x,W) for x in X])

def get_accuracy(actual_labels, predicted_labels):
    return np.count_nonzero(actual_labels == predicted_labels)/actual_labels.size

def run_experiments(X, classes):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    fold_count = 0
    fold_accuracies = np.zeros(5)
    for train_index, test_index in skf.split(X, classes):
        fold_count += 1

        # Get fold data
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = classes[train_index], classes[test_index]
        
        # Get weights from training set
        fold_weights = train(X_train, Y_train)

        # Get results from test set
        fold_results = get_classification_results(X_test, fold_weights)
        fold_accuracy = get_accuracy(Y_test, fold_results)
        fold_accuracies[fold_count - 1] = fold_accuracy
        print("Fold {} accuracy: {}".format(fold_count, fold_accuracy))
    
    return fold_accuracies


if __name__ == '__main__':
    data = sio.loadmat(DATA_DIR)['data']
    
    # Extract our classes
    Y = data[:,0]
    binary_classes = np.where(Y <= 2, 0, 1)
    # Remove classes column from input data
    # data = np.delete(data, 0, 1)
    # Normalize data
    # data = stats.zscore(data)
    scaler = preprocessing.StandardScaler()
    data = scaler.fit_transform(data)
    # Turn first column into all 1's for bias (first column had the classes, which is irrelevant)
    data[:,0] = 1

    # standardize the validation set as follows:
    # standardized_validation = scaler.transform(validation)
    # standardized_validation[:,0] = 1
    # classify as follows
    # weights = train(data, classes)
    # np.array([resultT]).dot(standardized_validation)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)
    foldCount = 0
    for train, test in skf.split(data, Y):
        print("\nFold {}".format(foldCount + 1))
        trainX, testX = data[train], data[test]
        trainY, testY = Y[train], Y[test]
        
        trainedWeights = train(trainX, trainY, NUM_EPOCHS)
