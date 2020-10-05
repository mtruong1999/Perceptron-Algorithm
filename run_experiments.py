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

def run_experiments(X, classes, accuracy_thresh):
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

if __name__ == '__main__':
    data = sio.loadmat(DATA_DIR)['data']
    
    # Extract our classes
    Y = data[:,0]
    binary_classes = np.where(Y <= 2, 0, 1)

    scaler = preprocessing.StandardScaler()
    data = scaler.fit_transform(data)
    # Turn first column into all 1's for bias (first column had the classes, which is irrelevant)
    data[:,0] = 1

    run_experiments(data, binary_classes, ACCURACY_THRESHOLD)

    pca = PCA(n_components=2)
    # Fit and transform data without the first column
    transformed_data = pca.fit_transform(np.delete(data, 0, 1))
    num_rows = transformed_data.shape[0]
    # Add column of 1's for perceptron algorithm
    transformed_data = np.hstack((
            np.ones((num_rows, 1)),
            transformed_data
    ))
    run_experiments(transformed_data, binary_classes, 0.92)

