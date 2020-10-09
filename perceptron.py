"""
Author: Michael Truong
"""
import numpy as np

class Perceptron:

    def __init__(self, X, Y, X_test=None, Y_test=None, accuracy_threshold=None, epochs=None):
        """Initialized Perceptron class"""
        self.X = X
        self.Y = Y

        if (X_test is None or Y_test is None or accuracy_threshold is None):
            if (epochs == None):
                raise Exception("Must specify number of epochs if not performing validation")
            self.epochs = epochs
            self.validate=False
        else:
            self.validate=True
            self.X_test = X_test
            self.Y_test = Y_test
            self.acc_thresh = accuracy_threshold

        self.iterations_required = 0
        
        

    def train(self):
        """Runs the perceptron algorithm. Stops after a specified
        threshold accuracy is met when using validation. Otherwise,
        trains until a specified number of epochs have passed.
        """

        # Initialize wieghts with zeros
        w = np.zeros(self.X.shape[1])
        accuracy_met = False
        num_epochs_elapsed = 0

        if(self.validate):
            while not accuracy_met:
                num_epochs_elapsed += 1
                self.epoch_fit(w)
                self.predicted_labels = self.get_validation_labels(w)
                epoch_accuracy = Perceptron.get_accuracy(self.Y_test, self.predicted_labels)

                print("\tEpoch: {}\tAccuracy: {}".format(num_epochs_elapsed,
                                                        epoch_accuracy))

                if epoch_accuracy >= self.acc_thresh:
                    accuracy_met = True
            return num_epochs_elapsed, epoch_accuracy
        else:
            for _ in range(self.epochs):
                self.epoch_fit(w)
            return w

    def epoch_fit(self, w):
        """Iteratively adjusts a set of weights for the perceptron algorithm"""
        for idx,element in enumerate(self.X):
            if self.Y[idx] == 1 and w.dot(element) < 0:
                w[:] += element
            elif self.Y[idx] == 0 and w.dot(element) >= 0:
                w[:] -= element

    @staticmethod
    def classify(x, weights):
        """Returns class label according to the dot product of weights and 
        our data instance
        """
        return 1 if weights.dot(x) >= 0 else 0

    def get_validation_labels(self, W):
        """Classifies all test data elements using trained set of weights."""
        return np.array([Perceptron.classify(x,W) for x in self.X_test])

    @staticmethod
    def get_accuracy(actual_labels, predicted_labels):
        """Returns the accuracy given the actual and predicted labels."""
        return np.count_nonzero(actual_labels == predicted_labels)/actual_labels.size