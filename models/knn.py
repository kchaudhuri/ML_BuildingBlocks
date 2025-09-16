"""
Simple K Nearest Neighbors
"""

import numpy as np
from collections import Counter

class KNN:

    def __init__(self, k=3):

        self.k = k
    
    def fit(self, X, y):
        """
        Training function (not really)
        """

        self.X_train = X
        self.y_train = y
    
    def _euclidean_distance(self, x1, x2):
        """
        Distance calculation
        """

        return np.sqrt(np.sum((x1 - x2)**2))
    
    def _predict_single_point(self, x):

        # Compute distances to all training points
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        
        # Get indices of k nearest samples
        k_indices = np.argsort(distances)[:self.k]
        
        # Extract the labels of the k nearest samples
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # Majority vote
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
    
    def predict(self, X):

        predictions = [self._predict_single_point(x) for x in X]

        return np.array(predictions)
    
    
