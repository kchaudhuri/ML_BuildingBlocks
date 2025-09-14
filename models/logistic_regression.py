"""
Simple Logistic Regression Algo
"""

import numpy as np

class LogisticRegression:
    def __init__(self, 
                 learning_rate=0.01, 
                 epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))


    def fit(self, X, y):
        """
        Training function
        """

        n_samples, n_features = X.shape

        # Initiate zeros for weights and biases
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):

            # Calculate y
            output = np.dot(x, self.weights) + self.bias
            y_predict = self._sigmoid(output)

            # Compute Gradient
            dw = (1 / n_samples) * np.dot(X.T, (y_predict - y))
            db = (1 / n_samples) * np.sum(y_predict - y)

            # Step towards gradient
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    
    def predict_proba(self, X):
        """
        Calculate probability output to zero or one
        """
        output = np.dot(X, self.weights) + self.bias
        return self._sigmoid(output)
        



    def predict(self, X):
        """
        Given set of input predict the outputs
        """

        y_probs = self.predict_proba(X)
        return (y_probs >= 0.5).astype(int)
