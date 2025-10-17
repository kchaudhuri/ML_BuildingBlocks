"""
Simple Linear Regrassion Algo
"""

import numpy as np

class LinearRegression:

    def __init__(self, lr, epochs):

        self.weights = None
        self.bias = None

        self.learning_rate = lr
        self.epochs = epochs

    def fit(self, X, y):
        """
        Training function
        """
        
        # # convert X and y to numpy array
        # X = np.asarray(X)
        # y = np.asarray(y)
        
        # Get number of samples
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_samples)
        self.bias = 0

        for epoch, _ in enumerate(range(self.epochs)):

            print(f'Epoch #{epoch}/{self.epochs}')

            # Predict output
            y_hat = np.dot(X, self.weights) + self.bias

            # Compute loss
            mse_loss = np.mean((y_hat - y) **2)

            # Compute gradients
            dw = (1/n_samples) * np.dot(X.T, mse_loss)
            db = (1/n_samples) * np.sum(mse_loss)

            # Update weights & biases
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        # # OLS Method
        # # convert X and y to numpy array
        # X = np.asarray(X)
        # y = np.asarray(y)
        
        # # Get number of samples
        # n_samples = X.shape[0]

        # # Create bias column (intercept term)
        # bias_column = np.ones((n_samples, 1))

        # # Concatenate bias and original X
        # X_b = np.concatenate([bias_column, X], axis=1)
        
        # # Normal Equation: (X^T * X)^(-1) * X^T * y
        # # Transpose of the feature matrix
        # X_transpose = X_b.T

        # # Compute X^T * X
        # XTX = X_transpose @ X_b

        # # Compute the pseudo-inverse of (X^T * X)
        # XTX_inv = np.linalg.pinv(XTX)

        # # Compute X^T * y
        # XTy = X_transpose @ y

        # # Compute theta (weights and bias)
        # theta_best = XTX_inv @ XTy

        # self.bias = theta_best[0]
        # self.weights = theta_best[1:]

    def predict(self, X):

        X = np.asarray(X)

        return X @ self.weights + self.bias

    def score(self, X, y):

        """Calculate R² score"""
        y_pred = self.predict(X)
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)

        return 1 - (ss_residual / ss_total)



if __name__ == "__main__":

    # Dummy data
    X = np.array([[1], [2], [4], [3], [5]])
    y = np.array([1, 3, 3, 2, 5])

    model = LinearRegression()
    model.fit(X, y)