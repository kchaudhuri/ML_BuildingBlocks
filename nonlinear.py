"""
Collection of non linear functions
"""

import numpy as np


class NonLinear():

    def softmax():

    def sigmoid(x):

        """
        Sigmoid (Logistic) function

        Args:
            x (float or np.ndarray): Input array.

        Returns:
            np.ndarray: Output after applying Sigmoid.
        """

        return 1/(1 + np.exp(-x))

    def relu(x):

        """
        Rectified Linear Unit

        Args:
            x (float or np.ndarray): Input array.

        Returns:
            np.ndarray: Output after applying ReLU.
        """
        
        return np.maximum(0, x)

    def leaky_relu(x, alpha=0.01):

        """
        Leaky Rectified Linear Unit

        Args:
            x (float or np.ndarray): Input array.

        Returns:
             np.ndarray: Output after applying Leaky ReLU.
        """

        return np.where(x > 0, x, x * alpha) # If positive value then as is else value multiplied by alpha factor