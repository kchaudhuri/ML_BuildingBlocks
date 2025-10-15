"""
Preprocessing class
"""

import pandas as pd
import numpy as np

class Dataset():
    def __init__(self, data):
        self.data = data


    def split(self, test_pct=0.2, random_state=42):
        
        """
        Splits the loaded dataset into train test split

        Args:
            test_pct (float): Percent of test dataset
            random_state (int): Random seed to be set

        Returns:
            pd.DataFrame:
            pd.DataFrame:
        """

        test_size = int(len(self.data) * test_pct)
        self.test_data = self.data.sample(n = test_size, random_state = random_state)
        
        self.train_data = self.data.drop(self.test_data.index)
        
        return self.train_data, self.test_data
    
    def normalize(x):

        """
        Normalization of the input array (Min Max Scaling)
        """

        x_min, x_max = np.min(x), np.max(x)

        x_scaled = (x - x_min) / (x_max - x_min)

        return x_scaled
    
    def standardize(x):

        """
        Standardization of the input array (Z-score Normalization).
        """

        mean = np.mean(x)
        std = np.std(x)

        # Handle zero standard deviation (avoid division by zero)
        if std == 0:
            return np.zeros_like(x)

        return (x - mean) / std

        
