"""
Preprocessing class
"""

import pandas as pd

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
        
