"""
Preprocessing class
"""

import pandas as pd

class Dataset():
    def __init__(self, data):
        self.data = data


    def split(self, test_pct=0.2, random_state=42):

        train_size = int(len(self.data))
        self.train_data = self.data.sample(n=train_size, 
                                           random_state=random_state)
        
        return self.train_data
        
