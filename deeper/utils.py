"""
Inspired by Andrej Karpathy's Micrograd

A custom library for understanding the weeds of a deep learning module.
"""

import numpy as np

class Value():

    def __init__(self, data):
        self.data = data

    def __repr__(self):
        print(f'Value(data={self.data})')

    def __add__(self, other):
        return Value(self.data + other.data)
    

        