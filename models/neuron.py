"""
Simple Neuron implementation
"""

import numpy as np

class Neuron:

    def __init__(self, examples):

        np.random.seed(42)
        self.weights = np.random.normal(0, 1, 3 + 1)
        self.examples = examples
        self.train()

    def _sigmoid(z):

        return 1.0 / (1.0 + np.exp(-z))

    def forward(self, X):

        z = X @ self.weights[:3] + self.weights[3]
        y_hat = self._sigmoid(z)

        return y_hat

    def calculate_loss():

        return

    def train(self, learning_rate=0.01, batch_size=10, epochs=200):

        for epoch in range(epochs):
            for batch in range(int(self.examples.shape[0]//batch_size)):

                batch_idx = batch * batch_size

                #forward
                output = forward(self.examples['features'])

                #loss
                loss = calculate_loss(gt, output)

                #gradient


                #step backward
                
        pass

    def predict(self, features):

        pass
