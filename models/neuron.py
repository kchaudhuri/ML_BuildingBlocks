

import numpy as np

class Neuron:

    def __init__(self, examples):

        np.random.seed(42)
        self.weights = np.random.normal(0, 1, 3 + 1)
        self.examples = examples
        self.train()

    def forward():

        return

    def calculate_loss():

        return

    def train(self, learning_rate=0.01, batch_size=10, epochs=200):

        for epoch in range(epochs):
            for batch in range(int(self.examples.shape[0]//batch_size)):

                batch_idx = batch * batch_size
                #forward
                #loss
                #gradient
                #step backward
                
        pass

    def predict(self, features):

        pass
