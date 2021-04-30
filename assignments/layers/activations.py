import numpy as np
from . import layer


class ReLU(layer.Layer):
    def __init__(self):
        super().__init__("relu")

    def forward(self, x, training=False):
        self.last_input = x
        self.last_output = np.maximum(0, x)
        return self.last_output


class Softmax(layer.Layer):
    def __init__(self):
        super().__init__("softmax")

    def forward(self, x, training=False):
        self.last_input = x
        self.last_output = np.exp(x) / np.sum(np.exp(x), axis=0)
        return self.last_output
