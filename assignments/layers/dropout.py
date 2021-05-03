import numpy as np
from . import layer


class Dropout(layer.Layer):
    def __init__(self, dropout_rate):
        super().__init__("dropout")
        self.dropout_rate = dropout_rate
        self.last_dropout = None

    def forward(self, x, training=False):
        if training:
            self.last_input = x
            self.last_dropout = np.random.binomial(1, self.dropout_rate, size=x.shape) / self.dropout_rate
            self.last_output = x * self.last_dropout
        else:
            self.last_input = self.last_output = x
        return self.last_output

    def backward(self, g, reg_lambda):
        # TODO aixo sha de fer?
        return g * self.last_dropout
