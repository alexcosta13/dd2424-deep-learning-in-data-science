import numpy as np
from . import layer


class Dense(layer.Layer):
    def __init__(self, input_size, output_size):
        super().__init__("dense")
        np.random.seed(400)
        self.input_size = input_size
        self.output_size = output_size
        self.bias = np.zeros(shape=(self.output_size, 1))
        self.kernel = np.random.normal(loc=0, scale=2/np.sqrt(self.input_size),
                                       size=(self.output_size, self.input_size))

        self.grad_kernel = None
        self.grad_bias = None

    def initialize_weights(self):
        np.random.seed(400)
        self.bias = np.zeros(shape=(self.output_size, 1))
        self.kernel = np.random.normal(loc=0, scale=2/np.sqrt(self.input_size),
                                       size=(self.output_size, self.input_size))

    def forward(self, x, training=False):
        self.last_input = x
        self.last_output = self.kernel @ x + self.bias
        return self.last_output

    def backward(self, g, reg_lambda):
        batch_size = g.shape[1]
        self.grad_kernel = 1 / batch_size * g @ self.last_input.T + 2 * reg_lambda * self.kernel
        self.grad_bias = 1 / batch_size * g @ np.ones(shape=(batch_size, 1))

        g = self.kernel.T @ g
        g[self.last_input <= 0] = 0

        return g

    def update_weights(self, eta):
        self.kernel -= eta * self.grad_kernel
        self.bias -= eta * self.grad_bias
