import sys

import numpy as np
from . import layer

epsilon = sys.float_info.epsilon
alpha = 0.9


class BatchNormalization(layer.Layer):
    def __init__(self, nodes):
        super().__init__("bn")
        self.nodes = nodes
        self.s_hat = None

        self.last_mu = None
        self.last_var = None

        self.mu_av = np.zeros((self.nodes, 1))
        self.var_av = np.ones((self.nodes, 1))

        self.gamma = np.ones((nodes, 1))
        self.beta = np.zeros((nodes, 1))

        self.grad_gamma = None
        self.grad_beta = None

    def initialize_weights(self):
        self.s_hat = None

        self.last_mu = None
        self.last_var = None

        self.mu_av = np.zeros((self.nodes, 1))
        self.var_av = np.ones((self.nodes, 1))

        self.gamma = np.ones((self.nodes, 1))
        self.beta = np.zeros((self.nodes, 1))

        self.grad_gamma = None
        self.grad_beta = None

    def forward(self, x, training=False):
        self.last_input = x
        self.batch_normalize(x) if training else self.test_normalize(x)
        self.last_output = self.gamma * self.s_hat + self.beta
        if training:
            self.mu_av = alpha * self.mu_av + (1 - alpha) * self.last_mu
            self.var_av = alpha * self.var_av + (1 - alpha) * self.last_var
        return self.last_output

    def backward(self, g, reg_lambda):
        batch_size = g.shape[1]
        self.grad_gamma = 1 / batch_size * (g * self.s_hat) @ np.ones(shape=(batch_size, 1))
        self.grad_beta = 1 / batch_size * g @ np.ones(shape=(batch_size, 1))

        g = g * (self.gamma @ np.ones(shape=(batch_size, 1)).T)

        return self.batch_normalize_back(g)

    def batch_normalize(self, x):
        batch_size = x.shape[1]
        self.last_mu = 1 / batch_size * np.sum(x, axis=1)[:, np.newaxis]  # double check axis
        self.last_var = 1 / batch_size * np.sum((x - self.last_mu) ** 2, axis=1)

        self.s_hat = np.linalg.inv(np.diag(np.sqrt(self.last_var + epsilon))) @ (x - self.last_mu)

        self.last_var = self.last_var[:, np.newaxis]

    def test_normalize(self, x):
        self.s_hat = np.linalg.inv(np.diag(np.sqrt(self.var_av + epsilon)[:, 0])) @ (x - self.mu_av)

    def batch_normalize_back(self, g):
        n = g.shape[1]

        sigma_1 = ((self.last_var + epsilon) ** -0.5)
        sigma_2 = ((self.last_var + epsilon) ** -1.5)

        g_1 = g * (sigma_1 @ np.ones(shape=(n, 1)).T)
        g_2 = g * (sigma_2 @ np.ones(shape=(n, 1)).T)

        d = self.last_input - self.last_mu @ np.ones(shape=(n, 1)).T

        c = (g_2 * d) @ np.ones(shape=(n, 1))

        g = g_1 - 1 / n * (g_1 @ np.ones(shape=(n, 1))) @ np.ones(shape=(n, 1)).T \
            - 1 / n * d * (c @ np.ones(shape=(n, 1)).T)

        return g

    def update_weights(self, eta):
        self.gamma -= eta * self.grad_gamma
        self.beta -= eta * self.grad_beta
