import sys

import numpy as np
from tqdm import tqdm
from utils import softmax, sample_from_probability, index_2_one_hot, indices_2_chars

epsilon = sys.float_info.epsilon


class RNN:
    def __init__(self, input_size, output_size, hidden_size=100, **kwargs):
        np.random.seed(400)
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.h = np.zeros((self.hidden_size, 1))
        self.hidden_states = None
        self.last_predictions = None

        self.weights = {}
        self.gradients = {}
        self.m_theta = {}

        self.char_2_indices = kwargs['char_2_indices']
        self.indices_2_char = kwargs['indices_2_char']
        self.kwargs = kwargs
        self.initialize_weights()

    def initialize_weights(self):
        np.random.seed(400)
        sigma = self.kwargs["sigma"] if "sigma" in self.kwargs else 0.01
        W = np.random.normal(loc=0, scale=sigma, size=(self.hidden_size, self.hidden_size))
        U = np.random.normal(loc=0, scale=sigma, size=(self.hidden_size, self.input_size))
        b = np.zeros(shape=(self.hidden_size, 1))
        V = np.random.normal(loc=0, scale=sigma, size=(self.output_size, self.hidden_size))
        c = np.zeros(shape=(self.output_size, 1))

        self.weights = {
            'W': W,
            'U': U,
            'b': b,
            'V': V,
            'c': c,
        }

        for param in self.weights:
            self.gradients[param] = np.zeros_like(self.weights[param])
            self.m_theta[param] = np.zeros_like(self.weights[param])

    def generate(self, x0=None, h0=None, seq_length=25):
        if x0 is None:
            x0 = index_2_one_hot(0, self.output_size)
        if h0 is None:
            h0 = np.zeros((self.hidden_size, 1))
        output = []
        self.h = h0
        xt = x0
        for _ in range(seq_length):
            at = self.weights['W'] @ self.h + self.weights['U'] @ xt + self.weights['b']
            self.h = np.tanh(at)
            ot = self.weights['V'] @ self.h + self.weights['c']
            pt = softmax(ot).squeeze()
            sample = sample_from_probability(self.output_size, pt)
            output.append(sample[0])
            xt = index_2_one_hot(sample, self.output_size)
        return indices_2_chars(output, self.indices_2_char)

    def forward(self, x, h0=None):
        seq_length = x.shape[1]
        self.h = h0 if h0 is not None else self.h
        self.hidden_states = [h0]
        self.last_predictions = []

        for t in range(seq_length):
            at = self.weights['W'] @ self.h + self.weights['U'] @ x[:, [t]] + self.weights['b']
            self.h = np.tanh(at)
            ot = self.weights['V'] @ self.h + self.weights['c']
            pt = softmax(ot)  # .squeeze()

            self.hidden_states.append(self.h)
            self.last_predictions.append(pt)

    def compute_gradients(self, x, y, reg_lambda=0.0):
        seq_length = x.shape[1]

        dLdo = []

        for t in range(seq_length):
            dLdo.append(- (y[:, [t]] - self.last_predictions[t]).T)
            g = dLdo[t]
            self.gradients['V'] += g.T @ self.hidden_states[t + 1].T
            self.gradients['c'] += g.T

        dLda = np.zeros((1, self.hidden_size))

        for t in reversed(range(seq_length - 1)):
            dLdh = dLdo[t] @ self.weights['V'] + dLda @ self.weights['W']
            dLda = dLdh @ np.diag(1 - (self.hidden_states[t + 1].squeeze() ** 2))
            g = dLda
            self.gradients['W'] += g.T @ self.hidden_states[t + 1].T
            self.gradients['U'] += g.T @ x[:, [t]].T
            self.gradients['b'] += g.T

        return self.gradients

    def update_weights(self, eta):
        for param in self.weights:
            self.m_theta[param] += self.gradients[param] ** 2
            self.weights[param] -= eta / np.sqrt(self.m_theta[param] + epsilon) * self.gradients[param]

    def compute_loss(self, x, y):
        loss = 0
        for p in self.last_predictions:
            loss += - np.sum(np.log(np.sum(y * p, axis=0)))
        return loss

    def compute_gradients_num(self, x, y, reg_lambda=0.0, h=1e-4):
        def compute_loss():
            loss = 0
            hidden_state = np.zeros(shape=(self.hidden_size, 1))
            for t in range(x.shape[1]):
                a = self.weights['W'] @ hidden_state + self.weights['U'] @ x[:, [t]] + self.weights['b']
                hidden_state = np.tanh(a)
                ot = self.weights['V'] @ hidden_state + self.weights['c']
                p = softmax(ot)
                loss += - np.sum(np.log(np.sum(y * p, axis=0)))
            return loss

        num_gradients = {}
        for param in self.weights:
            num_gradients[param] = np.zeros_like(self.weights[param])

        """# Check gradient for W
        for i in range(len(self.weights['W'])):
            for j in range(len(self.weights['W'][i])):
                self.weights['W'][i][j] -= h
                c1 = compute_loss()
                self.weights['W'][i][j] += 2 * h
                c2 = compute_loss()
                num_gradients['W'][i][j] = (c2 - c1) / (2 * h)
                self.weights['W'][i][j] -= h

        # Check gradient for U
        for i in range(len(self.weights['U'])):
            for j in range(len(self.weights['U'][i])):
                self.weights['U'][i][j] -= h
                c1 = compute_loss()
                self.weights['U'][i][j] += 2 * h
                c2 = compute_loss()
                num_gradients['U'][i][j] = (c2 - c1) / (2 * h)
                self.weights['U'][i][j] -= h

        # Check gradient for b
        for i in range(len(self.weights['b'])):
            self.weights['b'][i] -= h
            c1 = compute_loss()
            self.weights['b'][i] += 2 * h
            c2 = compute_loss()
            num_gradients['b'][i] = (c2 - c1) / (2 * h)
            self.weights['b'][i] -= h"""

        # Check gradient for V
        for i in range(len(self.weights['V'])):
            for j in range(len(self.weights['V'][i])):
                self.weights['V'][i][j] -= h
                c1 = compute_loss()
                self.weights['V'][i][j] += 2 * h
                c2 = compute_loss()
                num_gradients['V'][i][j] = (c2 - c1) / (2 * h)
                self.weights['V'][i][j] -= h

        """# Check gradient for c
        for i in range(len(self.weights['c'])):
            self.weights['c'][i] -= h
            c1 = compute_loss()
            self.weights['c'][i] += 2 * h
            c2 = compute_loss()
            num_gradients['c'][i] = (c2 - c1) / (2 * h)
            self.weights['c'][i] -= h"""

        return num_gradients


"""
        g = -(y.T - self.last_predictions.T).T

        self.gradients['V'] = g @ self.hidden_states[:, 1:].T
        self.gradients['c'] = np.sum(g, axis=-1, keepdims=True)

        dLdh = np.zeros((seq_length, self.hidden_size))
        dLda = np.zeros((self.hidden_size, seq_length))

        dLdh[-1] = g.T[-1] @ self.weights['V']
        # multiply?
        dLda[:, -1] = dLdh[-1].T * (1 - (self.hidden_states[:, -1] + self.hidden_states[:, -1]))

        for t in reversed(range(seq_length - 1)):
            dLdh[t] = g.T[t] @ self.weights['V'] + dLda[:, t+1] @ self.weights['W']
            # multiply?
            dLda[:, t] = dLdh[t].T * (1 - (self.hidden_states[:, t] + self.hidden_states[:, t]))

        self.gradients['W'] = dLda @ self.hidden_states[:, :-1].T
        self.gradients['U'] = dLda @ x.T
        self.gradients['b'] = np.sum(dLda, axis=-1, keepdims=True)"""