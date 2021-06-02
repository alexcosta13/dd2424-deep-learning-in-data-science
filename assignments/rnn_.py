import sys

import numpy as np
from tqdm import tqdm
from utils import softmax, sample_from_probability, index_2_one_hot, indices_2_chars, chars_2_indices

epsilon = sys.float_info.epsilon


class RNN:
    def __init__(self, input_size, output_size, hidden_size=100, **kwargs):
        np.random.seed(400)
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.h, self.p = None, None

        self.weights = {}
        self.gradients = {}
        self.m_theta = {}

        self.char_2_indices = kwargs['char_2_indices']
        self.indices_2_char = kwargs['indices_2_char']
        self.kwargs = kwargs
        self.initialize_weights()

    def initialize_weights(self):
        np.random.seed(42)
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

    def generate(self, x0=None, h0=None, seq_length=200):
        if x0 is None:
            x0 = index_2_one_hot(0, self.output_size)
        if h0 is None:
            h0 = np.zeros((self.hidden_size, 1))
        output = []
        h = h0
        xt = x0
        for _ in range(seq_length):
            at = self.weights['W'] @ h + self.weights['U'] @ xt + self.weights['b']
            h = np.tanh(at)
            ot = self.weights['V'] @ h + self.weights['c']
            pt = softmax(ot).squeeze()
            sample = sample_from_probability(self.output_size, pt)
            output.append(sample[0])
            xt = index_2_one_hot(sample, self.output_size)
        return indices_2_chars(output, self.indices_2_char)

    def forward(self, x, y, h0):
        seq_length = x.shape[1]
        self.p, self.h = [None] * seq_length, [None] * (seq_length + 1)

        self.h[0] = h0
        loss = 0

        for t in range(seq_length):
            a = self.weights['W'] @ self.h[t] + self.weights['U'] @ x[:, [t]] + self.weights['b']
            self.h[t + 1] = np.tanh(a)
            o = self.weights['V'] @ self.h[t + 1] + self.weights['c']
            self.p[t] = softmax(o)  # .squeeze()
            loss -= np.log(y[:, [t]].T @ self.p[t])[0, 0]

        return loss

    def backward(self, x, y):
        seq_length = x.shape[1]

        self.gradients = {}
        for param in self.weights:
            self.gradients[param] = np.zeros_like(self.weights[param])

        dLda = np.zeros((1, self.hidden_size))
        for t in reversed(range(seq_length)):
            g = -(y[:, [t]] - self.p[t]).T
            self.gradients['V'] += g.T @ self.h[t + 1].T
            self.gradients['c'] += g.T
            dLdh = g @ self.weights['V'] + dLda @ self.weights['W']
            dLda = dLdh @ np.diag(1 - self.h[t + 1].squeeze() ** 2)
            self.gradients['W'] += dLda.T @ self.h[t].T
            self.gradients['U'] += dLda.T @ x[:, [t]].T
            self.gradients['b'] += dLda.T

        for param in self.weights:
            self.gradients[param] = np.clip(self.gradients[param], -5, 5)
            
        return self.gradients

    def update_weights(self, eta):
        for param in self.weights:
            self.m_theta[param] += self.gradients[param] ** 2
            self.weights[param] -= eta / np.sqrt(self.m_theta[param] + epsilon) * self.gradients[param]

    def fit(self, text, seq_length=25, eta=0.1, iterations=300000, verbose=None):
        h_prev = np.zeros(shape=(self.hidden_size, 1))
        e = 0
        smooth_losses = []

        for iteration in range(iterations):
            if e + seq_length > len(text):
                h_prev = np.zeros(shape=(self.hidden_size, 1))
                e = 0

            x_chars = index_2_one_hot(np.array(chars_2_indices(text[e:e + seq_length],
                                                               self.char_2_indices)), self.output_size)
            y_chars = index_2_one_hot(np.array(chars_2_indices(text[e + 1:e + seq_length + 1],
                                                               self.char_2_indices)), self.output_size)
            loss = self.forward(x_chars, y_chars, h_prev)
            self.backward(x_chars, y_chars)
            self.update_weights(eta)

            smooth_loss = 0.999 * smooth_loss + 0.001 * loss if "smooth_loss" in locals() else loss

            smooth_losses.append(smooth_loss)

            if verbose and not iteration % verbose:
                print(f"Update step {iteration} with loss: {smooth_loss}")
                print(f"Generated text:\n{self.generate(x0=x_chars[:, [0]], h0=h_prev)}\n\n\n")

            h_prev = self.h[-1]
            e += seq_length

        return smooth_losses

    def compute_gradients_num(self, x, y, h=1e-4):
        num_gradients = {}
        for param in self.weights:
            num_gradients[param] = np.zeros_like(self.weights[param])

        for param in ['W', 'U', 'V']:
            for i in range(len(self.weights[param])):
                for j in range(len(self.weights[param][i])):
                    self.weights[param][i][j] -= h
                    c1 = self.forward(x, y, np.zeros(shape=(10, 1)))
                    self.weights[param][i][j] += 2 * h
                    c2 = self.forward(x, y, np.zeros(shape=(10, 1)))
                    num_gradients[param][i][j] = (c2 - c1) / (2 * h)
                    self.weights[param][i][j] -= h

        for param in ['b', 'c']:
            for i in range(len(self.weights[param])):
                self.weights[param][i] -= h
                c1 = self.forward(x, y, np.zeros(shape=(10, 1)))
                self.weights[param][i] += 2 * h
                c2 = self.forward(x, y, np.zeros(shape=(10, 1)))
                num_gradients[param][i] = (c2 - c1) / (2 * h)
                self.weights[param][i] -= h

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