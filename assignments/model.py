import numpy as np
import random
from tqdm import tqdm
from utils import horizontal_flip


class Model:
    def __init__(self):
        self.layers = []
        self.input = 0

    def initialize_weights(self):
        for layer in self.layers:
            layer.initialize_weights()

    def add(self, layer):
        # TODO check if layer is of type layer
        # TODO check if dimensions match - otherwise the training might fail
        self.layers.append(layer)

    def predict(self, x, train=False):
        partial_output = x
        for layer in self.layers:
            partial_output = layer.forward(partial_output, train)
        return self.layers[-1].last_output

    def compute_loss(self, x, y):
        batch_size = x.shape[1]
        p = self.predict(x)
        loss = - np.log(np.sum(y * p, axis=0))
        return np.sum(loss) / batch_size

    def accuracy(self, x, y):
        prediction = self.predict(x)
        a = np.argmax(prediction, axis=0) == np.argmax(y, axis=0)
        return np.count_nonzero(a) / np.size(a)

    def fit(self, x, y, x_val=None, y_val=None, n_batch=100, eta_min=1e-5, eta_max=1e-1, ns=500, k=1, reg_lambda=0.01,
            shuffle=True, random_jitter=False, random_flip=False):
        data_points = x.shape[1]
        train_loss, val_loss, train_accuracy, val_accuracy = [], [], [], []
        n_epochs = int(2 * ns * k / (data_points / n_batch))
        cycle = [eta_min + (t / ns) * (eta_max - eta_min) for t in range(ns)] \
                + [eta_max - ((t - ns) / ns) * (eta_max - eta_min) for t in range(ns, 2 * ns)]
        cycle *= k
        cycle = iter(cycle)
        for _ in tqdm(range(n_epochs)):
            if shuffle:
                p = np.random.permutation(x.shape[1])
                x = x[:, p]
                y = y[:, p]
            for i in range(int(data_points / n_batch) - 1):
                start = i * n_batch
                end = min((i + 1) * n_batch, data_points)
                x_batch = x[:, start:end]
                y_batch = y[:, start:end]

                if random_jitter:
                    noise = np.random.normal(loc=0.0, scale=0.15, size=x_batch.shape)
                    x_batch += noise

                if random_flip and random.choice([True, False]):
                    x_batch = horizontal_flip(x_batch)

                eta = next(cycle)

                self.compute_gradients(x_batch, y_batch, reg_lambda)
                self.update_weights(eta)

            train_loss.append(self.compute_loss(x, y))
            train_accuracy.append(self.accuracy(x, y))
            if x_val is not None:
                val_loss.append(self.compute_loss(x_val, y_val))
                val_accuracy.append(self.accuracy(x_val, y_val))
        return (train_loss, val_loss), (train_accuracy, val_accuracy)

    def update_weights(self, eta):
        for layer in self.layers:
            layer.update_weights(eta)

    def compute_gradients(self, x, y, reg_lambda):
        dense_gradients, batch_gradients = [], []
        p = self.predict(x, train=True)
        g = - (y - p)
        for layer in reversed(self.layers):
            g = layer.backward(g, reg_lambda)
            if layer.name == "dense":
                dense_gradients.append((layer.grad_kernel, layer. grad_bias))
            elif layer.name == "bn":
                batch_gradients.append((layer.grad_gamma, layer.grad_beta))
        return dense_gradients, batch_gradients

    def compute_gradients_num(self, x, y, reg_lambda, h=1e-5):
        def compute_loss():
            batch_size = x.shape[1]
            p = self.predict(x, True)
            loss = - np.log(np.sum(y * p, axis=0))
            return np.sum(loss) / batch_size

        W_grads, b_grads, gamma_grads, beta_grads = [], [], [], []

        for layer in self.layers:
            # Create grads for layer
            if layer.name == "dense":
                W_grad = np.zeros_like(layer.kernel)
                b_grad = np.zeros_like(layer.bias)
                #  Check gradient for W  and b.
                for i in range(len(layer.kernel[:10, :])):
                    for j in range(len(layer.kernel[i])):
                        layer.kernel[i][j] -= h
                        c1 = compute_loss()
                        layer.kernel[i][j] += 2 * h
                        c2 = compute_loss()
                        W_grad[i][j] = (c2 - c1) / (2 * h)
                        layer.kernel[i][j] -= h

                for i in range(len(layer.bias)):
                    layer.bias[i] -= h
                    c1 = compute_loss()
                    layer.bias[i] += 2 * h
                    c2 = compute_loss()
                    b_grad[i] = (c2 - c1) / (2 * h)
                    layer.bias[i] -= h

                W_grads.append(W_grad)
                b_grads.append(b_grad)

            if layer.name == "bn":
                gamma_grad = np.zeros_like(layer.gamma)
                beta_grad = np.zeros_like(layer.beta)
                #  Check gradient for gamma  and beta.
                for i in range(len(layer.gamma)):
                    layer.gamma[i] -= h
                    c1 = compute_loss()
                    layer.gamma[i] += 2 * h
                    c2 = compute_loss()
                    gamma_grad[i] = (c2 - c1) / (2 * h)
                    layer.gamma[i] -= h

                for i in range(len(layer.beta)):
                    layer.beta[i] -= h
                    c1 = compute_loss()
                    layer.beta[i] += 2 * h
                    c2 = compute_loss()
                    beta_grad[i] = (c2 - c1) / (2 * h)
                    layer.beta[i] -= h
                gamma_grads.append(gamma_grad)
                beta_grads.append(beta_grad)

        return W_grads, b_grads, gamma_grads, beta_grads
