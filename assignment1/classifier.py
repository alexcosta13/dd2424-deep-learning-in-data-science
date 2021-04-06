import numpy as np
from utils import softmax
from tqdm import tqdm


class Classifier:
    def __init__(self, input_size, output_size):
        np.random.seed(seed=400)
        self.input_size = input_size
        self.output_size = output_size
        self.W = np.random.normal(loc=0, scale=0.01, size=(output_size, input_size))
        self.b = np.random.normal(loc=0, scale=0.01, size=(output_size, 1))

    def predict(self, x):  # equivalent to function P = EvaluateClassifier(X, W, b)
        s = self.W @ x + self.b
        return softmax(s)

    def compute_cost(self, x, y, reg_lambda,
                     loss_function="cross_entropy"):  # equivalent to function J = ComputeCost(X, Y, W, b, lambda)
        batch_size = x.shape[1]
        p = self.predict(x)
        if loss_function == "cross_entropy":
            loss = - np.log(np.sum(y * p, axis=0))
        elif loss_function == "svm":
            s_y = np.sum(y * p, axis=0)
            s = np.maximum(np.zeros(p.shape), p - s_y + 1)
            s[y.astype(bool)] = 0
            loss = np.sum(s, axis=0)
        else:
            raise Exception("Loss function not implemented")
        reg = reg_lambda * np.sum(self.W ** 2)
        return np.sum(loss) / batch_size + reg

    def accuracy(self, x, y):  # equivalent to function acc = ComputeAccuracy(X, y, W, b)
        prediction = self.predict(x)
        a = np.argmax(prediction, axis=0) == np.argmax(y, axis=0)
        return np.count_nonzero(a) / np.size(a)

    def compute_gradients(self, x, y, reg_lambda, loss_function="cross_entropy"):  # function [grad W, grad b] = ComputeGradients(X, Y, P, W, lambda)
        batch_size = x.shape[1]
        p = self.predict(x)
        if loss_function == "cross_entropy":
            g = - (y - p)
            grad_W = 1 / batch_size * g @ x.T + 2 * reg_lambda * self.W
            grad_b = 1 / batch_size * g @ np.ones(shape=(batch_size, 1))

        elif loss_function == "svm":
            s_y = np.sum(y * p, axis=0)
            gradient = np.heaviside(p - s_y + 1, 0)
            gradient[y.astype(bool)] = 0
            misclassification_count = np.sum(gradient, axis=0)
            gradient[np.argmax(y, axis=0), np.arange(batch_size)] = - misclassification_count

            grad_W = gradient @ x.T / batch_size + reg_lambda * self.W
            grad_b = gradient @ np.ones(shape=(batch_size, 1)) / batch_size

        else:
            raise Exception("Loss function not implemented")

        return grad_W, grad_b

    def fit(self, x, y, x_val=None, y_val=None, n_batch=100, eta=0.001, n_epochs=40,
            reg_lambda=0, shuffle=False, decay_factor=1, loss_function="cross_entropy"):  # function [Wstar, bstar] = MiniBatchGD(X, Y, GDparams, W, b, lambda)

        self.W = np.random.normal(loc=0, scale=0.01, size=(self.output_size, self.input_size))
        self.b = np.random.normal(loc=0, scale=0.01, size=(self.output_size, 1))

        data_points = x.shape[1]
        train_errors, val_errors = [], []
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
                grad_W, grad_b = self.compute_gradients(x_batch, y_batch, reg_lambda, loss_function)
                self.W -= eta * grad_W
                self.b -= eta * grad_b
            train_errors.append(self.compute_cost(x, y, reg_lambda, loss_function))
            if x_val is not None:
                val_errors.append(self.compute_cost(x_val, y_val, reg_lambda, loss_function))
            eta *= decay_factor
        return train_errors, val_errors

    def compute_gradients_num(self, x, y, reg_lambda, h=1e-6):
        grad_W = np.zeros(shape=self.W.shape)
        grad_b = np.zeros(shape=self.b.shape)

        cost = self.compute_cost_num(x, y, reg_lambda)

        for i in range(self.output_size):
            b_try = np.copy(self.b)
            b_try[i] += h
            cost2 = self.compute_cost_num(x, y, reg_lambda, b=b_try)
            grad_b[i] = (cost2 - cost) / h

        for i in range(self.output_size):
            for j in range(self.input_size):
                W_try = np.copy(self.W)
                W_try[i, j] += h
                cost2 = self.compute_cost_num(x, y, reg_lambda, W=W_try)
                grad_W[i, j] = (cost2 - cost) / h

        return grad_W, grad_b

    def compute_gradients_num_slow(self, x, y, reg_lambda, h=1e-6):
        grad_W = np.zeros(shape=self.W.shape)
        grad_b = np.zeros(shape=self.b.shape)

        for i in range(self.output_size):
            b_try = np.copy(self.b)
            b_try[i] -= h
            cost1 = self.compute_cost_num(x, y, reg_lambda, b=b_try)

            b_try = np.copy(self.b)
            b_try[i] += h
            cost2 = self.compute_cost_num(x, y, reg_lambda, b=b_try)

            grad_b[i] = (cost2 - cost1) / (2 * h)

        for i in range(self.output_size):
            for j in range(self.input_size):
                W_try = np.copy(self.W)
                W_try[i, j] -= h
                cost1 = self.compute_cost_num(x, y, reg_lambda, W=W_try)

                W_try = np.copy(self.W)
                W_try[i, j] += h
                cost2 = self.compute_cost_num(x, y, reg_lambda, W=W_try)

                grad_W[i, j] = (cost2 - cost1) / (2 * h)

        return grad_W, grad_b

    def compute_cost_num(self, x, y, reg_lambda, W=None, b=None, loss_function="cross_entropy"):
        if W is None:
            W = self.W
        if b is None:
            b = self.b

        batch_size = x.shape[1]

        if loss_function == "cross_entropy":
            p = softmax(W @ x + b)
            # loss = np.sum(-np.log(y.T @ p), axis=1)
            # loss = - np.log(np.trace(y.T @ p))
            loss = -np.log(np.sum(y * p, axis=0))
        else:
            loss = 0

        reg = reg_lambda * np.sum(W ** 2)

        return np.sum(loss) / batch_size + reg
