import numpy as np
from utils import softmax
from tqdm import tqdm


class Classifier:
    class Layer:
        def __init__(self, input_size, output_size, activation):
            np.random.seed(400)
            self.input_size = input_size
            self.output_size = output_size
            self.activation = activation
            self.W = np.random.normal(loc=0, scale=1 / np.sqrt(input_size), size=(output_size, input_size))
            self.b = np.zeros(shape=(output_size, 1))

        def reset(self):
            np.random.seed(400)
            self.W = np.random.normal(loc=0, scale=1 / np.sqrt(self.input_size),
                                      size=(self.output_size, self.input_size))
            self.b = np.zeros(shape=(self.output_size, 1))

        def activation_function(self, x):
            if self.activation == "relu":
                return np.maximum(0, x)
            elif self.activation == "softmax":
                return softmax(x)
            else:
                raise Exception("Activation function not implemented")

        def forward_pass(self, x):
            s = self.W @ x + self.b
            return self.activation_function(s)

    def __init__(self):
        self.layers = []
        self.input = 0

    def reset(self):
        for layer in self.layers:
            layer.reset()

    def set_input(self, size):
        assert not self.layers, "input has to be set before creating layers"
        self.input = size

    def add_layer(self, nodes, activation="relu"):
        assert self.input, "input size cannot be 0"
        input_size = self.input if not self.layers else self.layers[-1].output_size
        self.layers.append(self.Layer(input_size, nodes, activation))

    def predict(self, x):
        return self.forward_pass(x)[-1]

    def forward_pass(self, x):
        acc = []
        partial_output = x
        for layer in self.layers:
            partial_output = layer.forward_pass(partial_output)
            acc.append(partial_output)
        return acc

    def compute_gradients(self, x, y, reg_lambda):
        batch_size = x.shape[1]
        [h, p] = self.forward_pass(x)
        g = - (y - p)

        grad_W_2 = 1 / batch_size * g @ h.T + 2 * reg_lambda * self.layers[1].W
        grad_b_2 = 1 / batch_size * g @ np.ones(shape=(batch_size, 1))

        g = self.layers[1].W.T @ g
        g[h <= 0] = 0
        # g = np.multiply(g, np.heaviside(h, 0))

        grad_W_1 = 1 / batch_size * g @ x.T + 2 * reg_lambda * self.layers[0].W
        grad_b_1 = 1 / batch_size * g @ np.ones(shape=(batch_size, 1))

        return grad_W_2, grad_b_2, grad_W_1, grad_b_1

    def compute_cost(self, x, y, reg_lambda):
        reg = reg_lambda * (np.sum(self.layers[0].W ** 2) + np.sum(self.layers[1].W ** 2))
        return self.compute_loss(x, y) + reg

    def compute_loss(self, x, y):
        batch_size = x.shape[1]
        p = self.predict(x)
        loss = - np.log(np.sum(y * p, axis=0))
        return np.sum(loss) / batch_size

    def accuracy(self, x, y):  # equivalent to function acc = ComputeAccuracy(X, y, W, b)
        prediction = self.predict(x)
        a = np.argmax(prediction, axis=0) == np.argmax(y, axis=0)
        return np.count_nonzero(a) / np.size(a)

    def fit(self, x, y, x_val=None, y_val=None, n_batch=100, eta_min=1e-5, eta_max=1e-1, ns=500, k=1, reg_lambda=0.01,
            jitter=False):
        data_points = x.shape[1]
        train_cost, val_cost, train_loss, val_loss, train_accuracy, val_accuracy = [], [], [], [], [], []
        n_epochs = int(2 * ns * k / (data_points / n_batch))
        cycle = [eta_min + (t / ns) * (eta_max - eta_min) for t in range(ns)] \
                + [eta_max - ((t - ns) / ns) * (eta_max - eta_min) for t in range(ns, 2 * ns)]
        cycle *= k
        cycle = iter(cycle)
        for _ in tqdm(range(n_epochs)):
            for i in range(int(data_points / n_batch) - 1):
                start = i * n_batch
                end = min((i + 1) * n_batch, data_points)
                x_batch = x[:, start:end]
                y_batch = y[:, start:end]
                if jitter:
                    noise = np.random.normal(0, 0.001, size=x_batch.shape)
                    x_batch = x_batch + noise

                eta = next(cycle)

                grad_W_2, grad_b_2, grad_W_1, grad_b_1 = self.compute_gradients(x_batch, y_batch, reg_lambda)
                self.layers[0].W -= eta * grad_W_1
                self.layers[0].b -= eta * grad_b_1
                self.layers[1].W -= eta * grad_W_2
                self.layers[1].b -= eta * grad_b_2

            train_cost.append(self.compute_cost(x, y, reg_lambda))
            train_loss.append(self.compute_loss(x, y))
            train_accuracy.append(self.accuracy(x, y))
            if x_val is not None:
                val_cost.append(self.compute_cost(x_val, y_val, reg_lambda))
                val_loss.append(self.compute_loss(x_val, y_val))
                val_accuracy.append(self.accuracy(x_val, y_val))
        return (train_cost, val_cost), (train_loss, val_loss), (train_accuracy, val_accuracy)

    def compute_gradients_num(self, x, y, lam, h=1e-5):
        W1 = self.layers[0].W
        W2 = self.layers[1].W

        b1 = self.layers[0].b
        b2 = self.layers[1].b

        grad_W1 = np.zeros(W1.shape)
        grad_b1 = np.zeros(b1.shape)
        grad_W2 = np.zeros(W2.shape)
        grad_b2 = np.zeros(b2.shape)

        c = self.compute_cost_num(x, y, W1, W2, b1, b2, lam)

        for i in range(len(b1)):
            b1_try = np.array(b1)
            b1_try[i] += h
            c2 = self.compute_cost_num(x, y, W1, W2, b1_try, b2, lam)
            grad_b1[i] = (c2 - c) / h

        for i in range(W1.shape[0]):
            for j in range(W1.shape[1]):
                W1_try = np.array(W1)
                W1_try[i, j] += h
                c2 = self.compute_cost_num(x, y, W1_try, W2, b1, b2, lam)
                grad_W1[i, j] = (c2 - c) / h

        for i in range(len(b2)):
            b2_try = np.array(b2)
            b2_try[i] += h
            c2 = self.compute_cost_num(x, y, W1, W2, b1, b2_try, lam)
            grad_b2[i] = (c2 - c) / h

        for i in range(W2.shape[0]):
            for j in range(W2.shape[1]):
                W2_try = np.array(W2)
                W2_try[i, j] += h
                c2 = self.compute_cost_num(x, y, W1, W2_try, b1, b2, lam)
                grad_W2[i, j] = (c2 - c) / h

        return grad_W2, grad_b2, grad_W1, grad_b1

    def compute_gradients_num_slow(self, x, y, lam, h=1e-5):
        W1 = self.layers[0].W
        W2 = self.layers[1].W

        b1 = self.layers[0].b
        b2 = self.layers[1].b

        grad_W1 = np.zeros(W1.shape)
        grad_b1 = np.zeros(b1.shape)
        grad_W2 = np.zeros(W2.shape)
        grad_b2 = np.zeros(b2.shape)

        for i in range(len(b1)):
            b1_try = np.array(b1)
            b1_try[i] -= h
            c1 = self.compute_cost_num(x, y, W1, W2, b1_try, b2, lam)

            b1_try = np.array(b1)
            b1_try[i] += h
            c2 = self.compute_cost_num(x, y, W1, W2, b1_try, b2, lam)

            grad_b1[i] = (c2 - c1) / (2 * h)

        for i in range(W1.shape[0]):
            for j in range(W1.shape[1]):
                W1_try = np.array(W1)
                W1_try[i, j] -= h
                c1 = self.compute_cost_num(x, y, W1_try, W2, b1, b2, lam)

                W1_try = np.array(W1)
                W1_try[i, j] += h
                c2 = self.compute_cost_num(x, y, W1_try, W2, b1, b2, lam)

                grad_W1[i, j] = (c2 - c1) / (2 * h)

        for i in range(len(b2)):
            b2_try = np.array(b2)
            b2_try[i] -= h
            c1 = self.compute_cost_num(x, y, W1, W2, b1, b2_try, lam)

            b2_try = np.array(b2)
            b2_try[i] += h
            c2 = self.compute_cost_num(x, y, W1, W2, b1, b2_try, lam)

            grad_b2[i] = (c2 - c1) / (2 * h)

        for i in range(W2.shape[0]):
            for j in range(W2.shape[1]):
                W2_try = np.array(W2)
                W2_try[i, j] -= h
                c1 = self.compute_cost_num(x, y, W1, W2_try, b1, b2, lam)

                W2_try = np.array(W2)
                W2_try[i, j] += h
                c2 = self.compute_cost_num(x, y, W1, W2_try, b1, b2, lam)

                grad_W2[i, j] = (c2 - c1) / (2 * h)

        return grad_W2, grad_b2, grad_W1, grad_b1

    def compute_cost_num(self, x, y, W1, W2, b1, b2, reg_lambda):
        batch_size = x.shape[1]

        s = np.maximum(0, W1 @ x + b1)
        p = softmax(W2 @ s + b2)
        loss = -np.log(np.sum(y * p, axis=0))

        reg = reg_lambda * (np.sum(W1 ** 2) + np.sum(W2 ** 2))

        return np.sum(loss) / batch_size + reg
