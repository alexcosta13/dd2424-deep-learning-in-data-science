import numpy as np
from tqdm import tqdm
from utils import softmax, sample_from_probability, index_2_one_hot, indices_2_chars


class RNN:
    def __init__(self, input_size, output_size, hidden_size=100, **kwargs):
        np.random.seed(400)
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.W = None
        self.U = None
        self.b = None
        self.V = None
        self.c = None
        self.h = None
        self.char_2_indices = kwargs['char_2_indices']
        self.indices_2_char = kwargs['indices_2_char']
        self.kwargs = kwargs
        self.initialize_weights()

        self.last_h = None
        self.last_p = None

        self.grad_W = None
        self.grad_U = None
        self.grad_b = None
        self.grad_V = None
        self.grad_c = None

    def initialize_weights(self):
        np.random.seed(400)
        sigma = self.kwargs["sigma"] if "sigma" in self.kwargs else 0.01
        self.W = np.random.normal(loc=0, scale=sigma, size=(self.hidden_size, self.hidden_size))
        self.U = np.random.normal(loc=0, scale=sigma, size=(self.hidden_size, self.input_size))
        self.b = np.zeros(shape=(self.hidden_size, 1))
        self.V = np.random.normal(loc=0, scale=sigma, size=(self.output_size, self.hidden_size))
        self.c = np.zeros(shape=(self.output_size, 1))
        self.h = np.zeros(shape=(self.hidden_size, 1))

    def generate(self, x0=None, h0=None, seq_length=25):
        if x0 is None:
            pass
            x0 = index_2_one_hot(0, self.output_size)
        if h0 is None:
            h0 = np.zeros((self.hidden_size, 1))
        output = []
        self.h = h0
        xt = x0
        for _ in range(seq_length):
            at = self.W @ self.h + self.U @ xt + self.b
            self.h = np.tanh(at)
            ot = self.V @ self.h + self.c
            pt = softmax(ot).squeeze()
            sample = sample_from_probability(self.output_size, pt)
            output.append(sample[0])
            xt = index_2_one_hot(sample, self.output_size)
        return indices_2_chars(output, self.indices_2_char)

    def forward(self, x, h0=None):
        if h0 is None:
            self.h = np.zeros(shape=(self.hidden_size, 1))
        else:
            self.h = h0
        self.last_h = [self.h]
        self.last_p = []

        for i in range(x.shape[1]):
            at = self.W @ self.h + self.U @ x[:, [i]] + self.b
            self.h = np.tanh(at)
            ot = self.V @ self.h + self.c
            pt = softmax(ot)#.squeeze()
            self.last_h.append(self.h)
            self.last_p.append(pt)

        #self.last_h = np.array(self.last_h)
        #self.last_p = np.array(self.last_p)
        return self.last_p

    def compute_gradients(self, x, y, reg_lambda=0.0):
        seq_length = x.shape[1]
        self.forward(x)

        self.grad_W = np.zeros(self.W.shape)
        self.grad_U = np.zeros(self.U.shape)
        self.grad_b = np.zeros(self.b.shape)
        self.grad_V = np.zeros(self.V.shape)
        self.grad_c = np.zeros(self.c.shape)

        dLdo = []

        for t in range(seq_length):
            dLdo.append(- (y[:, [t]] - self.last_p[t]).T)
            g = dLdo[t]
            self.grad_V += g.T @ self.last_h[t + 1].T
            self.grad_c += g.T

        dLda = np.zeros((1, self.hidden_size))

        for t in reversed(range(seq_length - 1)):
            dLdh = dLdo[t] @ self.V + dLda @ self.W
            dLda = dLdh @ np.diag(1 - (self.last_h[t + 1].squeeze() ** 2))
            g = dLda
            self.grad_W += g.T @ self.last_h[t].T
            self.grad_U += g.T @ x[:, [t]].T
            self.grad_b += g.T

        return self.grad_W, self.grad_U, self.grad_b, self.grad_V, self.grad_c

    def update_weights(self, eta):
        self.m_theta_W += self.grad_W ** 2
        self.m_theta_U += self.grad_U ** 2
        self.m_theta_b += self.grad_b ** 2
        self.m_theta_V += self.grad_V ** 2
        self.m_theta_c += self.grad_c ** 2

        denom = (self.m_theta['dLd' + key] + 1e-10) ** -0.5
        self.params[key] = self.params[key] - self.eta * np.multiply(denom, self.gradients['dLd' + key])
    self.W -= eta * self.grad_W
        self.U -= eta * self.grad_U
        self.b -= eta * self.grad_b
        self.V -= eta * self.grad_V
        self.c -= eta * self.grad_c

    def compute_loss(self, x, y):
        loss = 0
        for p in self.last_p:
            loss += - np.sum(np.log(np.sum(y * p, axis=0)))
        return loss

    def compute_cost(self, y0, pt):
        loss = 0
        for t in range(len(pt)):
            y = np.reshape(y0.T[t], (-1, 1))
            loss -= sum(np.log(np.dot(y.T, pt[t])))
        return loss

    def fit(self, x, y, x_val=None, y_val=None, n_epochs=25, n_batch=100, eta=0.1, reg_lambda=0.01, seq_length=25):
        data_points = x.shape[1]
        train_loss, val_loss, train_accuracy, val_accuracy = [], [], [], []
        for _ in tqdm(range(n_epochs)):
            for i in range(int(data_points / n_batch) - 1):
                start = i * n_batch
                end = min((i + 1) * n_batch, data_points)
                x_batch = x[:, start:end]
                y_batch = y[:, start:end]

                self.compute_gradients(x_batch, y_batch, reg_lambda)
                self.update_weights(eta)

            train_loss.append(self.compute_loss(x, y))
            train_accuracy.append(self.accuracy(x, y))
            if x_val is not None:
                val_loss.append(self.compute_loss(x_val, y_val))
                val_accuracy.append(self.accuracy(x_val, y_val))
        return (train_loss, val_loss), (train_accuracy, val_accuracy)

    def compute_gradients_num(self, x, y, reg_lambda=0.0, h=1e-4):
        def compute_loss():
            loss = 0
            hidden_state = np.zeros(shape=(self.hidden_size, 1))
            for i in range(x.shape[1]):
                a = self.W @ hidden_state + self.U @ x[:, [i]] + self.b
                hidden_state = np.tanh(a)
                ot = self.V @ hidden_state + self.c
                p = softmax(ot)
                loss += - np.sum(np.log(np.sum(y * p, axis=0)))
            return loss

        grad_W = np.zeros_like(self.W)
        grad_U = np.zeros_like(self.U)
        grad_b = np.zeros_like(self.b)
        grad_V = np.zeros_like(self.V)
        grad_c = np.zeros_like(self.c)

        # Check gradient for W
        for i in range(len(self.W)):
            for j in range(len(self.W[i])):
                self.W[i][j] -= h
                c1 = compute_loss()
                self.W[i][j] += 2 * h
                c2 = compute_loss()
                grad_W[i][j] = (c2 - c1) / (2 * h)
                self.W[i][j] -= h

        # Check gradient for U
        for i in range(len(self.U)):
            for j in range(len(self.U[i])):
                self.U[i][j] -= h
                c1 = compute_loss()
                self.U[i][j] += 2 * h
                c2 = compute_loss()
                grad_U[i][j] = (c2 - c1) / (2 * h)
                self.U[i][j] -= h

        # Check gradient for b
        for i in range(len(self.b)):
            self.b[i] -= h
            c1 = compute_loss()
            self.b[i] += 2 * h
            c2 = compute_loss()
            grad_b[i] = (c2 - c1) / (2 * h)
            self.b[i] -= h

        # Check gradient for V
        for i in range(len(self.V)):
            for j in range(len(self.V[i])):
                self.V[i][j] -= h
                c1 = compute_loss()
                self.V[i][j] += 2 * h
                c2 = compute_loss()
                grad_V[i][j] = (c2 - c1) / (2 * h)
                self.V[i][j] -= h

        # Check gradient for c
        for i in range(len(self.c)):
            self.c[i] -= h
            c1 = compute_loss()
            self.c[i] += 2 * h
            c2 = compute_loss()
            grad_c[i] = (c2 - c1) / (2 * h)
            self.c[i] -= h

        return grad_W, grad_U, grad_b, grad_V, grad_c
