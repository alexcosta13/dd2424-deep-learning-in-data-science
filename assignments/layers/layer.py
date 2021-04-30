class Layer:
    def __init__(self, name):
        self.name = name

        self.last_input = None
        self.last_output = None

        self.grad_kernel = None

    def initialize_weights(self):
        pass

    def forward(self, x, training=False):
        self.last_input = self.last_output = x
        return x

    def backward(self, g, reg_lambda):
        return g

    def update_weights(self, eta):
        pass
