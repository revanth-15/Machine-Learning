class Sigmoid(Layer):
    def __init__(self):
        super().__init__()
        self.output = None

    def forward(self, input):
        self.output = 1 / (1 + np.exp(-input))
        return self.output

    def backward(self, upstream_gradient):
        return upstream_gradient * (self.output * (1 - self.output))