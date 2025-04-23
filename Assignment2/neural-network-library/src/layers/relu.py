class ReLU(Layer):
    def __init__(self):
        super().__init__()
        self.input = None

    def forward(self, x):
        self.input = x
        return np.maximum(0, x)

    def backward(self, dout):
        dx = dout * (self.input > 0)
        return dx