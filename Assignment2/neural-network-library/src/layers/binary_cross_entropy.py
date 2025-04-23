class BinaryCrossEntropy(Layer):
    def __init__(self):
        super().__init__()
        self.y_pred = None
        self.y_true = None

    def forward(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        loss = - (y_true * np.log(y_pred + 1e-15) + (1 - y_true) * np.log(1 - y_pred + 1e-15))
        return np.mean(loss)

    def backward(self):
        batch_size = self.y_true.shape[0]
        grad = (self.y_pred - self.y_true) / (self.y_pred * (1 - self.y_pred) * batch_size)
        return grad