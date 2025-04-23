import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class LogisticRegression:
    def __init__(self, lr=0.01, epoch=100):
        self.weights = None
        self.bias = None
        self.lr = lr
        self.epoch = epoch

    def predict(self, X):

        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        z = np.dot(X, self.weights) + self.bias
        pred = self.sigmoid(z)
        return (pred >= 0.5).astype(int)

    def fit(self, X, y):

        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        num_of_samples, num_of_features = X.shape
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        for _ in range(self.epoch):
            z = np.dot(X, self.weights) + self.bias
            pred = self.sigmoid(z)

            dw = np.dot(X.T, (pred - y)) / num_of_samples
            db = np.sum(pred - y) / num_of_samples

            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        print(z)


    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))


