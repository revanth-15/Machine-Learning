import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, lr=0.01, batch_size=32, regularization=0, max_epochs=100, patience=3):
        """Linear Regression using Gradient Descent.

        Parameters:
        -----------
        batch_size: int
            The number of samples per batch.
        regularization: float
            The regularization parameter.
        max_epochs: int
            The maximum number of epochs.
        patience: int
            The number of epochs to wait before stopping if the validation loss
            does not improve.
        """
        self.lr = lr
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience
        self.weights = None
        self.bias = None
        self.batch_loss = []


    def fit(self, X, y, batch_size=32, regularization=0, max_epochs=100, patience=3):
        """Fit a linear model.

        Parameters:
        -----------
        batch_size: int
            The number of samples per batch.
        regularization: float
            The regularization parameter.
        max_epochs: int
            The maximum number of epochs.
        patience: int
            The number of epochs to wait before stopping if the validation loss
            does not improve.
        """
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience

        # TODO: Initialize the weights and bias based on the shape of X and y.
        try:
            num_of_samples, num_of_features = X.shape
        except ValueError:
            num_of_samples, num_of_features = X[:, None].shape

        self.weights = np.zeros(num_of_features)
        self.bias = 0
        best_loss= float('inf')
        # TODO: Implement the training loop.
        indices = np.arange(num_of_samples)
        np.random.shuffle(indices)
        for _ in range(self.max_epochs):
            for start in range(0, num_of_samples, self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                try:
                    y_predicted = np.dot(X, self.weights) + self.bias
                except ValueError:
                    y_predicted = np.dot(X[batch_indices[:, None]], self.weights) + self.bias

                dw = (1 / self.batch_size) * np.dot(X[batch_indices].T, (y_predicted - y[batch_indices])) + (self.regularization / self.batch_size) * self.weights
                db = (1 / self.batch_size) * np.sum(y_predicted - y[batch_indices])


                self.weights = self.weights - self.lr * dw
                self.bias = self.bias - self.lr * db

                loss = np.mean(((y_predicted - y[batch_indices]) ** 2)) # MSE
                self.batch_loss.append(loss)


                # print(f"Batch {start} Loss :\n", self.batch_loss)
                # print("Average_batch_loss=", abl)
                if max_epochs % 100 == 0:
                    print(f"Epoch {_}, Loss: {loss}")

            # early stopping
            if loss < best_loss:
                best_loss = loss
                patience_count = 0
            else:
                patience_count +=1

            if patience_count >=patience:
                print(f"\nStopping early at epoch {_+1}")
                break


        print(f"Best weights is {self.weights} and best bias is {self.bias}: ")
        print("Loss:",self.batch_loss)
        plt.plot(np.arange(len(self.batch_loss)), self.batch_loss)
        plt.xlabel("Step(1 step= 100 units)")
        plt.ylabel("Loss")
        plt.title("Loss vs Step Number")
        plt.show()



    def predict(self, X):
        """Predict using the linear model.

        Parameters
        ----------
        X: numpy.ndarray
            The input data.
        """
        # TODO: Implement the prediction function.
        try:
            y_predicted = np.dot(X, self.weights) + self.bias
        except ValueError:
            y_predicted = np.dot(X[:, None], self.weights) + self.bias
        return y_predicted

    def score(self, X, y):
        """Evaluate the linear model using the mean squared error.

        Parameters
        ----------
        X: numpy.ndarray
            The input data.
        y: numpy.ndarray
            The target data.
        """
        # TODO: Implement the scoring function.
        try:
            y_predicted = np.dot(X, self.weights) + self.bias
        except ValueError:
            y_predicted = np.dot(X[:, None], self.weights) + self.bias
        return np.mean(((y_predicted - y) ** 2))  # MSE
