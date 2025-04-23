import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import pickle
class Multi_LinearRegression:
    def __init__(self, learning_rate=0.001, regularization_strength=0, num_epochs=100, patience=3):
        self.learning_rate = learning_rate
        self.regularization_strength = regularization_strength
        self.num_epochs = num_epochs
        self.weights = None
        self.bias = None
        self.loss_history = []
        self.patience = patience

    def mean_squared_error_loss(self, y_true, y_pred):
        return np.mean((y_pred - y_true)**2)

    def fit(self, X, y):
        num_samples, num_features = X.shape
        num_outputs = y.shape[1]

        # Initialize weights and bias
        self.weights = np.zeros((num_features, num_outputs))
        self.bias = np.zeros(num_outputs)
        best_loss = float('inf')

        # Gradient Descent with L2 regularization
        for epoch in range(self.num_epochs):
            # Forward pass
            y_pred = np.dot(X, self.weights) + self.bias

            # Compute gradients
            dw = (1/num_samples) * np.dot(X.T, (y_pred - y)) + (self.regularization_strength / num_samples) * self.weights
            db = (1/num_samples) * np.sum(y_pred - y, axis=0)  # bias doesn't get regularized

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Compute loss
            loss = self.mean_squared_error_loss(y, y_pred)
            self.loss_history.append(loss)

            print(f"Epoch {epoch}, Loss: {loss}")

            if loss < best_loss:
                best_loss = loss
                patience_count = 0
            else:
                patience_count += 1

            if patience_count >= self.patience:
                print(f"\nStopping early at epoch {epoch + 1}")
                break

        print("Loss History:\n", self.loss_history)
        plt.plot(np.arange(len(self.loss_history)), self.loss_history)
        plt.title("Loss Curve")
        plt.xlabel("Steps/epochs")
        plt.ylabel("Loss")
        plt.show()


    def predict(self, X):
        return np.dot(X, self.weights) + self.bias





iris= load_iris()
X= iris.data[:,[0,1]]  # sepal length , sepal width
y= iris.data[:,[2,3]]  # petal length , petal width
# print("Xshape",X.shape)
# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


# Instantiate the model
model = Multi_LinearRegression(learning_rate=0.01, regularization_strength=0)

# Fit the model
model.fit(X_train, y_train)

with open('mlr_model.pkl', 'wb') as f:
    pickle.dump(model,f)
