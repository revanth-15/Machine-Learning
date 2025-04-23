import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import itertools

# ---------------- PART 1: Neural Network Library ----------------

class Layer:
    def forward(self, input_data):
        raise NotImplementedError
    
    def backward(self, grad_output):
        raise NotImplementedError

class Linear(Layer):
    def __init__(self, input_dim, output_dim):
        self.weights = np.random.randn(output_dim, input_dim) * np.sqrt(2 / input_dim)
        self.bias = np.zeros((output_dim, 1))
        self.velocity_w = np.zeros_like(self.weights)
        self.velocity_b = np.zeros_like(self.bias)
    
    def forward(self, input_data):
        self.input = input_data
        return np.dot(self.weights, input_data) + self.bias
    
    def backward(self, grad_output, learning_rate=0.01, momentum=0.9):
        grad_input = np.dot(self.weights.T, grad_output)
        grad_weights = np.dot(grad_output, self.input.T)
        grad_bias = np.sum(grad_output, axis=1, keepdims=True)
        
        self.velocity_w = momentum * self.velocity_w - learning_rate * grad_weights
        self.velocity_b = momentum * self.velocity_b - learning_rate * grad_bias
        
        self.weights += self.velocity_w
        self.bias += self.velocity_b
        
        return grad_input

class ReLU(Layer):
    def forward(self, input_data):
        self.input = input_data
        return np.maximum(0, input_data)
    
    def backward(self, grad_output):
        return grad_output * (self.input > 0)

class MeanSquaredErrorLoss:
    def forward(self, predictions, targets):
        self.predictions = predictions
        self.targets = targets
        return np.mean((predictions - targets) ** 2)
    
    def backward(self):
        return 2 * (self.predictions - self.targets) / self.targets.size

class Sequential:
    def __init__(self):
        self.layers = []
        self.loss_history = []
    
    def add(self, layer):
        self.layers.append(layer)
    
    def forward(self, input_data):
        for layer in self.layers:
            input_data = layer.forward(input_data)
        return input_data
    
    def backward(self, grad_output, learning_rate=0.01, momentum=0.9):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output, learning_rate, momentum) if isinstance(layer, Linear) else layer.backward(grad_output)

    def train(self, X, y, loss_fn, epochs=2000, learning_rate=0.01, decay=0.995, momentum=0.9):
        for epoch in range(epochs):
            predictions = self.forward(X)
            loss = loss_fn.forward(predictions, y)
            self.loss_history.append(loss)
            grad_output = loss_fn.backward()
            self.backward(grad_output, learning_rate, momentum)
            learning_rate *= decay
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}, Learning Rate: {learning_rate:.6f}")

    def plot_loss(self):
        plt.plot(self.loss_history, label='Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss over Epochs')
        plt.legend()
        plt.show()

# ---------------- PART 2: NYC Taxi Trip Prediction ----------------

file_path = "C:/Users/revan/Downloads/nyc_taxi_data.npy"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Dataset file '{file_path}' not found. Please ensure it is in the correct directory.")

dataset = np.load(file_path, allow_pickle=True).item()
X_train, y_train, X_test, y_test = dataset["X_train"], dataset["y_train"], dataset["X_test"], dataset["y_test"]

X_train = pd.DataFrame(X_train).select_dtypes(include=[np.number]).values
X_test = pd.DataFrame(X_test).select_dtypes(include=[np.number]).values

y_train = np.log1p(y_train).values.reshape(1, -1)
y_test = np.log1p(y_test).values.reshape(1, -1)

scaler = StandardScaler()
X_train_processed = scaler.fit_transform(X_train)
X_test_processed = scaler.transform(X_test)

model = Sequential()
model.add(Linear(X_train_processed.shape[1], 128))
model.add(ReLU())
model.add(Linear(128, 64))
model.add(ReLU())
model.add(Linear(64, 32))
model.add(ReLU())
model.add(Linear(32, 1))

loss_fn = MeanSquaredErrorLoss()
model.train(X_train_processed.T, y_train, loss_fn, epochs=2000, learning_rate=0.01, decay=0.995, momentum=0.9)

model.plot_loss()

predictions = model.forward(X_test_processed.T)
plt.figure(figsize=(8,6))
sns.scatterplot(x=np.expm1(y_test.flatten()), y=np.expm1(predictions.flatten()), alpha=0.5)
plt.xlabel("Actual Trip Duration (seconds)")
plt.ylabel("Predicted Trip Duration (seconds)")
plt.title("Predictions vs Actuals")
plt.show()

# Plot Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Short", "Long"], yticklabels=["Short", "Long"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.show()

# Convert regression output to classification
threshold = np.median(np.expm1(y_test.flatten()))
y_test_class = (np.expm1(y_test.flatten()) > threshold).astype(int)
y_pred_class = (np.expm1(predictions.flatten()) > threshold).astype(int)

plot_confusion_matrix(y_test_class, y_pred_class)
