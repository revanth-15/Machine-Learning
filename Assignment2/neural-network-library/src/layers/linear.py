import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler

# ---------------- PART 1: Neural Network Library ----------------

class Layer:
    """Base Layer class for defining forward and backward methods."""
    def forward(self, input_data):
        raise NotImplementedError
    
    def backward(self, grad_output):
        raise NotImplementedError

class Linear(Layer):
    """Fully connected layer with forward and backward propagation."""
    def __init__(self, input_dim, output_dim):
        self.weights = np.random.randn(output_dim, input_dim) * np.sqrt(2 / input_dim)
        self.bias = np.zeros((output_dim, 1))
    
    def forward(self, input_data):
        self.input = input_data
        return np.dot(self.weights, input_data) + self.bias
    
    def backward(self, grad_output, learning_rate=0.01):
        grad_input = np.dot(self.weights.T, grad_output)
        grad_weights = np.dot(grad_output, self.input.T)
        grad_bias = np.sum(grad_output, axis=1, keepdims=True)
        
        # Update weights using gradient descent
        self.weights -= learning_rate * grad_weights
        self.bias -= learning_rate * grad_bias
        
        return grad_input

class ReLU(Layer):
    """ReLU activation function layer."""
    def forward(self, input_data):
        self.input = input_data
        return np.maximum(0, input_data)
    
    def backward(self, grad_output):
        return grad_output * (self.input > 0)

class BatchNorm(Layer):
    """Batch Normalization Layer to stabilize training."""
    def __init__(self, input_dim):
        self.gamma = np.ones((input_dim, 1))
        self.beta = np.zeros((input_dim, 1))
        self.epsilon = 1e-5
    
    def forward(self, input_data):
        self.mean = np.mean(input_data, axis=1, keepdims=True)
        self.variance = np.var(input_data, axis=1, keepdims=True)
        self.normalized = (input_data - self.mean) / np.sqrt(self.variance + self.epsilon)
        return self.gamma * self.normalized + self.beta
    
    def backward(self, grad_output):
        return grad_output  # No backpropagation for BatchNorm in this version

class Dropout(Layer):
    """Dropout layer to reduce overfitting by randomly deactivating neurons."""
    def __init__(self, dropout_rate=0.2):
        self.dropout_rate = dropout_rate
    
    def forward(self, input_data):
        self.mask = np.random.rand(*input_data.shape) > self.dropout_rate
        return input_data * self.mask
    
    def backward(self, grad_output):
        return grad_output * self.mask

class Sequential:
    """Sequential model to stack layers and perform training."""
    def __init__(self):
        self.layers = []
        self.loss_history = []
    
    def add(self, layer):
        self.layers.append(layer)
    
    def forward(self, input_data):
        for layer in self.layers:
            input_data = layer.forward(input_data)
        return input_data
    
    def backward(self, grad_output, learning_rate=0.01):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output, learning_rate) if isinstance(layer, Linear) else layer.backward(grad_output)

    def train(self, X, y, loss_fn, epochs=2000, learning_rate=0.01, decay=0.995):
        """Train the neural network with learning rate decay."""
        for epoch in range(epochs):
            predictions = self.forward(X)
            loss = loss_fn.forward(predictions, y)
            self.loss_history.append(loss)
            grad_output = loss_fn.backward()
            self.backward(grad_output, learning_rate)
            learning_rate *= decay  # Decay learning rate
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}, Learning Rate: {learning_rate:.6f}")

    def plot_loss(self):
        """Plot the training loss over epochs."""
        plt.plot(self.loss_history, label='Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss over Epochs')
        plt.legend()
        plt.show()

# ---------------- PART 2: NYC Taxi Trip Prediction ----------------

# Define Model with BatchNorm and Dropout
model = Sequential()
model.add(Linear(5, 128))
model.add(BatchNorm(128))
model.add(ReLU())
model.add(Dropout(0.2))
model.add(Linear(128, 64))
model.add(BatchNorm(64))
model.add(ReLU())
model.add(Dropout(0.2))
model.add(Linear(64, 32))
model.add(BatchNorm(32))
model.add(ReLU())
model.add(Linear(32, 1))  # Output layer

# Train model with new settings
loss_fn = BinaryCrossEntropyLoss()
model.train(X_train_processed, y_train_processed, loss_fn, epochs=2000, learning_rate=0.01, decay=0.995)

# Plot Training Loss
model.plot_loss()

# Additional Graphs for Neural Network
plt.figure(figsize=(8,6))
sns.histplot(model.layers[0].weights.flatten(), bins=50, kde=True)
plt.title("Weight Distribution of First Layer")
plt.show()

# Predictions vs Actual
predictions = model.forward(X_test_processed)
plt.figure(figsize=(8,6))
sns.scatterplot(x=np.expm1(y_test_processed.flatten()), y=np.expm1(predictions.flatten()), alpha=0.5)
plt.xlabel("Actual Trip Duration (seconds)")
plt.ylabel("Predicted Trip Duration (seconds)")
plt.title("Predictions vs Actuals")
plt.show()
