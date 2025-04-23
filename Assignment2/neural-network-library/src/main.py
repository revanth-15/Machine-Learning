# Neural Network Library Main Entry Point

from layers.linear import Linear
from layers.sigmoid import Sigmoid
from layers.relu import ReLU
from layers.binary_cross_entropy import BinaryCrossEntropy
from models.sequential import Sequential
import numpy as np

def main():
    # Example: Constructing a simple neural network to solve the XOR problem
    model = Sequential()
    model.add(Linear(input_size=2, output_size=2))  # Hidden layer with 2 nodes
    model.add(Sigmoid())
    model.add(Linear(input_size=2, output_size=1))  # Output layer
    model.add(Sigmoid())

    # XOR input and output
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Training the model (pseudo-code)
    learning_rate = 0.1
    for epoch in range(10000):
        # Forward pass
        predictions = model.forward(X)

        # Compute loss
        loss = BinaryCrossEntropy().forward(predictions, y)

        # Backward pass
        model.backward(y)

        # Update weights (pseudo-code)
        for layer in model.layers:
            layer.weights -= learning_rate * layer.d_weights
            layer.bias -= learning_rate * layer.d_bias

        if epoch % 1000 == 0:
            print(f'Epoch {epoch}, Loss: {loss}')

    # Save the model weights
    model.save_weights('XOR_solved.w')

if __name__ == "__main__":
    main()