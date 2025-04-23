import numpy as np

class Layer:
    """Base Layer class that all layers inherit from."""
    def forward(self, input_data):
        raise NotImplementedError
    
    def backward(self, grad_output):
        raise NotImplementedError

class Linear(Layer):
    """Implements a fully connected layer with forward and backward propagation."""
    def __init__(self, input_dim, output_dim):
        self.weights = np.random.randn(output_dim, input_dim) * np.sqrt(2 / input_dim)  # He Initialization
        self.bias = np.zeros((output_dim, 1))
        self.velocity_w = np.zeros_like(self.weights)  # For momentum optimization
        self.velocity_b = np.zeros_like(self.bias)
    
    def forward(self, input_data):
        self.input = input_data
        return np.dot(self.weights, input_data) + self.bias
    
    def backward(self, grad_output, learning_rate=0.01, momentum=0.9):
        grad_input = np.dot(self.weights.T, grad_output)
        grad_weights = np.dot(grad_output, self.input.T)
        grad_bias = np.sum(grad_output, axis=1, keepdims=True)
        
        # Momentum-based updates
        self.velocity_w = momentum * self.velocity_w - learning_rate * grad_weights
        self.velocity_b = momentum * self.velocity_b - learning_rate * grad_bias
        
        self.weights += self.velocity_w
        self.bias += self.velocity_b
        
        return grad_input

class Sigmoid(Layer):
    """Implements the Sigmoid activation function."""
    def forward(self, input_data):
        self.output = 1 / (1 + np.exp(-input_data))
        return self.output
    
    def backward(self, grad_output):
        return grad_output * (self.output * (1 - self.output))

class ReLU(Layer):
    """Implements the ReLU activation function."""
    def forward(self, input_data):
        self.input = input_data
        return np.maximum(0, input_data)
    
    def backward(self, grad_output):
        return grad_output * (self.input > 0)

class Tanh(Layer):
    """Implements the Tanh activation function."""
    def forward(self, input_data):
        self.output = np.tanh(input_data)
        return self.output
    
    def backward(self, grad_output):
        return grad_output * (1 - self.output ** 2)

class BinaryCrossEntropyLoss:
    """Implements Binary Cross-Entropy Loss for classification problems."""
    def forward(self, predictions, targets):
        self.predictions = np.clip(predictions, 1e-9, 1 - 1e-9)
        self.targets = targets
        return -np.mean(targets * np.log(self.predictions) + (1 - targets) * np.log(1 - self.predictions))
    
    def backward(self):
        return (self.predictions - self.targets) / (self.targets * (1 - self.targets) + 1e-9)

class Sequential:
    """Container class for stacking layers and performing forward and backward propagation."""
    def __init__(self):
        self.layers = []
    
    def add(self, layer):
        self.layers.append(layer)
    
    def forward(self, input_data):
        for layer in self.layers:
            input_data = layer.forward(input_data)
        return input_data
    
    def backward(self, grad_output, learning_rate=0.01, momentum=0.9):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output, learning_rate, momentum) if isinstance(layer, Linear) else layer.backward(grad_output)

    def train(self, X, y, loss_fn, epochs=1000, learning_rate=0.01, decay=0.99, momentum=0.9):
        for epoch in range(epochs):
            predictions = self.forward(X)
            loss = loss_fn.forward(predictions, y)
            grad_output = loss_fn.backward()
            self.backward(grad_output, learning_rate, momentum)
            
            # Decaying learning rate for better convergence
            learning_rate *= decay
            
            if epoch % 100 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch}, Loss: {loss:.4f}, Learning Rate: {learning_rate:.6f}")

    def save_weights(self, filename):
        weights = [layer.weights for layer in self.layers if isinstance(layer, Linear)]
        biases = [layer.bias for layer in self.layers if isinstance(layer, Linear)]
        np.savez(filename, weights=weights, biases=biases)

    def load_weights(self, filename):
        data = np.load(filename)
        weights, biases = data['weights'], data['biases']
        idx = 0
        for layer in self.layers:
            if isinstance(layer, Linear):
                layer.weights = weights[idx]
                layer.bias = biases[idx]
                idx += 1

# Test XOR problem with Sigmoid, Tanh
XOR_input = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
XOR_output = np.array([[0, 1, 1, 0]])

for activation in [Sigmoid(), Tanh()]:
    print(f"\nTraining XOR with {activation.__class__.__name__} Activation")
    model = Sequential()
    model.add(Linear(2, 4))  # Increased neurons for better learning
    model.add(activation)
    model.add(Linear(4, 1))
    model.add(Sigmoid())
    
    loss_fn = BinaryCrossEntropyLoss()
    model.train(XOR_input, XOR_output, loss_fn, epochs=2000, learning_rate=0.1, decay=0.995, momentum=0.9)
