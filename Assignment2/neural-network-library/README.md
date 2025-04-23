# Neural Network Library

This project implements a neural network library that allows users to easily construct and train neural networks with an arbitrary number of layers and nodes. The library is designed to provide a clear understanding of the fundamental components of neural networks, including layers, activation functions, loss functions, and model management.

## Features

- **Layer Classes**: Implementations of various layer types, including linear layers, activation functions (sigmoid and ReLU), and loss functions (binary cross-entropy).
- **Model Management**: A `Sequential` class that allows users to stack layers and manage the forward and backward passes.
- **Weight Saving and Loading**: Functions to save and load model weights, enabling easy model persistence and sharing.
- **Testing Framework**: Unit tests for layers, models, and utility functions to ensure reliability and correctness.

## Installation

To install the required dependencies, run:

```
pip install -r requirements.txt
```

## Usage

1. **Import the Library**: Import the necessary classes from the library to create your neural network.

2. **Construct a Neural Network**: Use the `Sequential` class to build your model by adding layers.

3. **Train the Model**: Implement the training loop, including forward and backward passes, and update the weights.

4. **Evaluate the Model**: Test the model on validation and test datasets to assess performance.

5. **Save and Load Weights**: Use the provided utility functions to save your trained model's weights and load them for future use.

## Example

Here is a simple example of how to create a neural network for the XOR problem:

```python
from src.models.sequential import Sequential
from src.layers.linear import Linear
from src.layers.sigmoid import Sigmoid
from src.utils.save_load import save_weights, load_weights

# Create a neural network
model = Sequential()
model.add(Linear(input_size=2, output_size=2))
model.add(Sigmoid())
model.add(Linear(input_size=2, output_size=1))
model.add(Sigmoid())

# Train the model on XOR data...

# Save the model weights
save_weights(model, 'XOR_solved.w')

# Load the model weights
load_weights(model, 'XOR_solved.w')
```

## Testing

To run the tests, navigate to the `tests` directory and execute:

```
pytest
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.