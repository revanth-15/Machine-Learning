import unittest
from src.layers.linear import Linear
from src.layers.sigmoid import Sigmoid
from src.layers.relu import ReLU
from src.layers.binary_cross_entropy import BinaryCrossEntropy

class TestLayers(unittest.TestCase):

    def setUp(self):
        self.linear_layer = Linear(input_size=2, output_size=1)
        self.sigmoid_layer = Sigmoid()
        self.relu_layer = ReLU()
        self.bce_layer = BinaryCrossEntropy()

    def test_linear_forward(self):
        x = [[1, 2], [3, 4]]
        expected_output = [[5], [11]]  # Assuming weights and bias are set to produce this output
        output = self.linear_layer.forward(x)
        self.assertEqual(output, expected_output)

    def test_linear_backward(self):
        x = [[1, 2], [3, 4]]
        grad_output = [[1], [1]]
        expected_grad_input = [[1, 2], [3, 4]]  # Assuming weights and bias are set to produce this gradient
        self.linear_layer.forward(x)
        grad_input = self.linear_layer.backward(grad_output)
        self.assertEqual(grad_input, expected_grad_input)

    def test_sigmoid_forward(self):
        x = [[0], [1], [2]]
        expected_output = [[0.5], [0.7311], [0.8808]]  # Sigmoid outputs
        output = self.sigmoid_layer.forward(x)
        self.assertAlmostEqual(output[0][0], expected_output[0][0], places=4)
        self.assertAlmostEqual(output[1][0], expected_output[1][0], places=4)
        self.assertAlmostEqual(output[2][0], expected_output[2][0], places=4)

    def test_sigmoid_backward(self):
        x = [[0], [1], [2]]
        grad_output = [[1], [1], [1]]
        self.sigmoid_layer.forward(x)
        grad_input = self.sigmoid_layer.backward(grad_output)
        # Check if the gradient is computed correctly
        self.assertEqual(len(grad_input), len(grad_output))

    def test_relu_forward(self):
        x = [[-1], [0], [1]]
        expected_output = [[0], [0], [1]]  # ReLU outputs
        output = self.relu_layer.forward(x)
        self.assertEqual(output, expected_output)

    def test_relu_backward(self):
        x = [[-1], [0], [1]]
        grad_output = [[1], [1], [1]]
        self.relu_layer.forward(x)
        grad_input = self.relu_layer.backward(grad_output)
        # Check if the gradient is computed correctly
        self.assertEqual(len(grad_input), len(grad_output))

    def test_bce_forward(self):
        y_true = [[1], [0]]
        y_pred = [[0.9], [0.1]]
        expected_loss = 0.1054  # Example expected loss
        loss = self.bce_layer.forward(y_true, y_pred)
        self.assertAlmostEqual(loss, expected_loss, places=4)

    def test_bce_backward(self):
        y_true = [[1], [0]]
        y_pred = [[0.9], [0.1]]
        grad_output = self.bce_layer.backward(y_true, y_pred)
        # Check if the gradient is computed correctly
        self.assertEqual(len(grad_output), len(y_true))

if __name__ == '__main__':
    unittest.main()