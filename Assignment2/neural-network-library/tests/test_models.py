import unittest
from src.models.sequential import Sequential
from src.layers.linear import Linear
from src.layers.sigmoid import Sigmoid
from src.layers.relu import ReLU
from src.layers.binary_cross_entropy import BinaryCrossEntropy

class TestSequentialModel(unittest.TestCase):

    def setUp(self):
        self.model = Sequential()
        self.model.add(Linear(input_size=2, output_size=2))
        self.model.add(Sigmoid())
        self.model.add(Linear(input_size=2, output_size=1))
        self.model.add(Sigmoid())

    def test_forward_pass(self):
        input_data = [[0.1, 0.2], [0.3, 0.4]]
        output = self.model.forward(input_data)
        self.assertEqual(len(output), 2)
        self.assertEqual(len(output[0]), 1)

    def test_backward_pass(self):
        input_data = [[0.1, 0.2], [0.3, 0.4]]
        target = [[1], [0]]
        output = self.model.forward(input_data)
        loss = BinaryCrossEntropy().forward(output, target)
        gradients = self.model.backward(loss)
        self.assertIsNotNone(gradients)

    def test_add_layer(self):
        self.model.add(ReLU())
        self.assertEqual(len(self.model.layers), 4)

if __name__ == '__main__':
    unittest.main()