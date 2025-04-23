import unittest
from src.utils.save_load import save_weights, load_weights
import numpy as np

class TestUtils(unittest.TestCase):

    def setUp(self):
        self.weights = np.array([[0.1, 0.2], [0.3, 0.4]])
        self.file_path = 'test_weights.npy'

    def test_save_weights(self):
        save_weights(self.weights, self.file_path)
        loaded_weights = load_weights(self.file_path)
        np.testing.assert_array_equal(self.weights, loaded_weights)

    def test_load_weights(self):
        save_weights(self.weights, self.file_path)
        loaded_weights = load_weights(self.file_path)
        self.assertIsInstance(loaded_weights, np.ndarray)
        self.assertEqual(loaded_weights.shape, self.weights.shape)

    def tearDown(self):
        import os
        if os.path.exists(self.file_path):
            os.remove(self.file_path)

if __name__ == '__main__':
    unittest.main()