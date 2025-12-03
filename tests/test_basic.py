import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import numpy as np
from utils import sigmoid
from config import Config

class TestUtils(unittest.TestCase):
    
    def test_sigmoid_scalar(self):
        """Test sigmoid with scalar value (0 should be 0.5)"""
        self.assertEqual(sigmoid(0), 0.5)
        
    def test_sigmoid_array(self):
        """Test sigmoid with numpy array"""
        x = np.array([0, 0])
        res = sigmoid(x)
        np.testing.assert_array_equal(res, np.array([0.5, 0.5]))

    def test_config_integrity(self):
        """Ensure critical config values are set"""
        self.assertIsInstance(Config.LEARNING_RATE, float)
        self.assertEqual(Config.NUM_FEATURES, 3)

if __name__ == '__main__':
    unittest.main()