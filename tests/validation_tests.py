import validation
import unittest
import numpy as np

import numpy.testing as nptest


class ValidationTest(unittest.TestCase):
    def test_from_categorical(self):
        print('Testing validation.from_categorical')
        x = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0]
        ])
        y = [2, 1, 0]
        self.assertEqual(validation.from_categorical(x), y)

    def test_extract_validation(self):
        print('Testing validation.extract_validation')
        x1 = [0, 2, 0, 1, 1, 3, 1, 2, 0, 3]
        x2 = [0, 2, 0, 1, 2, 3, 1, 2, 0, 2]

        (acc, accs, matrix) = validation.extract_validation(x1, x2, categorical=False)

        self.assertAlmostEqual(acc, 0.8, places=4)
        accs_correct = [1.0, 0.6667, 1.0, 0.5]
        nptest.assert_array_almost_equal(accs, accs_correct, decimal=4)

        nptest.assert_array_equal(matrix, [
            [3, 0, 0, 0],
            [0, 2, 1, 0],
            [0, 0, 2, 0],
            [0, 0, 1, 1]
        ])
