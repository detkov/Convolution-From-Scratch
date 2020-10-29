import numpy as np
import unittest

from convolution import convolve


class TestConvolution(unittest.TestCase):
    def test_regular_params(self):
        data = np.array([[0, 4, 3, 2, 0, 1, 0], 
                         [4, 3, 0, 1, 0, 1, 0],
                         [1, 3, 4, 2, 0, 1, 0],
                         [3, 4, 2, 2, 0, 1, 0],
                         [0, 0, 0, 0, 0, 1, 0]])

        kernel = np.array([[1, 1, 3], 
                           [0, 2, 3],
                           [3, 3, 3]])

        # with padding = 0, stride = 1, dilation = 1
        result_1 = convolve(data, kernel)
        answer_1 = np.array([[43, 43, 25, 17, 6], 
                             [52, 44, 17, 16, 6],
                             [30, 23, 10, 11, 6]], dtype=float)

        # with padding = 0, stride = 2, dilation = 1
        result_2 = convolve(data, kernel, stride=(2, 2))
        answer_2 = np.array([[43, 25, 6], 
                             [30, 10, 6]], dtype=float)

        # with padding = 0, stride = 1, dilation = 2
        result_3 = convolve(data, kernel, dilation=(2, 2))
        answer_3 = np.array([[11, 19, 3]], dtype=float)

        # with padding = 0, stride = 2, dilation = 2
        result_4 = convolve(data, kernel, stride=(2, 2), dilation=(2, 2))
        answer_4 = np.array([[11, 3]], dtype=float)

        self.assertEqual(result_1.tolist(), answer_1.tolist())
        self.assertEqual(result_2.tolist(), answer_2.tolist())
        self.assertEqual(result_3.tolist(), answer_3.tolist())
        self.assertEqual(result_4.tolist(), answer_4.tolist())


if __name__ == '__main__':
    unittest.main()