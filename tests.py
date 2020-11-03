import numpy as np
import unittest

from convolution import conv2d, add_padding


class TestConvolution(unittest.TestCase):
    def test_paddings_shape(self, N: int = 1000):
        for _ in range(N):
            m_h = np.random.randint(3, 100)
            m_w = np.random.randint(3, 100)
            random_matrix = np.random.rand(m_h, m_w)

            rows, cols = np.random.randint(0, 100, 2)
            random_matrix_with_padding = add_padding(random_matrix, (rows, cols))
            
            self.assertEqual(random_matrix_with_padding.shape, (m_h + rows*2, m_w + cols*2))


    def test_random_case(self, N: int = 1000):
        for _ in range(N): 
            d = np.random.randint(1, 100, 2)
            k = np.random.choice([1, 3, 5, 7, 9, 10], 2) # `10` is to check oddness assertion
            random_matrix = np.random.rand(*d)
            random_kernel = np.random.rand(*k)
            for __ in range(N):
                stride = np.random.randint(0, 5, 2) # `0` is to check parameters assertion
                dilation = np.random.randint(0, 5, 2) # `0` is to check parameters assertion
                padding = np.random.randint(-1, 5, 2) # `-1` is to check parameters assertion
                try: # `try` in case of division by zero when stride[0] or stride[1] equal to zero
                    h_out = np.floor((d[0] + 2 * padding[0] - k[0] - (k[0] - 1) * (dilation[0] - 1)) / stride[0]).astype(int) + 1
                    w_out = np.floor((d[1] + 2 * padding[1] - k[1] - (k[1] - 1) * (dilation[1] - 1)) / stride[1]).astype(int) + 1
                except:
                    h_out, w_out = None, None
                # print(f'Matr: {d} | Kern: {k} | Stri: {stride} | Dila: {dilation} | Padd: {padding} | OutD: {h_out, w_out}') # for debugging
                
                if (stride[0] < 1 or stride[1] < 1 or dilation[0] < 1 or dilation[1] < 1 or padding[0] < 0 or padding[1] < 0 or
                    not isinstance(stride[0], int) or not isinstance(stride[1], int) or not isinstance(dilation[0], int) or 
                    not isinstance(dilation[1], int) or not isinstance(padding[0], int) or not isinstance(padding[1], int)):
                    with self.assertRaises(AssertionError):
                        matrix_conved = conv2d(random_matrix, random_kernel, stride=stride, dilation=dilation, padding=padding)
                elif k[0] % 2 != 1 or k[1] % 2 != 1:
                    with self.assertRaises(AssertionError):
                        matrix_conved = conv2d(random_matrix, random_kernel, stride=stride, dilation=dilation, padding=padding)
                elif d[0] < k[0] or d[1] < k[1]:
                    with self.assertRaises(AssertionError):
                        matrix_conved = conv2d(random_matrix, random_kernel, stride=stride, dilation=dilation, padding=padding)
                elif h_out <= 0 or w_out <= 0:
                    with self.assertRaises(AssertionError):
                        matrix_conved = conv2d(random_matrix, random_kernel, stride=stride, dilation=dilation, padding=padding)
                else:
                    matrix_conved = conv2d(random_matrix, random_kernel, stride=stride, dilation=dilation, padding=padding)
                    self.assertEqual(matrix_conved.shape, (h_out, w_out))


    def test_kernel_3x3_easy(self):
        matrix = np.array([[0, 4, 3, 2, 0, 1, 0], 
                         [4, 3, 0, 1, 0, 1, 0],
                         [1, 3, 4, 2, 0, 1, 0],
                         [3, 4, 2, 2, 0, 1, 0],
                         [0, 0, 0, 0, 0, 1, 0]])

        kernel = np.array([[1, 1, 3], 
                           [0, 2, 3],
                           [3, 3, 3]])

        # stride = 1, dilation = 1, padding = 0
        result_110 = conv2d(matrix, kernel)
        answer_110 = np.array([[43, 43, 25, 17, 6], 
                               [52, 44, 17, 16, 6],
                               [30, 23, 10, 11, 6]])
        
        # stride = 1, dilation = 1, padding = 1
        result_111 = conv2d(matrix, kernel, padding=(1, 1))
        answer_111 = np.array([[33, 38, 24,  7,  9,  5,  3],
                               [41, 43, 43, 25, 17,  6,  4],
                               [45, 52, 44, 17, 16,  6,  4],
                               [28, 30, 23, 10, 11,  6,  4],
                               [15, 13, 12,  4,  8,  3,  1]])
        
        # stride = 1, dilation = 2, padding = 0
        result_120 = conv2d(matrix, kernel, dilation=(2, 2))
        answer_120 = np.array([[11, 19, 3]])

        # stride = 1, dilation = 2, padding = 1
        result_121 = conv2d(matrix, kernel, dilation=(2, 2), padding=(1, 1))
        answer_121 = np.array([[27, 15, 26,  6, 11],
                               [22, 11, 19,  3,  8],
                               [20,  8, 14,  0,  4]])
        
        # stride = 2, dilation = 1, padding = 0
        result_210 = conv2d(matrix, kernel, stride=(2, 2))
        answer_210 = np.array([[43, 25, 6], 
                               [30, 10, 6]])
        
        # stride = 2, dilation = 1, padding = 1
        result_211 = conv2d(matrix, kernel, stride=(2, 2), padding=(1, 1))
        answer_211 = np.array([[33, 24,  9,  3],
                               [45, 44, 16,  4],
                               [15, 12,  8,  1]])
        
        # stride = 2, dilation = 2, padding = 0
        result_220 = conv2d(matrix, kernel, stride=(2, 2), dilation=(2, 2))
        answer_220 = np.array([[11, 3]])

        # stride = 2, dilation = 2, padding = 1
        result_221 = conv2d(matrix, kernel, stride=(2, 2), dilation=(2, 2), padding=(1, 1))
        answer_221 = np.array([[27, 26, 11], 
                               [20, 14,  4]])

        self.assertEqual(result_110.tolist(), answer_110.tolist())
        self.assertEqual(result_111.tolist(), answer_111.tolist())
        self.assertEqual(result_120.tolist(), answer_120.tolist())
        self.assertEqual(result_121.tolist(), answer_121.tolist())
        self.assertEqual(result_210.tolist(), answer_210.tolist())
        self.assertEqual(result_211.tolist(), answer_211.tolist())
        self.assertEqual(result_220.tolist(), answer_220.tolist())
        self.assertEqual(result_221.tolist(), answer_221.tolist())


    def test_kernel_5x5_difficult(self):
        matrix = np.array([[1, 4, 4, 2, 1, 0, 0, 1, 0, 0, 3, 3, 3, 4], 
                           [0, 2, 0, 2, 0, 3, 4, 4, 2, 1, 1, 3, 0, 4],
                           [1, 1, 0, 0, 3, 4, 2, 4, 4, 2, 3, 0, 0, 4],
                           [4, 0, 1, 2, 0, 2, 0, 3, 3, 3, 0, 4, 1, 0],
                           [3, 0, 0, 3, 3, 3, 2, 0, 2, 1, 1, 0, 4, 2],
                           [2, 4, 3, 1, 1, 0, 2, 1, 3, 4, 4, 0, 2, 3],
                           [2, 4, 3, 3, 2, 1, 4, 0, 3, 4, 1, 2, 0, 0],
                           [2, 1, 0, 1, 1, 2, 2, 3, 0, 0, 1, 2, 4, 2],
                           [3, 3, 1, 1, 1, 1, 4, 4, 2, 3, 2, 2, 2, 3]])

        kernel = np.array([[2, 0, 2, 2, 2], 
                           [2, 3, 1, 1, 3], 
                           [3, 1, 1, 3, 1], 
                           [2, 2, 3, 1, 1],
                           [0, 0, 1, 0, 0]])

        # default params
        result_11_11_00 = conv2d(matrix, kernel)
        answer_11_11_00 = np.array([[44., 58., 59., 62., 70., 80., 75., 92., 64., 72.],
                                    [52., 52., 59., 87., 92., 83., 77., 74., 71., 67.],
                                    [66., 63., 60., 64., 76., 79., 75., 82., 77., 64.],
                                    [75., 69., 64., 64., 69., 75., 70., 71., 75., 74.],
                                    [74., 71., 63., 66., 61., 75., 79., 47., 73., 76.]])

        # only stride: (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (4, 6)
        result_12_11_00 = conv2d(matrix, kernel, stride=(1, 2))
        answer_12_11_00 = np.array([[44., 59., 70., 75., 64.],
                                    [52., 59., 92., 77., 71.],
                                    [66., 60., 76., 75., 77.],
                                    [75., 64., 69., 70., 75.],
                                    [74., 63., 61., 79., 73.]])

        result_13_11_00 = conv2d(matrix, kernel, stride=(1, 3))
        answer_13_11_00 = np.array([[44., 62., 75., 72.],
                                    [52., 87., 77., 67.],
                                    [66., 64., 75., 64.],
                                    [75., 64., 70., 74.],
                                    [74., 66., 79., 76.]])

        result_21_11_00 = conv2d(matrix, kernel, stride=(2, 1))
        answer_21_11_00 = np.array([[44., 58., 59., 62., 70., 80., 75., 92., 64., 72.],
                                    [66., 63., 60., 64., 76., 79., 75., 82., 77., 64.],
                                    [74., 71., 63., 66., 61., 75., 79., 47., 73., 76.]])

        result_22_11_00 = conv2d(matrix, kernel, stride=(2, 2))
        answer_22_11_00 = np.array([[44., 59., 70., 75., 64],
                                    [66., 60., 76., 75., 77],
                                    [74., 63., 61., 79., 73]])

        result_23_11_00 = conv2d(matrix, kernel, stride=(2, 3))
        answer_23_11_00 = np.array([[44., 62., 75., 72.],
                                    [66., 64., 75., 64.],
                                    [74., 66., 79., 76.]])

        result_31_11_00 = conv2d(matrix, kernel, stride=(3, 1))
        answer_31_11_00 = np.array([[44., 58., 59., 62., 70., 80., 75., 92., 64., 72.],
                                    [75., 69., 64., 64., 69., 75., 70., 71., 75., 74.]])

        result_32_11_00 = conv2d(matrix, kernel, stride=(3, 2))
        answer_32_11_00 = np.array([[44., 59., 70., 75., 64.],
                                    [75., 64., 69., 70., 75.]])

        result_46_11_00 = conv2d(matrix, kernel, stride=(4, 6))
        answer_46_11_00 = np.array([[44., 75.],
                                    [74., 79.]])

        # only dilation: (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)
        result_11_12_00 = conv2d(matrix, kernel, dilation=(1, 2))
        answer_11_12_00 = np.array([[46., 70., 50., 77., 65., 94.],
                                    [67., 68., 67., 76., 53., 95.],
                                    [80., 65., 60., 64., 70., 73.],
                                    [74., 74., 77., 73., 79., 55.],
                                    [81., 66., 74., 60., 70., 58.]])

        result_11_13_00 = conv2d(matrix, kernel, dilation=(1, 3))
        answer_11_13_00 = np.array([[48., 77.],
                                    [65., 65.],
                                    [73., 55.],
                                    [97., 67.],
                                    [84., 68.]])
        result_11_21_00 = conv2d(matrix, kernel, dilation=(2, 1))
        answer_11_21_00 = np.array([[78., 73., 64., 72., 81., 69., 73., 69., 68., 81.]])

        result_11_22_00 = conv2d(matrix, kernel, dilation=(2, 2))
        answer_11_22_00 = np.array([[67., 55., 80., 63., 77., 79.]])

        result_11_23_00 = conv2d(matrix, kernel, dilation=(2, 3))
        answer_11_23_00 = np.array([[65., 79.]])

        # only paddings: (0, 1), (1, 0), (1, 1)
        result_11_11_01 = conv2d(matrix, kernel, padding=(0, 1))
        answer_11_11_01 = np.array([[41., 44., 58., 59., 62., 70., 80., 75., 92., 64., 72., 71.],
                                    [34., 52., 52., 59., 87., 92., 83., 77., 74., 71., 67., 43.],
                                    [51., 66., 63., 60., 64., 76., 79., 75., 82., 77., 64., 57.],
                                    [63., 75., 69., 64., 64., 69., 75., 70., 71., 75., 74., 43.],
                                    [51., 74., 71., 63., 66., 61., 75., 79., 47., 73., 76., 54.]])

        result_11_11_10 = conv2d(matrix, kernel, padding=(1, 0))
        answer_11_11_10 = np.array([[39., 45., 45., 61., 52., 58., 66., 63., 53., 56.],
                                    [44., 58., 59., 62., 70., 80., 75., 92., 64., 72.],
                                    [52., 52., 59., 87., 92., 83., 77., 74., 71., 67.],
                                    [66., 63., 60., 64., 76., 79., 75., 82., 77., 64.],
                                    [75., 69., 64., 64., 69., 75., 70., 71., 75., 74.],
                                    [74., 71., 63., 66., 61., 75., 79., 47., 73., 76.],
                                    [70., 59., 64., 55., 72., 83., 81., 77., 70., 69.]])

        result_11_11_11 = conv2d(matrix, kernel, padding=(1, 1))
        answer_11_11_11 = np.array([[26., 39., 45., 45., 61., 52., 58., 66., 63., 53., 56., 51.],
                                    [41., 44., 58., 59., 62., 70., 80., 75., 92., 64., 72., 71.],
                                    [34., 52., 52., 59., 87., 92., 83., 77., 74., 71., 67., 43.],
                                    [51., 66., 63., 60., 64., 76., 79., 75., 82., 77., 64., 57.],
                                    [63., 75., 69., 64., 64., 69., 75., 70., 71., 75., 74., 43.],
                                    [51., 74., 71., 63., 66., 61., 75., 79., 47., 73., 76., 54.],
                                    [59., 70., 59., 64., 55., 72., 83., 81., 77., 70., 69., 58.]])

        # different sets of parameters
        result_21_13_00 = conv2d(matrix, kernel, stride=(2, 1), dilation=(1, 3), padding=(0, 0))
        answer_21_13_00 = np.array([[48., 77.],
                                    [73., 55.],
                                    [84., 68.]])

        result_23_13_13 = conv2d(matrix, kernel, stride=(2, 3), dilation=(1, 3), padding=(1, 3))
        answer_23_13_13 = np.array([[28., 36., 31.],
                                    [53., 65., 47.],
                                    [62., 97., 70.],
                                    [64., 79., 74.]])

        result_32_23_22 = conv2d(matrix, kernel, stride=(3, 2), dilation=(2, 3), padding=(2, 2))
        answer_32_23_22 = np.array([[54., 55., 34.],
                                    [34., 69., 43.]])
        
        # default params
        self.assertEqual(result_11_11_00.tolist(), answer_11_11_00.tolist())
        # only stride: (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (4, 6)
        self.assertEqual(result_12_11_00.tolist(), answer_12_11_00.tolist())
        self.assertEqual(result_13_11_00.tolist(), answer_13_11_00.tolist())
        self.assertEqual(result_21_11_00.tolist(), answer_21_11_00.tolist())
        self.assertEqual(result_22_11_00.tolist(), answer_22_11_00.tolist())
        self.assertEqual(result_23_11_00.tolist(), answer_23_11_00.tolist())
        self.assertEqual(result_31_11_00.tolist(), answer_31_11_00.tolist())
        self.assertEqual(result_32_11_00.tolist(), answer_32_11_00.tolist())
        self.assertEqual(result_46_11_00.tolist(), answer_46_11_00.tolist())
        # only dilation: (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)
        self.assertEqual(result_11_12_00.tolist(), answer_11_12_00.tolist())
        self.assertEqual(result_11_13_00.tolist(), answer_11_13_00.tolist())
        self.assertEqual(result_11_21_00.tolist(), answer_11_21_00.tolist())
        self.assertEqual(result_11_22_00.tolist(), answer_11_22_00.tolist())
        self.assertEqual(result_11_23_00.tolist(), answer_11_23_00.tolist())
        # only paddings: (0, 1), (1, 0), (1, 1)
        self.assertEqual(result_11_11_01.tolist(), answer_11_11_01.tolist())
        self.assertEqual(result_11_11_10.tolist(), answer_11_11_10.tolist())
        self.assertEqual(result_11_11_11.tolist(), answer_11_11_11.tolist())
        # different sets of parameters
        self.assertEqual(result_21_13_00.tolist(), answer_21_13_00.tolist())
        self.assertEqual(result_23_13_13.tolist(), answer_23_13_13.tolist())
        self.assertEqual(result_32_23_22.tolist(), answer_32_23_22.tolist())


    def test_kernel_5x3_difficult(self):
        matrix = np.array([[0, 4, 3, 2, 0, 1, 0], 
                           [4, 3, 0, 1, 0, 1, 0],
                           [1, 3, 4, 2, 0, 1, 0],
                           [3, 4, 2, 2, 0, 1, 0],
                           [0, 0, 0, 0, 0, 1, 0]])

        kernel = np.array([[1, 1, 3], 
                           [0, 2, 3],
                           [3, 3, 3],
                           [0, 2, 1],
                           [3, 3, 0]])
        
        # default params
        result_11_11_00 = conv2d(matrix, kernel, stride=(1, 1), dilation=(1, 1), padding=(0, 0))
        answer_11_11_00 = np.array([[53., 49., 29., 18., 11.]])

        # different sets of parameters
        result_21_13_00 = conv2d(matrix, kernel, stride=(2, 1), dilation=(1, 3), padding=(0, 0))
        answer_21_13_00 = np.array([[17.]])

        result_23_13_13 = conv2d(matrix, kernel, stride=(2, 3), dilation=(1, 3), padding=(1, 3))
        answer_23_13_13 = np.array([[34., 38.,  9.],
                                    [30., 24.,  7.]])

        result_32_23_42 = conv2d(matrix, kernel, stride=(3, 2), dilation=(2, 3), padding=(4, 2))
        answer_32_23_42 = np.array([[18., 10., 17.],
                                    [18., 17., 11.]])

        result_21_12_04 = conv2d(matrix, kernel, stride=(2, 1), dilation=(1, 2), padding=(0, 4))
        answer_21_12_04 = np.array([[18., 34., 40., 44., 22., 37., 15., 19.,  0.,  7.,  0.]])

        result_22_12_04 = conv2d(matrix, kernel, stride=(2, 2), dilation=(1, 2), padding=(0, 4))
        answer_22_12_04 = np.array([[18., 40., 22., 15.,  0.,  0.]])

        result_23_13_25 = conv2d(matrix, kernel, stride=(2, 3), dilation=(1, 3), padding=(2, 5))
        answer_23_13_25 = np.array([[15., 27., 21.,  0.],
                                    [34., 27., 13.,  0.],
                                    [21., 11.,  3.,  0.]])

        result_11_11_33 = conv2d(matrix, kernel, stride=(1, 1), dilation=(1, 1), padding=(3, 3))
        answer_11_11_33 = np.array([[ 0.,  0., 16., 32., 17.,  7.,  4.,  5.,  3.,  0.,  0.],
                                    [ 0.,  4., 26., 39., 49., 35., 16.,  8.,  6.,  0.,  0.],
                                    [ 0., 13., 47., 69., 52., 23., 16., 10.,  6.,  0.,  0.],
                                    [ 0., 18., 51., 53., 49., 29., 18., 11.,  7.,  0.,  0.],
                                    [ 0., 24., 45., 52., 44., 17., 17.,  8.,  4.,  0.,  0.],
                                    [ 0., 12., 28., 30., 23., 10., 11.,  6.,  4.,  0.,  0.],
                                    [ 0.,  9., 15., 13., 12.,  4.,  8.,  3.,  1.,  0.,  0.]])

        # default params
        self.assertEqual(result_11_11_00.tolist(), answer_11_11_00.tolist())
        # different sets of parameters
        self.assertEqual(result_21_13_00.tolist(), answer_21_13_00.tolist())
        self.assertEqual(result_23_13_13.tolist(), answer_23_13_13.tolist())
        self.assertEqual(result_32_23_42.tolist(), answer_32_23_42.tolist())
        self.assertEqual(result_21_12_04.tolist(), answer_21_12_04.tolist())
        self.assertEqual(result_22_12_04.tolist(), answer_22_12_04.tolist())
        self.assertEqual(result_23_13_25.tolist(), answer_23_13_25.tolist())
        self.assertEqual(result_11_11_33.tolist(), answer_11_11_33.tolist())

if __name__ == '__main__':
    unittest.main()
