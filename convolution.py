from math import floor
import numpy as np
from typing import List, Tuple


def add_padding(matrix: List[List[float]], 
                padding: Tuple[int, int]) -> List[List[float]]:    
    n, m = matrix.shape
    add_rows, add_cols = padding
    
    matrix_with_padding = np.zeros((n + add_rows * 2, m + add_cols * 2))
    matrix_with_padding[add_rows:n + add_rows, add_cols:m + add_cols] = matrix
    
    return matrix_with_padding


def check_params(matrix, kernel, stride, dilation, padding):
    params_are_correct = (stride[0]   >= 1 and stride[1]   >= 1 and 
                          dilation[0] >= 1 and dilation[1] >= 1 and
                          padding[0]  >= 0 and padding[1]  >= 0 and
                          isinstance(stride[0], int)   and isinstance(stride[1], int)   and
                          isinstance(dilation[0], int) and isinstance(dilation[1], int) and
                          isinstance(padding[0], int)  and isinstance(padding[1], int))
    assert params_are_correct, 'Parameters should be integers equal or greater than default values.'
    if not isinstance(matrix, np.ndarray):
        matrix = np.array(matrix)
    n, m = matrix.shape
    matrix = matrix if list(padding) == [0, 0] else add_padding(matrix, padding)
    n_p, m_p = matrix.shape

    if not isinstance(kernel, np.ndarray):
        kernel = np.array(kernel)
    k = kernel.shape
    
    kernel_is_correct = k[0] % 2 == 1 and k[1] % 2 == 1
    assert kernel_is_correct, 'Kernel shape should be odd.'
    matrix_to_kernel_is_correct = n_p >= k[0] and m_p >= k[1]
    assert matrix_to_kernel_is_correct, 'Kernel can\'t be bigger than matrix in terms of shape.'
    
    h_out = floor((n + 2 * padding[0] - k[0] - (k[0] - 1) * (dilation[0] - 1)) / stride[0]) + 1
    w_out = floor((m + 2 * padding[1] - k[1] - (k[1] - 1) * (dilation[1] - 1)) / stride[1]) + 1
    out_dimensions_are_correct = h_out > 0 and w_out > 0
    assert out_dimensions_are_correct, 'Can\'t apply input parameters, one of resulting output dimension is non-positive.'

    return matrix, kernel, k, h_out, w_out


def convolve(matrix: List[List[float]], kernel: List[List[float]], 
             stride: Tuple[int, int] = (1, 1), 
             dilation: Tuple[int, int] = (1, 1), 
             padding: Tuple[int, int] = (0, 0)) -> List[List[float]]:
    matrix, kernel, k, h_out, w_out = check_params(matrix, kernel, stride, dilation, padding)
    matrix_out = np.zeros((h_out, w_out))
    
    b = k[0] // 2, k[1] // 2
    center_x_0 = b[0] * dilation[0]
    center_y_0 = b[1] * dilation[1]
    for i in range(h_out):
        center_x = center_x_0 + i*stride[0]
        indices_x = [center_x + l*dilation[0] for l in range(-b[0], b[0] + 1)]
        for j in range(w_out):
            center_y = center_y_0 + j*stride[1]
            indices_y = [center_y + l*dilation[1] for l in range(-b[1], b[1] + 1)]

            matrix_submatrix = matrix[indices_x, :][:, indices_y]
            prod = np.multiply(matrix_submatrix, kernel)

            matrix_out[i][j] = np.sum(prod)
    return matrix_out


def apply_filter_to_image(image: np.ndarray, 
                          kernel: List[List[float]]) -> np.ndarray:
    kernel = np.asarray(kernel)
    b = kernel.shape[0], kernel.shape[1]
    return np.dstack([convolve(image[:, :, z], kernel, padding=(b[0]//2,  b[1]//2)) 
                      for z in range(3)]).astype('uint8')
