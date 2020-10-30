from math import floor
import numpy as np
from typing import List, Tuple


def add_padding(data: List[List[float]], 
                padding: Tuple[int, int]) -> List[List[float]]:    
    n, m = data.shape
    add_rows, add_cols = padding
    
    data_with_padding = np.zeros((n + add_rows * 2, m + add_cols * 2))
    data_with_padding[add_rows:n + add_rows, add_cols:m + add_cols] = data
    
    return data_with_padding


def check_params(data, kernel, stride, dilation, padding):
    params_are_correct = (stride[0]   >= 1 and stride[1]   >= 1 and 
                          dilation[0] >= 1 and dilation[1] >= 1 and
                          padding[0]  >= 0 and padding[1]  >= 0 and
                          isinstance(stride[0], int)   and isinstance(stride[1], int)   and
                          isinstance(dilation[0], int) and isinstance(dilation[1], int) and
                          isinstance(padding[0], int)  and isinstance(padding[1], int))
    assert params_are_correct, 'Parameters should be integers equal or greater than default values.'
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    n, m = data.shape
    data = data if list(padding) == [0, 0] else add_padding(data, padding)
    n_p, m_p = data.shape

    if not isinstance(kernel, np.ndarray):
        kernel = np.array(kernel)
    k = kernel.shape
    
    kernel_is_correct = k[0] % 2 == 1 and k[1] % 2 == 1
    assert kernel_is_correct, 'Kernel shape should be odd.'
    data_to_kernel_is_correct = n_p >= k[0] and m_p >= k[1]
    assert data_to_kernel_is_correct, 'Kernel can\'t be bigger than data in terms of shape.'
    
    h_out = floor((n + 2 * padding[0] - k[0] - (k[0] - 1) * (dilation[0] - 1)) / stride[0]) + 1
    w_out = floor((m + 2 * padding[1] - k[1] - (k[1] - 1) * (dilation[1] - 1)) / stride[1]) + 1
    out_dimensions_are_correct = h_out > 0 and w_out > 0
    assert out_dimensions_are_correct, 'Can\'t apply input parameters, one of resulting output dimension is non-positive.'

    return data, kernel, k, h_out, w_out


def convolve(data: List[List[float]], kernel: List[List[float]], 
             stride: Tuple[int, int] = (1, 1), 
             dilation: Tuple[int, int] = (1, 1), 
             padding: Tuple[int, int] = (0, 0)) -> List[List[float]]:
    data, kernel, k, h_out, w_out = check_params(data, kernel, stride, dilation, padding)
    data_out = np.zeros((h_out, w_out))
    
    b = k[0] // 2, k[1] // 2
    center_x_0 = b[0] * dilation[0]
    center_y_0 = b[1] * dilation[1]
    for i in range(h_out):
        center_x = center_x_0 + i*stride[0]
        indices_x = [center_x + l*dilation[0] for l in range(-b[0], b[0] + 1)]
        for j in range(w_out):
            center_y = center_y_0 + j*stride[1]
            indices_y = [center_y + l*dilation[1] for l in range(-b[1], b[1] + 1)]

            data_submatrix = data[indices_x, :][:, indices_y]
            prod = np.multiply(data_submatrix, kernel)

            data_out[i][j] = np.sum(prod)
    return data_out
