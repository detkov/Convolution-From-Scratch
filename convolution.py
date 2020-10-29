import numpy as np
from typing import List, Tuple
from math import floor


def add_padding(data: List[List[float]], 
                padding: Tuple[int, int]) -> List[List[float]]:
    assert padding[0] >= 0 and padding[1] >= 0, 'Padding value can not be less than 0.'
    
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    n, m = data.shape
    add_rows, add_cols = padding
    
    data_with_padding = np.zeros((n + add_rows * 2, m + add_cols * 2))
    data_with_padding[add_rows:n + add_rows, add_cols:m + add_cols] = data
    
    return data_with_padding


def convolve(data: List[List[float]], kernel: List[List[float]], 
             stride: Tuple[int, int] = (1, 1), 
             dilation: Tuple[int, int] = (1, 1), 
             padding: Tuple[int, int] = (0, 0)) -> List[List[float]]:
    assert (stride[0] >= 1 and stride[1] >= 1 and 
            dilation[0] >= 1 and dilation[1] >= 1 and
            padding[0] >= 0 and padding[1] >= 0), 'Parameters should be equal or greater than default values.' 
    assert (data.shape[0] >= kernel.shape[0] and 
            data.shape[1] >= kernel.shape[1]), 'Kernel can\'t be bigger than data in terms of shape.'
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    n, m = data.shape
    data = data if list(padding) == [0, 0] else add_padding(data, padding)
    
    if not isinstance(kernel, np.ndarray):
        kernel = np.array(kernel)
    assert (kernel.shape[0] == kernel.shape[1]), 'Kernel must be squared.'
    k = kernel.shape
    b = k[0]//2, k[1]//2

    h_out = floor((n + 2 * padding[0] - k[0] - (k[0] - 1) * (dilation[0] - 1)) / stride[0]) + 1
    w_out = floor((m + 2 * padding[1] - k[1] - (k[1] - 1) * (dilation[1] - 1)) / stride[1]) + 1
    assert h_out > 0 and w_out > 0, 'Can\'t apply input parameters, one of resulting output dimension is non-positive.'


    data_out = np.zeros((h_out, w_out))

    for i in range(h_out):
        for j in range(w_out):
            center_x = dilation[0] + i*stride[0]
            center_y = dilation[1] + j*stride[1]
            
            indices_x = [center_x + l*dilation[0] for l in range(-b[0], b[0] + 1)]
            indices_y = [center_y + l*dilation[1] for l in range(-b[1], b[1] + 1)]
                        
            data_submatrix = data[indices_x, :][:, indices_y]
            prod = np.multiply(data_submatrix, kernel)
            data_out[i][j] = np.sum(prod)
    return data_out