import numpy as np
from typing import List, Tuple


def add_padding(data: List[List[float]], 
                padding: Tuple[int, int]) -> List[List[float]]:
    assert padding[0] >= 0 and padding[1] >= 0, 'Padding can not be lower than 0'
    
    if not isinstance(data, np.ndarray):
        input_data = np.array(data)

    n, m = data.shape
    add_rows, add_cols = padding
    
    data_with_padding = np.zeros((n + add_rows * 2, m + add_cols * 2))
    data_with_padding[add_rows:n + add_rows, add_cols:m + add_cols] = data
    
    return data_with_padding


def convolve(input_data: List[List[float]], kernel: List[List[float]], 
             stride: Tuple[int, int] = (1, 1), 
             dilation: Tuple[int, int] = (1, 1), 
             padding: Tuple[int, int] = (0, 0)) -> List[List[float]]:
    if not isinstance(input_data, np.ndarray):
        input_data = np.array(input_data)
    if not isinstance(kernel, np.ndarray):
        kernel = np.array(kernel)

    data = input_data.copy() if padding == (0, 0) else add_padding(input_data, padding)
    n, m = data.shape
    k = kernel.shape

    h_out = int((n + 2 * padding[0] - k[0] - (k[0] - 1) * (dilation[0] - 1)) / stride[0]) + 1
    w_out = int((m + 2 * padding[1] - k[1] - (k[1] - 1) * (dilation[1] - 1)) / stride[1]) + 1
    assert h_out > 0 and w_out > 0, 'Can\'t apply input parameters'
    
    data_out = np.zeros((h_out, w_out))

    for i in range(h_out):
        for j in range(w_out):
            center_x = dilation[0] + i*stride[0]
            indices_x = ([center_x - (k+1)*dilation[0] for k in range(k[0]//2)] + 
                         [center_x] + [center_x + (k+1)*dilation[0] for k in range(k[0]//2)])
            center_y = dilation[1] + j*stride[1]
            indices_y = ([center_y - (k+1)*dilation[1] for k in range(k[1]//2)] + 
                         [center_y] + [center_y + (k+1)*dilation[1] for k in range(k[1]//2)])
            
            data_submatrix = data[indices_x, :][:, indices_y]
            prod = np.multiply(data_submatrix, kernel)
            data_out[i][j] = np.sum(prod)
    return data_out