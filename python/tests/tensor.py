# (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

from magnetron import *

def test_tensor_creation():
    tensor = Tensor.empty((1, 2, 3, 4, 5, 6))
    assert tensor.shape == (1, 2, 3, 4, 5, 6)
    assert tensor.numel == (1 * 2 * 3 * 4 * 5 * 6)
    assert tensor.data_size == 4 * (1 * 2 * 3 * 4 * 5 * 6)
    assert tensor.data_ptr != 0
    assert tensor.is_permuted is False
    assert tensor.is_contiguous is True
    assert tensor.is_transposed is False
    assert tensor.num_rows == (2 * 3 * 4 * 5 * 6)
    assert tensor.num_cols == 1
    assert tensor.dtype == DType.F32
    assert tensor.is_scalar is False
    assert tensor.is_matrix is False
    assert tensor.is_vector is False
    assert tensor.is_volume is False

def test_tensor_scalar_get_set_physical():
    tensor = Tensor.empty((4, 4))
    tensor[0, 0] = 128
    assert tensor[0, 0] == 128
    tensor[3, 3] = 3.14
    assert abs(tensor[3, 3] - 3.14) < 1e-6

def test_tensor_scalar_get_set_virtual():
    tensor = Tensor.empty((4, 4))
    tensor[0] = 128
    assert tensor[0] == 128
    tensor[15] = 3.14
    assert abs(tensor[15] - 3.14) < 1e-6

def test_tensor_to_list():
    tensor = Tensor.zeros((2, 2))
    tensor[0] = 128
    tensor[1] = 255
    tensor[2] = -22333
    tensor[3] = 22
    assert tensor.to_list() == [128, 255, -22333, 22]
