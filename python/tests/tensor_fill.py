# (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

from magnetron import *

def test_tensor_fill_zero():
    tensor = Tensor.zeros((1, 2, 3, 4, 5, 6))
    data = tensor.to_list()
    assert len(data) == 1 * 2 * 3 * 4 * 5 * 6
    assert all([x == 0 for x in data])

def test_tensor_fill_x():
    tensor = Tensor.full((1, 2, 3, 4, 5, 6), fill_value=-22)
    data = tensor.to_list()
    assert len(data) == 1 * 2 * 3 * 4 * 5 * 6
    assert all([x == -22 for x in data])

