# (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

from magnetron import *
import numpy as np

def tonumpy(t: Tensor):
    return np.array(t.tolist(), dtype=np.float32).reshape(t.shape)

def test_matmul_squared():
    shapes = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    for shape in shapes:
        mag_a = Tensor.uniform((shape, shape))
        mag_b = Tensor.uniform((shape, shape))
        np_a = tonumpy(mag_a)
        np_b = tonumpy(mag_b)
        mag_result = mag_a @ mag_b
        np_result = np.matmul(np_a, np_b)
        assert mag_result.shape == np_result.shape
        assert np.allclose(tonumpy(mag_result), np_result)

def test_matmul():
    shapes = [(4, 8), (8, 16), (16, 32), (32, 64), (64, 128), (128, 256), (256, 512), (512, 1024)]
    for shape in shapes:
        mag_a = Tensor.uniform(shape)
        mag_b = Tensor.uniform((shape[1], shape[0]))
        np_a = tonumpy(mag_a)
        np_b = tonumpy(mag_b)
        mag_result = mag_a @ mag_b
        np_result = np.matmul(np_a, np_b)
        assert mag_result.shape == np_result.shape
        assert np.allclose(tonumpy(mag_result), np_result)

