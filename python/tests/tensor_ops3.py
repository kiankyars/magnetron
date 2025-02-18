# (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

from magnetron import *
import numpy as np

EPS = 1e-4


def tonumpy(t: Tensor) -> np.array:
    return np.array(t.tolist(), dtype=np.float32).reshape(t.shape)


def sigmoid(x: np.array) -> None:
    return 1 / (1 + np.exp(-x))


def test_simple_ff() -> None:
    truth_table = [[0, 0], [0, 1], [1, 0], [1, 1]]

    W1 = Tensor.uniform((2, 4))
    b1 = Tensor.uniform((1, 4))
    W2 = Tensor.uniform((4, 1))
    b2 = Tensor.uniform((1, 1))

    nW1 = tonumpy(W1)
    nb1 = tonumpy(b1)
    nW2 = tonumpy(W2)
    nb2 = tonumpy(b2)

    np_data = []
    for x in truth_table:
        z1 = x @ nW1 + nb1
        a1 = sigmoid(z1)
        z2 = a1 @ nW2 + nb2
        a2 = sigmoid(z2)
        np_data.append(a2)

    mag_data = []
    for x in truth_table:
        x = Tensor.const([x])
        z1 = x @ W1 + b1
        a1 = z1.sigmoid()
        z2 = a1 @ W2 + b2
        a2 = z2.sigmoid()
        mag_data.append(a2)

    for mag, np_ in zip(mag_data, np_data):
        np.testing.assert_allclose(tonumpy(mag), np_)


def test_matmul_squared() -> None:
    shapes = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    for shape in shapes:
        mag_a = Tensor.uniform((shape, shape))
        mag_b = Tensor.uniform((shape, shape))
        np_a = tonumpy(mag_a)
        np_b = tonumpy(mag_b)
        mag_result = mag_a @ mag_b
        np_result = np.matmul(np_a, np_b)
        assert mag_result.shape == np_result.shape
        np.testing.assert_allclose(tonumpy(mag_result), np_result, atol=EPS)


def test_matmul() -> None:
    shapes = [
        (4, 8),
        (8, 16),
        (16, 32),
        (32, 64),
        (64, 128),
        (128, 256),
        (256, 512),
        (512, 1024),
    ]
    for shape in shapes:
        mag_a = Tensor.uniform(shape)
        mag_b = Tensor.uniform((shape[1], shape[0]))
        np_a = tonumpy(mag_a)
        np_b = tonumpy(mag_b)
        mag_result = mag_a @ mag_b
        np_result = np.matmul(np_a, np_b)
        assert mag_result.shape == np_result.shape
        np.testing.assert_allclose(tonumpy(mag_result), np_result, atol=EPS)


def test_matmul_matrix_by_vector() -> None:
    shapes = [
        (4, 8),
        (8, 16),
        (16, 32),
        (32, 64),
        (64, 128),
        (128, 256),
        (256, 512),
        (512, 1024),
    ]
    for shape in shapes:
        mag_a = Tensor.uniform(shape)
        mag_b = Tensor.uniform((shape[1], 1))
        np_a = tonumpy(mag_a)
        np_b = tonumpy(mag_b)
        mag_result = mag_a @ mag_b
        np_result = np.matmul(np_a, np_b)
        assert mag_result.shape == np_result.shape
        np.testing.assert_allclose(tonumpy(mag_result), np_result, atol=EPS)


def test_matmul_vector_by_matrix() -> None:
    shapes = [
        (4, 8),
        (8, 16),
        (16, 32),
        (32, 64),
        (64, 128),
        (128, 256),
        (256, 512),
        (512, 1024),
    ]
    for shape in shapes:
        mag_a = Tensor.uniform((1, shape[0]))
        mag_b = Tensor.uniform(shape)
        np_a = tonumpy(mag_a)
        np_b = tonumpy(mag_b)
        mag_result = mag_a @ mag_b
        np_result = np.matmul(np_a, np_b)
        assert mag_result.shape == np_result.shape
        np.testing.assert_allclose(tonumpy(mag_result), np_result, atol=EPS)


def test_matmul_scalar_by_matrix() -> None:
    shapes = [
        (4, 8),
        (8, 16),
        (16, 32),
        (32, 64),
        (64, 128),
        (128, 256),
        (256, 512),
        (512, 1024),
    ]
    for shape in shapes:
        scalar = np.random.rand()
        mag_b = Tensor.uniform(shape)
        np_b = tonumpy(mag_b)
        mag_result = scalar * mag_b
        np_result = scalar * np_b
        assert mag_result.shape == np_result.shape
        np.testing.assert_allclose(tonumpy(mag_result), np_result, atol=EPS)


"""
def test_matmul_x_transposed() -> None:
    shape_a = (4, 2)
    shape_b = (4, 4)
    mag_a = Tensor.uniform(shape_a)
    mag_b = Tensor.uniform(shape_b)
    np_a = tonumpy(mag_a)
    np_b = tonumpy(mag_b)
    mag_result = mag_a.T.clone() @ mag_b
    np_result = np.matmul(np_a.T, np_b)
    assert mag_result.shape == np_result.shape
    assert mag_result.shape == (2, 4)
    np.testing.assert_allclose(tonumpy(mag_result), np_result, atol=EPS)
"""
