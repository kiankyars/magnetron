# (c) 2025 Mario 'Neo' Sieg. <mario.sieg.64@gmail.com>
import random

import pytest
from magnetron import Tensor, DType, no_grad
import numpy as np

DTYPE_TO_NUMPY: dict[DType, np.dtype] = {
    DType.F16: np.float16,
    DType.F32: np.float32,
}

DTYPE_EPS: dict[DType, float] = {DType.F16: 1e-3, DType.F32: 1e-6}


def tonumpy(t: Tensor, dtype: DType) -> np.array:
    return np.array(t.tolist(), dtype=dtype).reshape(t.shape)


def square_shape_permutations(f: callable, lim: int) -> None:
    lim += 1
    for i0 in range(1, lim):
        for i1 in range(1, lim):
            for i2 in range(1, lim):
                for i3 in range(1, lim):
                    for i4 in range(1, lim):
                        for i5 in range(1, lim):
                            f((i0, i1, i2, i3, i4, i5))


def binary_op_square(dtype: DType, f: callable, lim: int = 4) -> None:
    numpy_dt = DTYPE_TO_NUMPY[dtype]

    def compute(shape: tuple[int, ...]) -> None:
        x = Tensor.uniform(shape, dtype=dtype)
        y = Tensor.uniform(shape, dtype=dtype)
        r = f(x, y)
        np.testing.assert_allclose(
            tonumpy(r, numpy_dt),
            f(tonumpy(x, numpy_dt), tonumpy(y, numpy_dt)),
            rtol=DTYPE_EPS[dtype],
        )

    square_shape_permutations(compute, lim)


def unary_op(
    dtype: DType,
    magf: callable,
    npf: callable,
    lim: int = 4,
    interval: tuple[float, float] = (-1.0, 1.0),
) -> None:
    numpy_dt = DTYPE_TO_NUMPY[dtype]

    def compute(shape: tuple[int, ...]) -> None:
        x = Tensor.uniform(shape, dtype=dtype, interval=interval)
        r = magf(x.clone())
        np.testing.assert_allclose(
            tonumpy(r, numpy_dt), npf(tonumpy(x, numpy_dt)), rtol=DTYPE_EPS[dtype]
        )

    square_shape_permutations(compute, lim)


def scalar_op(dtype: DType, f: callable, rhs: bool = True, lim: int = 4) -> None:
    numpy_dt = DTYPE_TO_NUMPY[dtype]

    def compute(shape: tuple[int, ...]) -> None:  # x op scalar
        xi: float = random.uniform(-10.0, 10.0)
        x = Tensor.uniform(shape, dtype=dtype)
        r = f(x, xi)
        np.testing.assert_allclose(tonumpy(r, numpy_dt), f(tonumpy(x, numpy_dt), xi))

    square_shape_permutations(compute, lim)

    if not rhs:
        return

    def compute(shape: tuple[int, ...]) -> None:  # scalar op x
        xi: float = random.uniform(-10.0, 10.0)
        x = Tensor.uniform(shape)
        r = f(xi, x)
        np.testing.assert_allclose(tonumpy(r, numpy_dt), f(xi, tonumpy(x, numpy_dt)))

    square_shape_permutations(compute, lim)


@pytest.mark.parametrize('dtype', [DType.F16, DType.F32])
def test_unary_op_abs(dtype: DType) -> None:
    unary_op(dtype, lambda x: x.abs(), lambda x: np.abs(x))
    with no_grad():
        unary_op(dtype, lambda x: x.abs_(), lambda x: np.abs(x))


@pytest.mark.parametrize('dtype', [DType.F16, DType.F32])
def test_unary_op_neg(dtype: DType) -> None:
    unary_op(dtype, lambda x: -x, lambda x: -x)


@pytest.mark.parametrize('dtype', [DType.F16, DType.F32])
def test_unary_op_log(dtype: DType) -> None:
    unary_op(dtype, lambda x: x.log(), lambda x: np.log(x))
    with no_grad():
        unary_op(dtype, lambda x: x.log_(), lambda x: np.log(x))


@pytest.mark.parametrize('dtype', [DType.F16, DType.F32])
def test_unary_op_sqr(dtype: DType) -> None:
    unary_op(dtype, lambda x: x.sqr(), lambda x: x * x)
    with no_grad():
        unary_op(dtype, lambda x: x.sqr_(), lambda x: x * x)


@pytest.mark.parametrize('dtype', [DType.F16, DType.F32])
def test_unary_op_sqrt(dtype: DType) -> None:
    unary_op(dtype, lambda x: x.sqrt(), lambda x: np.sqrt(x))
    with no_grad():
        unary_op(dtype, lambda x: x.sqrt_(), lambda x: np.sqrt(x))


@pytest.mark.parametrize('dtype', [DType.F16, DType.F32])
def test_unary_op_sin(dtype: DType) -> None:
    unary_op(dtype, lambda x: x.sin(), lambda x: np.sin(x))
    with no_grad():
        unary_op(dtype, lambda x: x.sin_(), lambda x: np.sin(x))


@pytest.mark.parametrize('dtype', [DType.F16, DType.F32])
def test_unary_op_cos(dtype: DType) -> None:
    unary_op(dtype, lambda x: x.cos(), lambda x: np.cos(x))
    with no_grad():
        unary_op(dtype, lambda x: x.cos_(), lambda x: np.cos(x))


@pytest.mark.parametrize('dtype', [DType.F16, DType.F32])
def test_unary_op_step(dtype: DType) -> None:
    unary_op(dtype, lambda x: x.step(), lambda x: np.heaviside(x, 0))
    with no_grad():
        unary_op(dtype, lambda x: x.step_(), lambda x: np.heaviside(x, 0))


@pytest.mark.parametrize('dtype', [DType.F16, DType.F32])
def test_unary_op_exp(dtype: DType) -> None:
    unary_op(dtype, lambda x: x.exp(), lambda x: np.exp(x))
    with no_grad():
        unary_op(dtype, lambda x: x.exp_(), lambda x: np.exp(x))


@pytest.mark.parametrize('dtype', [DType.F16, DType.F32])
def test_binary_op_add(dtype: DType) -> None:
    binary_op_square(dtype, lambda x, y: x + y)


@pytest.mark.parametrize('dtype', [DType.F16, DType.F32])
def test_binary_op_sub(dtype: DType) -> None:
    binary_op_square(dtype, lambda x, y: x + y)


@pytest.mark.parametrize('dtype', [DType.F16, DType.F32])
def test_binary_op_mul(dtype: DType) -> None:
    binary_op_square(dtype, lambda x, y: x * y)


@pytest.mark.parametrize('dtype', [DType.F16, DType.F32])
def test_binary_op_div(dtype: DType) -> None:
    binary_op_square(dtype, lambda x, y: x / y)


@pytest.mark.parametrize('dtype', [DType.F16, DType.F32])
def test_scalar_op_add(dtype: DType) -> None:
    scalar_op(dtype, lambda x, xi: x + xi)


@pytest.mark.parametrize('dtype', [DType.F16, DType.F32])
def test_scalar_op_sub(dtype: DType) -> None:
    scalar_op(dtype, lambda x, xi: x + xi)


@pytest.mark.parametrize('dtype', [DType.F16, DType.F32])
def test_scalar_op_mul(dtype: DType) -> None:
    scalar_op(dtype, lambda x, xi: x * xi)


@pytest.mark.parametrize('dtype', [DType.F16, DType.F32])
def test_scalar_op_div(dtype: DType) -> None:
    scalar_op(dtype, lambda x, xi: x / xi)


@pytest.mark.parametrize('dtype', [DType.F16, DType.F32])
def test_scalar_op_pow(dtype: DType) -> None:
    scalar_op(dtype, lambda x, xi: x**xi, rhs=False)
