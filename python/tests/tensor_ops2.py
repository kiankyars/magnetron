# (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>
import random

from magnetron import Tensor
import numpy as np

EPS = 1e-6


def tonumpy(t: Tensor) -> np.array:
    return np.array(t.tolist(), dtype=np.float32).reshape(t.shape)


def square_shape_permutations(f: callable, lim: int) -> None:
    lim += 1
    for i0 in range(1, lim):
        for i1 in range(1, lim):
            for i2 in range(1, lim):
                for i3 in range(1, lim):
                    for i4 in range(1, lim):
                        for i5 in range(1, lim):
                            f((i0, i1, i2, i3, i4, i5))


def binary_op_square(f: callable, lim: int = 4) -> None:
    def compute(shape: tuple[int, ...]) -> None:
        x = Tensor.uniform(shape)
        y = Tensor.uniform(shape)
        r = f(x, y)
        np.testing.assert_allclose(tonumpy(r), f(tonumpy(x), tonumpy(y)), atol=EPS)

    square_shape_permutations(compute, lim)


def unary_op(
    magf: callable,
    npf: callable,
    lim: int = 4,
    interval: tuple[float, float] = (-1.0, 1.0),
) -> None:
    def compute(shape: tuple[int, ...]) -> None:
        x = Tensor.uniform(shape, interval=interval)
        r = magf(x.clone())
        np.testing.assert_allclose(tonumpy(r), npf(tonumpy(x)), atol=EPS)

    square_shape_permutations(compute, lim)


def scalar_op(f: callable, rhs: bool = True, lim: int = 4) -> None:
    def compute(shape: tuple[int, ...]) -> None:  # x op scalar
        xi: float = random.uniform(-10.0, 10.0)
        x = Tensor.uniform(shape)
        r = f(x, xi)
        np.testing.assert_allclose(tonumpy(r), f(tonumpy(x), xi))

    square_shape_permutations(compute, lim)

    if not rhs:
        return

    def compute(shape: tuple[int, ...]) -> None:  # scalar op x
        xi: float = random.uniform(-10.0, 10.0)
        x = Tensor.uniform(shape)
        r = f(xi, x)
        np.testing.assert_allclose(tonumpy(r), f(xi, tonumpy(x)))

    square_shape_permutations(compute, lim)


def test_unary_op_abs() -> None:
    unary_op(lambda x: x.abs(), lambda x: np.abs(x))
    unary_op(lambda x: x.abs_(), lambda x: np.abs(x))


def test_unary_op_neg() -> None:
    unary_op(lambda x: -x, lambda x: -x)


def test_unary_op_log() -> None:
    unary_op(lambda x: x.log(), lambda x: np.log(x))
    unary_op(lambda x: x.log_(), lambda x: np.log(x))


def test_unary_op_sqr() -> None:
    unary_op(lambda x: x.sqr(), lambda x: x * x)
    unary_op(lambda x: x.sqr_(), lambda x: x * x)


def test_unary_op_sqrt() -> None:
    unary_op(lambda x: x.sqrt(), lambda x: np.sqrt(x))
    unary_op(lambda x: x.sqrt_(), lambda x: np.sqrt(x))


def test_unary_op_sin() -> None:
    unary_op(lambda x: x.sin(), lambda x: np.sin(x))
    unary_op(lambda x: x.sin_(), lambda x: np.sin(x))


def test_unary_op_cos() -> None:
    unary_op(lambda x: x.cos(), lambda x: np.cos(x))
    unary_op(lambda x: x.cos_(), lambda x: np.cos(x))


def test_unary_op_step() -> None:
    unary_op(lambda x: x.step(), lambda x: np.heaviside(x, 0))
    unary_op(lambda x: x.step_(), lambda x: np.heaviside(x, 0))


def test_unary_op_exp() -> None:
    unary_op(lambda x: x.exp(), lambda x: np.exp(x))
    unary_op(lambda x: x.exp_(), lambda x: np.exp(x))


def test_binary_op_add() -> None:
    binary_op_square(lambda x, y: x + y)


def test_binary_op_sub() -> None:
    binary_op_square(lambda x, y: x + y)


def test_binary_op_mul() -> None:
    binary_op_square(lambda x, y: x * y)


def test_binary_op_div() -> None:
    binary_op_square(lambda x, y: x / y)


def test_scalar_op_add() -> None:
    scalar_op(lambda x, xi: x + xi)


def test_scalar_op_sub() -> None:
    scalar_op(lambda x, xi: x + xi)


def test_scalar_op_mul() -> None:
    scalar_op(lambda x, xi: x * xi)


def test_scalar_op_div() -> None:
    scalar_op(lambda x, xi: x / xi)


def test_scalar_op_pow() -> None:
    scalar_op(lambda x, xi: x**xi, rhs=False)
