# (c) 2024 Mario 'Neo' Sieg. <mario.sieg.64@gmail.com>

import approx
import math

import magnetron as mag


def silu(x: float) -> float:
    return x / (1.0 + math.exp(-x))


approx.plot_approximation_error('silu', silu, mag.Operator.SILU, domain=(-2, 2))
