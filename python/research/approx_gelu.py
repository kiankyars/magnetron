# (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

import approx
import math

import magnetron as mag


def gelu(x: float) -> float:
    return (
        0.5
        * x
        * (
            1.0
            + math.tanh(
                0.79788456080286535587989211986876 * x * (1.0 + 0.044715 * x * x)
            )
        )
    )


approx.plot_approximation_error('gelu', gelu, mag.Operator.GELU, domain=(-2, 2))
