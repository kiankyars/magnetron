# (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

import approx
import math

import magnetron as mag

approx.plot_approximation_error('softmax', math.exp, mag.Operator.SOFTMAX, domain=(-10, 10))
