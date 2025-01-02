# (c) 2024 Mario 'Neo' Sieg. <mario.sieg.64@gmail.com>

import approx
import math

import magnetron as mag

approx.plot_approximation_error('tanh', math.tanh, mag.Operator.TANH, domain=(-2, 2))
