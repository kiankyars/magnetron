# (c) 2024 Mario 'Neo' Sieg. <mario.sieg.64@gmail.com>

import magnetron as mag
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["axes.formatter.limits"] = (-99, 99)  # Disable scientific notation to show all digits


def plot_approximation_error(name: str, exact_func: callable, approx_op: mag.Operator, domain: (float, float),
                             step: float = 0.0001):
    x_values = [i * step for i in range(int(domain[0] / step), int(domain[1] / step))]
    exact = [exact_func(x) for x in x_values]
    approx = mag.Tensor.operator(approx_op, False, None, mag.Tensor.const(x_values)).data_as_f32()
    errors = [abs(exact[i] - approx[i]) for i in range(len(exact))]
    assert len(exact) == len(approx) == len(errors) == len(x_values)

    plt.figure(figsize=(10, 5))
    plt.plot(x_values, exact, label=f'Exact {name}', color='blue')
    plt.plot(x_values, approx, label=f'Approx {name}', color='orange', linestyle='--')
    plt.title(f'Exact vs Approx {name}')
    plt.xlabel('x')
    plt.ylabel(f'{name}(x)')
    plt.legend()
    plt.grid(True)

    plt.figure(figsize=(10, 5))
    plt.plot(x_values, errors, label='Absolute Error', color='red')
    plt.title(f'Error in {name} Approximation')
    plt.xlabel('x')
    plt.ylabel('Absolute Error')
    plt.plot([], [], ' ', label=f'Samples {len(errors)}')
    plt.plot([], [], ' ', label=f'Mean Error: {sum(errors) / len(errors):.20f}')
    plt.plot([], [], ' ', label=f'Min Error: {min(errors):.20f}')
    plt.plot([], [], ' ', label=f'Max Error: {max(errors):.20f}')
    plt.legend()
    plt.grid(True)

    plt.show()
