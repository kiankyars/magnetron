# (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

import magnetron as mag
import numpy as np
import torch
import matplotlib.pyplot as plt

from bench import *


shapes: list[int] = []
timings_np: list[float] = []
timings_torch: list[float] = []
timings_mag: list[float] = []

iters: int = 10000

def bench_operator(dim: int):
    print(f'Benchmarking {dim}x{dim}')

    # bench numpy
    np_a = np.full((dim, dim), fill_value=1.0, dtype=np.float32)
    np_b = np.full((dim, dim), fill_value=2.0, dtype=np.float32)
    np_result: BenchInfo = bench(dim, iters, np_a, np_b, lambda a, b: a + b)

    # bench torch
    device = torch.device('cpu')
    np_a = torch.full((dim, dim), fill_value=1.0, dtype=torch.float32).to(device)
    np_b = torch.full((dim, dim), fill_value=2.0, dtype=torch.float32).to(device)
    torch_result: BenchInfo = bench(dim, iters, np_a, np_b, lambda a, b: a + b)

    # bench magnetron
    mag_a = mag.Tensor.full((dim, dim), fill_value=1.0, dtype=mag.DType.F32)
    mag_b = mag.Tensor.full((dim, dim), fill_value=2.0, dtype=mag.DType.F32)
    mag_result: BenchInfo = bench(dim, iters, mag_a, mag_b, lambda a, b: a + b)

    shapes.append(dim)
    timings_np.append(np_result.avg_flops())
    timings_torch.append(torch_result.avg_flops())
    timings_mag.append(mag_result.avg_flops())

# Bench different matrix sizes
lim: int = 2048
step: int = 32
i: int = 1
while i <= lim:
    bench_operator(i)
    if i == 1:
        i = 2
    else:
        i += step

# Plot results
# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(shapes, timings_np, label="NumPy", marker='^')
plt.plot(shapes, timings_torch, label="PyTorch", marker='s')
plt.plot(shapes, timings_mag, label="Magnetron", marker='o')

# Add labels and title
plt.xlabel("Matrix Size (NxN)")
plt.ylabel("Average FLOP/s")
plt.title("Performance Comparison of Tensor Libraries")
plt.legend()
plt.grid(True)
plt.show()
