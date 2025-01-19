# (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

from bench_tool import (
    benchmark, BenchParticipant,
    generate_matmul_shapes, generate_elementwise_shapes, generate_square_shapes
)

import magnetron as mag
import numpy as np
import torch

class NumpyBenchmark(BenchParticipant):
    def __init__(self):
        super().__init__('Numpy')

    def allocate_args(self, shape_a: tuple[int, int], shape_b: tuple[int, int]):
        x = np.full(shape_a, fill_value=1.0, dtype=np.float32)
        y = np.full(shape_b, fill_value=2.0, dtype=np.float32)
        return x, y

class PyTorchBenchmark(BenchParticipant):
    def __init__(self):
        super().__init__('PyTorch')
        self.device = torch.device('cpu')

    def allocate_args(self, shape_a: tuple[int, int], shape_b: tuple[int, int]):
        x = torch.full(shape_a, fill_value=1.0, dtype=torch.float32).to(self.device)
        y = torch.full(shape_b, fill_value=2.0, dtype=torch.float32).to(self.device)
        return x, y

class MagnetronBenchmark(BenchParticipant):
    def __init__(self):
        super().__init__('Magnetron')

    def allocate_args(self, shape_a: tuple[int, int], shape_b: tuple[int, int]):
        x = mag.Tensor.full(shape_a, fill_value=1.0, dtype=mag.DType.F32)
        y = mag.Tensor.full(shape_b, fill_value=2.0, dtype=mag.DType.F32)
        return x, y

participants = [
    MagnetronBenchmark(),
    NumpyBenchmark(),
    PyTorchBenchmark(),
]

elementwise_ops = [
    ('Addition', lambda x, y: x + y),
    ('Subtraction', lambda x, y: x - y),
    ('Hadamard Product', lambda x, y: x * y),
    ('Division', lambda x, y: x / y),
]

matmul_ops = [
    ('Matrix Multiplication', lambda x, y: x @ y),
]

max_dim = 256
square_step = 8
all_step = max_dim // 4

print('Running performance benchmark...')
print('Magnetron VS')
for participant in participants:
    if not isinstance(participant, MagnetronBenchmark):
        print(f'    {participant.name}')

print('\nSquare Matrix Benchmarks (NxN):')
square_shapes = generate_square_shapes(max_dim, square_step)
for op in elementwise_ops:
    name, fn = op
    print(f'Benchmarking {name} Operator')
    benchmark(name, participants, fn, square_shapes, plot_style='lines')

for op in matmul_ops:
    name, fn = op
    print(f'Benchmarking {name} Operator')
    benchmark(name, participants, fn, square_shapes, plot_style='lines')

print('\nAll Shapes Benchmarks:')
print('Elementwise Operations:')
elementwise_shapes = generate_elementwise_shapes(max_dim, all_step)
for op in elementwise_ops:
    name, fn = op
    print(f'Benchmarking {name} Operator')
    benchmark(name, participants, fn, elementwise_shapes, plot_style='bars')

print('\nMatrix Multiplication:')
matmul_shapes = generate_matmul_shapes(max_dim, all_step)
for op in matmul_ops:
    name, fn = op
    print(f'Benchmarking {name} Operator')
    benchmark(name, participants, fn, matmul_shapes, plot_style='bars')
