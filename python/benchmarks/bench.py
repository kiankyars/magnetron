# (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

from bench_tool import benchmark, BenchParticipant

import magnetron as mag
import numpy as np
import torch

class NumpyBenchmark(BenchParticipant):
    def __init__(self):
        super().__init__('Numpy')

    def allocate_args(self, dim: int):
        x = np.full((dim, dim), fill_value=1.0, dtype=np.float32)
        y = np.full((dim, dim), fill_value=2.0, dtype=np.float32)
        return x, y

class PyTorchBenchmark(BenchParticipant):
    def __init__(self):
        super().__init__('PyTorch')
        self.device = torch.device('cpu')

    def allocate_args(self, dim: int):
        x = torch.full((dim, dim), fill_value=1.0, dtype=torch.float32).to(self.device)
        y = torch.full((dim, dim), fill_value=2.0, dtype=torch.float32).to(self.device)
        return x, y

class MagnetronBenchmark(BenchParticipant):
    def __init__(self):
        super().__init__('Magnetron')

    def allocate_args(self, dim: int):
        x = mag.Tensor.full((dim, dim), fill_value=1.0, dtype=mag.DType.F32)
        y = mag.Tensor.full((dim, dim), fill_value=2.0, dtype=mag.DType.F32)
        return x, y

participants = [
    MagnetronBenchmark(),
    NumpyBenchmark(),
    PyTorchBenchmark(),
]

operators: list[tuple[str, callable]] = [
    ('Addition', lambda x, y: x + y),
    ('Subtraction', lambda x, y: x - y),
    ('Hadamard Product', lambda x, y: x * y),
    ('Divison', lambda x, y: x / y),
    ('Multiplication', lambda x, y: x @ y)
]

lim: int = 512
step: int = 8

print('Running performance benchmark...')
print('Magnetron VS')
for participant in participants:
    if not isinstance(participant, MagnetronBenchmark):
        print(f'    {participant.name}')
print(f'The benchmark profiles every {step}th shape permutations up to {lim}')
print('The benchmark might take a while')

for op in operators:
    name, fn = op
    print(f'Benchmarking {name} Operator')
    result = benchmark(name, participants, fn, lim, step)
    result.plot()
