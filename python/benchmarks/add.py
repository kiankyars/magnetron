# (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

import magnetron as mag
import numpy
import torch
from bench import *

mag.GlobalConfig.verbose = True

n: int = 8192
iters: int = 500

print(f'Benchmarking addition of {n}x{n} matrices with {iters} iterations...')

# bench numpy
np_a = numpy.full((n, n), fill_value=1.0)
np_b = numpy.full((n, n), fill_value=2.0)

print('Benchmarking numpy...')
np_result: BenchInfo = bench(n, iters, np_a, np_b, lambda a, b: a + b)

# bench torch
device = torch.device('cpu')
np_a = torch.full((n, n), fill_value=1.0).to(device)
np_b = torch.full((n, n), fill_value=2.0).to(device)

print('Benchmarking torch...')
torch_result: BenchInfo = bench(n, iters, np_a, np_b, lambda a, b: a + b)

# bench magnetron

print('Benchmarking magnetron...')
mag_a = mag.Tensor.full((n, n), fill_value=1.0)
mag_b = mag.Tensor.full((n, n), fill_value=2.0)
mag_result: BenchInfo = bench(n, iters, mag_a, mag_b, lambda a, b: a + b)

print('Numpy:')
print(np_result)
print('Torch:')
print(torch_result)
print('Magnetron:')
print(mag_result)

avg_np: float = np_result.avg_flops()
avg_torch: float = torch_result.avg_flops()
avg_mag: float = mag_result.avg_flops()

print('==========================================================')
if avg_np < avg_mag:
    print(f'Magnetron is {avg_mag / avg_np:.2f} times faster than numpy.')
else:
    print(f'Magnetron is {avg_np / avg_mag:.2f} times slower than numpy.')
if avg_torch < avg_mag:
    print(f'Magnetron is {avg_mag / avg_torch:.2f} times faster than torch.')
else:
    print(f'Magnetron is {avg_torch / avg_mag:.2f} times slower than torch.')
