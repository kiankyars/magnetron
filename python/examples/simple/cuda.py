# (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

import magnetron as mag

mag.GlobalConfig.verbose = True
mag.GlobalConfig.compute_device = mag.ComputeDevice.CUDA

test = mag.Tensor.uniform(shape=(4, 4))
r = test + test
print(r)
