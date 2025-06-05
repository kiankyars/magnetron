from magnetron import *

a = Tensor.from_data([True, False, True, True])
b = Tensor.from_data([False, False, True, False])
print(a & b)
print(a | b)
print(a ^ b)

