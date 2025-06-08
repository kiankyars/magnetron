from magnetron import *

a = Tensor.bernoulli((4,))
b = Tensor.bernoulli((4,))
print(a)
print(a & ~b | b | True)
print(a.dtype)
