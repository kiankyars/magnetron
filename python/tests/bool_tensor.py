from magnetron import *

a = Tensor.bernoulli((4,))
b = Tensor.bernoulli((4,))
print(a & b)
print(a | b)
print(a ^ b)

