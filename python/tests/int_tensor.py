import magnetron as mag

a = mag.Tensor.full((4, 4), fill_value=3, dtype=mag.int32)
b = mag.Tensor.full((4, 4), fill_value=-100, dtype=mag.int32)
print(a*b+b)
print(a.dtype)
