import magnetron as mag

a = mag.Tensor.full((4, 4), fill_value=0x7fffffff, dtype=mag.int32)
b = mag.Tensor.full((4, 4), fill_value=0, dtype=mag.int32)
print(a)
print(a.dtype)
