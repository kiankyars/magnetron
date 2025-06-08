import magnetron as mag

a = mag.Tensor.full((4, 4), fill_value=1, dtype=mag.int32)
b = mag.Tensor.full((4, 4), fill_value=8, dtype=mag.int32)
print(((a << b) - 1) / 2)
