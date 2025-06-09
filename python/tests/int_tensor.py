import magnetron as mag

a = mag.Tensor.uniform((4, 4), from_=0, to=255, dtype=mag.int32)

print(a)
print(a.tolist())
