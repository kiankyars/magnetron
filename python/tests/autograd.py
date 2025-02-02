# (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

import magnetron as mag
import torch

def test_autograd_1():
    x = mag.Tensor.const([-4.0], requires_grad=True)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    magx, magy = x, y
    print(magx.grad)

    x = torch.Tensor([-4.0])
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    torchx, torchy = x, y

    assert magy.item() == torchy.data.item()
    assert magx.grad.item() == torchx.grad.item()
