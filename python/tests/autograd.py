# (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

import magnetron as mag
import torch


def test_autograd_1():
    x = mag.Tensor.const([3.0], requires_grad=True)
    y = mag.Tensor.const([2.0], requires_grad=True)
    assert x.requires_grad
    assert y.requires_grad
    y = (x + y) * (x - y)
    y.backward()
    magx, magy = x, y

    x = torch.Tensor([3.0])
    x.requires_grad = True
    y = torch.Tensor([2.0])
    y.requires_grad = True
    y = (x + y) * (x - y)
    y.backward()
    torchx, torchy = x, y

    assert magy.item() == torchy.data.item()
    assert magx.grad.item() == torchx.grad.item()

def test_autograd_2():
    x = mag.Tensor.const([-4.0], requires_grad=True)
    z = 2 * x + 2 + x
    assert z.requires_grad
    q = z.relu() + z * x
    assert q.requires_grad
    h = (z * z).relu()
    assert h.requires_grad
    y = h + q + q * x
    assert y.requires_grad
    y.backward()
    magx, magy = x, y

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
