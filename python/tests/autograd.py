# (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

from magnetron import *
import torch

def test_autograd_1():
    x = Tensor.const([-4.0], requires_grad=True)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    assert x.requires_grad
    assert z.requires_grad
    assert q.requires_grad
    assert h.requires_grad
    assert y.requires_grad
    y.backward()
    xmg, ymg = x, y

    x = torch.Tensor([-4.0])
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xpt, ypt = x, y

    assert ymg.item() == ypt.data.item()
    #assert xmg.grad.item() == xpt.grad.item()
