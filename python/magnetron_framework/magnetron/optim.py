# (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

from magnetron import Tensor


class PolyLR:
    def __init__(self, initial_lr: float, max_iter: float):
        self.initial_lr = initial_lr
        self.max_iter = max_iter

    def step(self, iter: float) -> float:
        y: float = iter / self.max_iter
        return max(self.initial_lr * (1 - y) ** 2, 1.0e-7)


class Optim:
    def __init__(self, lr: float):
        self.lr = lr

    def mse(self, y: Tensor, y_hat: Tensor) -> float:
        """Mean Squared Error"""
        return (y - y_hat).sqr_().mean()[0]

    def cross_entropy(self, y: Tensor, y_hat: Tensor) -> float:
        """Cross Entropy Loss"""
        return -(y * y_hat.log_()).sum()[0]
