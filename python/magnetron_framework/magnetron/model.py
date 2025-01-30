# (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from magnetron import Tensor
from magnetron.layer import Layer
from magnetron.optim import Optimizer


@dataclass
class HyperParams:
    lr: float = 0.01
    epochs: int = 10000
    epoch_step: int = 1000


class Model(ABC):
    """Abstract base class for all models"""

    def __init__(self, hyper_params: HyperParams) -> None:
        self.hyper_params = hyper_params

    @abstractmethod
    def forward(self, inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def backward(self, outputs: Tensor, targets: Tensor, rate: float) -> None:
        pass

    @abstractmethod
    def train(self, inputs: Tensor, targets: Tensor) -> None:
        pass

    @abstractmethod
    def summary(self) -> None:
        pass


class SequentialModel(Model):
    """Feedforward neural network model (multi-layer perceptron)"""

    def __init__(self, hyper_params: HyperParams, layers: list[Layer]) -> None:
        super().__init__(hyper_params)
        self.layers = layers

    def forward(self, inputs: Tensor) -> Tensor:
        x = inputs
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, outputs: Tensor, targets: Tensor, rate: float) -> None:
        delta = (outputs - targets) * outputs.sigmoid(derivative=True)
        for i in reversed(range(len(self.layers))):
            is_hidden = i > 0
            delta = self.layers[i].backward(is_hidden, delta, rate)

    def train(self, inputs: Tensor, targets: Tensor) -> None:
        epochs: int = self.hyper_params.epochs
        rate: float = self.hyper_params.lr

        print(f'Training started for {epochs} epochs with learning rate {rate}')
        start_time = time.time_ns()
        losses = []
        for epoch in range(epochs):
            output = self.forward(inputs)
            self.backward(output, targets, rate)
            loss = Optimizer.mse(output, targets)
            losses.append(loss)
            if epoch % self.hyper_params.epoch_step == 0:
                print(f'Epoch: {epoch}, Loss: {loss:.6f}')

        duration = (time.time_ns() - start_time) / 1e9
        print(f'Training finished in {duration:.2f} seconds')
        return losses

    def summary(self) -> None:
        pass
