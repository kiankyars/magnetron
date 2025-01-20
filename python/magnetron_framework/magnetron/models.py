# (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>
# Implements high level model classes for neural networks based on the magnetron.core module.

import time
from abc import ABC

import magnetron as mag


class Layer(ABC):
    def forward(self, inputs: mag.Tensor) -> mag.Tensor:
        pass

    def backward(self, is_in: bool, cache: mag.Tensor, delta: mag.Tensor, rate: float) -> mag.Tensor:
        pass


class Model(ABC):
    def forward(self, inputs: mag.Tensor) -> mag.Tensor:
        pass

    def backward(self, outputs: mag.Tensor, targets: mag.Tensor, rate: float):
        pass

    def train(self, inputs: list[mag.Tensor], targets: list[mag.Tensor], epochs: int, learning_rate: float):
        pass

    def summary(self):
        pass


class Optim:
    @staticmethod
    def mse(y: mag.Tensor, y_hat: mag.Tensor) -> float:
        """Mean Squared Error"""
        return (y - y_hat).sqr_().mean().scalar()

    @staticmethod
    def cross_entropy(y: mag.Tensor, y_hat: mag.Tensor) -> float:
        """Cross Entropy Loss"""
        return -(y * y_hat.log_()).sum().scalar()


class DenseLayer(Layer):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = mag.Tensor.uniform(shape=(out_features, in_features))
        self.bias = mag.Tensor.uniform(shape=(out_features, 1))
        self.cache = None

    def forward(self, prev: mag.Tensor) -> mag.Tensor:
        return (self.weight @ prev + self.bias).sigmoid()

    def backward(self, is_in: bool, cache: mag.Tensor, delta: mag.Tensor, rate: float) -> mag.Tensor:
        self.weight -= (delta @ cache.transpose().clone()) * rate
        self.bias -= delta * rate
        if is_in:
            return (self.weight.transpose() @ delta) * cache.sigmoid(derivative=True)
        else:
            return delta


class SequentialModel(Model):
    def __init__(self, layers: list[DenseLayer]):
        super().__init__()
        assert len(layers) > 0
        self.layers = layers
        self.cache = []
        self.loss_epoch_step = 1000

    def forward(self, inputs: mag.Tensor) -> mag.Tensor:
        prev = inputs
        self.cache.clear()
        self.cache.append(prev)
        for i in range(0, len(self.layers)):
            prev = self.layers[i].forward(prev)
            self.cache.append(prev)
        return prev

    def backward(self, outputs: mag.Tensor, targets: mag.Tensor, rate: float):
        delta = (outputs - targets) * outputs.sigmoid(derivative=True)
        for i in reversed(range(0, len(self.layers))):
            delta = self.layers[i].backward(i > 0, self.cache[i], delta, rate)

    def train(self, inputs: list[mag.Tensor], targets: list[mag.Tensor], epochs: int, learning_rate: float):
        assert len(inputs) == len(targets)
        print(f'Training started {epochs} epochs with learning rate {learning_rate}')
        now = time.time_ns()
        losses = []
        for epoch in range(0, epochs - 1):
            total_loss: float = 0
            for i in range(0, len(inputs)):
                pred: mag.Tensor = self.forward(inputs[i])
                self.backward(pred, targets[i], learning_rate)
                total_loss += Optim.mse(pred, targets[i])
            mean_loss = total_loss / len(inputs)
            losses.append(mean_loss)
            if epoch % self.loss_epoch_step == 0:
                print(f'Epoch: {epoch}, Loss: {mean_loss}')
        print(f'Training finished in {(time.time_ns() - now) / 1e9} seconds')
        return losses

    def summary(self):
        trainable_params = 0
        for layer in self.layers:
            trainable_params += layer.weight.numel + layer.bias.numel
        layers: int = len(self.layers)
        print('---- Model Summary ----')
        print(f'Trainable Parameters: {trainable_params}')
        print(f'Layers: {layers}')
        print('-----------------------')
