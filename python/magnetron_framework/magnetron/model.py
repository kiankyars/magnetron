# (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>
# Implements high level model classes for neural networks based on the magnetron.core module.

import time
from abc import ABC

from magnetron import Tensor


class Layer(ABC):
    def forward(self, inputs: Tensor) -> Tensor:
        pass

    def backward(self, is_hidden_layer: bool, delta: Tensor, rate: float) -> Tensor:
        pass


class Model(ABC):
    def forward(self, inputs: Tensor) -> Tensor:
        pass

    def backward(self, outputs: Tensor, targets: Tensor, rate: float):
        pass

    def train(self, inputs: Tensor, targets: Tensor, epochs: int, learning_rate: float):
        pass

    def summary(self):
        pass


class Optim:
    @staticmethod
    def mse(y: Tensor, y_hat: Tensor) -> float:
        """Mean Squared Error"""
        return (y - y_hat).sqr_().mean()[0]

    @staticmethod
    def cross_entropy(y: Tensor, y_hat: Tensor) -> float:
        """Cross Entropy Loss"""
        return -(y * y_hat.log_()).sum()[0]


class DenseLayer(Layer):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = Tensor.uniform(shape=(out_features, in_features))
        self.bias = Tensor.uniform(shape=(out_features, 1))
        self._x = None
        self._z = None
        self._out = None

    def forward(self, x: Tensor) -> Tensor:
        self._x = x
        self._z = self.weight @ x + self.bias
        self._out = self._z.sigmoid()
        return self._out

    def backward(self, is_hidden_layer: bool, delta: Tensor, rate: float) -> Tensor:
        self.weight -= (delta @ self._x.transpose().clone()) * rate
        batch_size = delta.shape[1]
        ones_vec = Tensor.const([[1.0] for _ in range(batch_size)])
        row_sums = delta @ ones_vec
        row_means = row_sums * (1.0 / batch_size)
        self.bias -= row_means * rate
        if is_hidden_layer:
            d_in = self.weight.transpose().clone() @ delta
            d_in *= self._z.sigmoid(derivative=True)
            return d_in
        else:
            return delta


class SequentialModel(Model):
    def __init__(self, layers: list[DenseLayer]):
        super().__init__()
        self.layers = layers
        self.loss_epoch_step = 1000

    def forward(self, inputs: Tensor) -> Tensor:
        x = inputs
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, outputs: Tensor, targets: Tensor, rate: float):
        error = outputs - targets
        delta = error * outputs.sigmoid(derivative=True)
        for i in reversed(range(len(self.layers))):
            is_hidden = (i > 0)
            delta = self.layers[i].backward(is_hidden, delta, rate)

    def train(self, inputs: Tensor, targets: Tensor, epochs: int, rate: float):
        print(f'Training started for {epochs} epochs with learning rate {rate}')
        import time
        start_time = time.time_ns()

        inputs = inputs.transpose().clone()
        targets = targets.transpose().clone()

        losses = []
        for epoch in range(epochs):
            output = self.forward(inputs)
            self.backward(output, targets, rate)
            loss = Optim.mse(output, targets)
            losses.append(loss)
            if epoch % self.loss_epoch_step == 0:
                print(f'Epoch: {epoch}, Loss: {loss:.6f}')

        duration = (time.time_ns() - start_time) / 1e9
        print(f'Training finished in {duration:.2f} seconds')
        return losses