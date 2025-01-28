# (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import unique, Enum

from magnetron import Tensor


class Layer(ABC):
    """Abstract base class for all layers"""
    @abstractmethod
    def forward(self, inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def backward(self, is_hidden_layer: bool, delta: Tensor, rate: float) -> Tensor:
        pass


@dataclass
class LayerInit:
    """Weight/bias initialization methods and parameters"""

    @unique
    class Dist(Enum):
        NORMAL = 0
        UNIFORM = 1

    @unique
    class Method(Enum):
        RANDOM = 0
        XAVIER = 1
        HE = 2

    method: Method
    distrib: Dist
    uniform_interval: (float, float) = (-1.0, 1.0)
    mean: float = 0.0
    stddev: float = 1.0
    gain: float = 1.0

    def __init__(self, method: Method, distrib: Dist, **kwargs):
        self.method = method
        self.distrib = distrib
        for key, value in kwargs.items():
            setattr(self, key, value)

    def apply(self, shape: tuple[int, ...]) -> Tensor:
        assert len(shape) >= 1

        if self.method == self.Method.RANDOM:
            if self.distrib == self.Dist.NORMAL:
                return Tensor.normal(shape, mean=self.mean, stddev=self.stddev)
            elif self.distrib == self.Dist.UNIFORM:
                return Tensor.uniform(shape, interval=self.uniform_interval)

        fan_in: int = shape[0]
        fan_out: int = shape[1] if self.method == self.Method.XAVIER else None
        factor: float = 1.0
        bound: float = 1.0

        if self.method == self.Method.XAVIER:
            factor = 2.0 / (fan_in + fan_out)
            bound = math.sqrt(6.0 / (fan_in + fan_out))
        elif self.method == self.Method.HE:
            factor = 1.0 / fan_in
            bound = math.sqrt(3.0 / fan_in)

        if self.distrib == self.Dist.NORMAL:
            stddev = self.gain * math.sqrt(factor)
            return Tensor.normal(shape, mean=0.0, stddev=stddev)
        elif self.distrib == self.Dist.UNIFORM:
            return Tensor.uniform(shape, interval=(-self.gain * bound, self.gain * bound))


class DenseLayer(Layer):
    """Fully connected layer"""
    def __init__(self, in_features: int, out_features: int, bias: bool,
                 init: LayerInit = LayerInit(LayerInit.Method.RANDOM, LayerInit.Dist.UNIFORM)):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = init.apply((in_features, out_features))
        self.bias = init.apply((1, out_features)) if bias else None
        self._x = None
        self._z = None
        self._out = None

    def forward(self, x: Tensor) -> Tensor:
        self._x = x
        self._z = x @ self.weight
        if self.bias is not None:
            self._z += self.bias
        self._out = self._z.sigmoid()
        return self._out

    def backward(self, is_hidden_layer: bool, delta: Tensor, rate: float) -> Tensor:
        dW = self._x.T.clone() @ delta
        ones_batch = Tensor.full((delta.shape[0], 1), fill_value=1.0)
        dB = (delta.T.clone() @ ones_batch).T.clone()
        self.weight -= dW * rate
        self.bias -= dB * rate

        next_delta = delta @ self.weight.T.clone()
        if is_hidden_layer:
            next_delta *= self._z.sigmoid(derivative=True)
        return next_delta
