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
        # For column-based math, shape=(out_features, in_features)
        self.weight = Tensor.uniform(shape=(out_features, in_features))
        self.bias = Tensor.uniform(shape=(out_features, 1))
        self._x = None
        self._z = None
        self._out = None

    def forward(self, x: Tensor) -> Tensor:
        """
        If we do:  z = W @ x + b,
        then out = sigmoid(z).

        We'll store both x and out (or z) for backward().
        """
        self._x = x  # store input (shape=(in_features, batch_size))
        self._z = self.weight @ x + self.bias
        self._out = self._z.sigmoid()
        return self._out

    def backward(self, is_hidden_layer: bool, delta: Tensor, rate: float) -> Tensor:
        """
        `delta` here is dL/d(output_of_this_layer).  We do:

            dW = delta @ x^T      (since x is shape=(in_features, batch_size))
            db = mean of delta, per each output neuron
            next_delta = W^T @ delta * σ'(z)   [ only if is_hidden_layer=True ]
        """
        # Weight update
        # delta shape = (out_features, batch_size)
        # x^T   shape = (batch_size, in_features)
        # so delta @ x^T is (out_features, in_features), which matches weight
        self.weight -= (delta @ self._x.transpose().clone()) * rate

        # Bias update: one bias per out_feature => take mean along batch_size axis=1
        # delta.mean(axis=1) gives shape (out_features,) so we keepdims to (out_features, 1)
        #self.bias -= delta.mean(axis=1, keepdims=True) * rate

        batch_size = delta.shape[1]
        ones_vec = Tensor.const([[1.0] for _ in range(batch_size)])
        row_sums = delta @ ones_vec  # shape (out_features, 1)
        row_means = row_sums * (1.0 / batch_size)  # shape (out_features, 1)

        # Then apply it to the bias update:
        self.bias -= row_means * rate

        # For the next layer’s delta = (W^T @ delta) * sigmoid'(z)
        # We must use the derivative of the *post*–linear pre-activation z,
        # or equivalently the derivative wrt the output if we have it stored.
        if is_hidden_layer:
            # shape(W^T) = (in_features, out_features)
            # shape(delta) = (out_features, batch_size)
            d_in = self.weight.transpose().clone() @ delta
            # Multiply by derivative of out = sigmoid(z)
            # i.e. out * (1 - out).  If your library’s .sigmoid(derivative=True)
            # expects the “pre-activated” z, you can do that here.
            d_in *= self._z.sigmoid(derivative=True)
            return d_in
        else:
            # For the last layer, we return delta as is,
            # or skip the activation derivative if you already did it in the top-level.
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
        """
        For the final layer delta, we do:  delta = dL/dOut * sigmoid'(Out)
        Then pass delta backward through each layer.
        """
        error = outputs - targets
        # For the final layer’s activation derivative:
        delta = error * outputs.sigmoid(derivative=True)

        # Backprop through layers from last to first
        for i in reversed(range(len(self.layers))):
            is_hidden = (i > 0)
            delta = self.layers[i].backward(is_hidden, delta, rate)

    def train(self, inputs: Tensor, targets: Tensor, epochs: int, rate: float):
        print(f'Training started for {epochs} epochs with learning rate {rate}')
        import time
        start_time = time.time_ns()

        # Optionally transpose if you want (features, batch) layout
        inputs = inputs.transpose().clone()
        targets = targets.transpose().clone()

        losses = []
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(inputs)
            # Backward pass
            self.backward(output, targets, rate)
            # Compute and record loss
            loss = Optim.mse(output, targets)
            losses.append(loss)
            if epoch % self.loss_epoch_step == 0:
                print(f'Epoch: {epoch}, Loss: {loss:.6f}')

        duration = (time.time_ns() - start_time) / 1e9
        print(f'Training finished in {duration:.2f} seconds')
        return losses