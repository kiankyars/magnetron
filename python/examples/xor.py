# (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

from magnetron import Tensor
from magnetron.layer import DenseLayer
from magnetron.model import SequentialModel, HyperParams
import matplotlib.pyplot as plt

EPOCHS: int = 10000
RATE: float = 0.1

# Inputs: shape (4, 2)
inputs = Tensor.const([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
])

# Targets: shape (4, 1)
targets = Tensor.const([
    [0.0],
    [1.0],
    [1.0],
    [0.0]
])

params = HyperParams(lr=RATE, epochs=EPOCHS)
mlp = SequentialModel(params, [
    DenseLayer(2, 8),
    DenseLayer(8, 1)
])

# Train
losses = mlp.train(inputs, targets)

# Inference
test_points = [
    (0.0, 0.0),
    (0.0, 1.0),
    (1.0, 0.0),
    (1.0, 1.0),
]

for (x_val, y_val) in test_points:
    result = mlp.forward(Tensor.const([x_val, y_val]))[0]
    print(f"{x_val} XOR {y_val} => {result:.4f}")

# Plot MSE loss
plt.plot(list(range(0, EPOCHS)), losses)
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.title('XOR Problem')
plt.show()
