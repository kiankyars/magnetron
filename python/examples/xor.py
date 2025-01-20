# (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

from magnetron import Tensor
from magnetron.models import SequentialModel, DenseLayer
import matplotlib.pyplot as plt

EPOCHS: int = 10000
RATE: float = 0.01

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

mlp = SequentialModel([
    DenseLayer(2, 4),
    DenseLayer(4, 1)
])

# Train
losses = mlp.train(inputs, targets, epochs=EPOCHS, rate=RATE)

# Inference
outputs = mlp.forward(inputs.transpose().clone())  # if your layer expects (features, batch)
outputs.print()

# Plot MSE loss
plt.plot(list(range(0, EPOCHS)), losses)
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.title('XOR Problem')
plt.show()
