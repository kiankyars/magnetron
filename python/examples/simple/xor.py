# (c) 2024 Mario 'Neo' Sieg. <mario.sieg.64@gmail.com>

import magnetron as mag
from magnetron.models import SequentialModel, DenseLayer
import matplotlib.pyplot as plt

mag.GlobalConfig.compute_device = mag.ComputeDevice.CPU(2)

EPOCHS: int = 10000
LEARNING_RATE: float = 0.8

# Inputs
inputs = [
    mag.Tensor.const([0.0, 0.0]),
    mag.Tensor.const([0.0, 1.0]),
    mag.Tensor.const([1.0, 0.0]),
    mag.Tensor.const([1.0, 1.0])
]

# Targets
targets = [
    mag.Tensor.const([0.0]),
    mag.Tensor.const([1.0]),
    mag.Tensor.const([1.0]),
    mag.Tensor.const([0.0])
]

mlp = SequentialModel([
    DenseLayer(2, 4),
    DenseLayer(4, 1)
])

# Train model
losses = mlp.train(inputs, targets, EPOCHS, LEARNING_RATE)

# Inference
for input_tensor in inputs:
    input_data = input_tensor.to_list()
    output: float = mlp.forward(input_tensor).scalar()
    print(f'{input_data[0]} ^ {input_data[1]} = {output}')

# Plot MSE loss
plt.plot(list(range(0, EPOCHS - 1)), losses)
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.title('XOR Problem')
plt.show()
