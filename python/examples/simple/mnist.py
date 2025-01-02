# (c) 2024 Mario 'Neo' Sieg. <mario.sieg.64@gmail.com>

from magnetron.models import SequentialModel, DenseLayer

EPOCHS: int = 10
LEARNING_RATE: float = 1e-3

network = SequentialModel([
    DenseLayer(784, 250),
    DenseLayer(250, 100),
    DenseLayer(100, 10)
])

network.summary()
