from random import random

import magnetron as mag
import magnetron.nn as nn
import magnetron.optim as optim

from matplotlib import pyplot as plt

BATCH: int = 256
EPOCHS: int = 5

train_x = mag.Tensor.zeros((BATCH, 784))
train_y = mag.Tensor.zeros((BATCH, 10))
test_images = mag.Tensor.zeros((BATCH, 784))
test_labels = mag.Tensor.zeros((BATCH, 10))

# Define the model architecture


class MNIST(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.Linear(784, 128)
        self.l2 = nn.Linear(128, 10)

    def forward(self, x: mag.Tensor) -> mag.Tensor:
        x = self.l1(x).relu()
        x = self.l2(x)
        return x


model = MNIST()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

loss_values: list[float] = []

for epoch in range(EPOCHS):
    idx = list(range(len(train_x)))
    random.shuffle(idx)
    for i in range(0, len(idx), BATCH):
        j = idx[i : i + BATCH]
        optimizer.zero_grad()
        loss = criterion(model(train_x[j]), train_y[j])
        loss.backward()
        optimizer.step()

        loss_values.append(loss.item())

    with mag.no_grad():
        pred = model(test_images).argmax(1)
        # acc = (pred == test_labels).float().mean().item()
    # print(f'epoch {epoch + 1}/{EPOCHS}, acc {acc:.4f}')

# Plot the loss

plt.figure()
plt.plot(loss_values)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss over Time')
plt.grid(True)
plt.show()
