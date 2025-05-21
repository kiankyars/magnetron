import random

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

model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
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
