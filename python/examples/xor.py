# This example demonstrates how to implement a simple XOR neural network using Magnetron.
# The XOR problem is a classic example in machine learning, where the model learns to output 1 if the inputs are different and 0 if they are the same.
# The model consists of two linear layers with a tanh activation function.
# The model is trained using the Mean Squared Error (MSE) loss function and the Stochastic Gradient Descent (SGD) optimizer.

import magnetron as mag
import magnetron.nn as nn
import magnetron.optim as optim

from matplotlib import pyplot as plt

EPOCHS: int = 2000

# Define the XOR model architecture
class XOR(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.Linear(2, 2)
        self.l2 = nn.Linear(2, 1)

    def forward(self, x: mag.Tensor) -> mag.Tensor:
        x = self.l1(x).tanh()
        x = self.l2(x).tanh()
        return x

# Create the model, optimizer, and loss function
model = XOR()
optimizer = optim.SGD(model.parameters(), lr=1e-1)
criterion = nn.MSELoss()
loss_values: list[float] = []

x = mag.Tensor.from_data([[0, 0], [0, 1], [1, 0], [1, 1]])
y = mag.Tensor.from_data([[0], [1], [1], [0]])

# Train the model
for epoch in range(EPOCHS):
    y_hat = model(x)
    loss = criterion(y_hat, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    loss_values.append(loss.item())

    if epoch % 100 == 0:
        print(f'Epoch: {epoch}, Loss: {loss.item()}')

# Print the final predictions after the training
print("=== Final Predictions ===")

with mag.no_grad():
    y_hat = model(x)
    for i in range(x.shape[0]):
        print(f'Expected: {y[i]}, Predicted: {y_hat[i]}')

# Plot the loss

plt.figure()
plt.plot(loss_values)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss over Time')
plt.grid(True)
plt.show()

# Cleanup

del model
del criterion
del optimizer
