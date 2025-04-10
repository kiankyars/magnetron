import magnetron as mag
import magnetron.nn as nn
import magnetron.optim as optim


class XOR(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.Linear(2, 2)
        self.l2 = nn.Linear(2, 1)

    def forward(self, x: mag.Tensor) -> mag.Tensor:
        x = self.l1(x).tanh()
        x = self.l2(x).tanh()
        return x


model = XOR()
optimizer = optim.SGD(model.parameters(), lr=1e-1)
criterion = nn.MSELoss()

x = mag.Tensor.const([[0, 0], [0, 1], [1, 0], [1, 1]])
y = mag.Tensor.const([[0], [1], [1], [0]])

for epoch in range(2000):
    y_hat = model(x)
    loss = criterion(y_hat, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if epoch % 100 == 0:
        print(f'Epoch: {epoch}, Loss: {loss.item()}')

with mag.no_grad():
    y_hat = model(x)
    print(y_hat.tolist())
