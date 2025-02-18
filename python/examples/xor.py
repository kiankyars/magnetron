# (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

from magnetron import Tensor, Module, Linear
from magnetron.optim import SGD, mse_loss


class XOR(Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = Linear(2, 2)
        self.l2 = Linear(2, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.l1(x).tanh()
        x = self.l2(x).tanh()
        return x


model = XOR()
optim = SGD(model.parameters(), lr=1e-1)

x = Tensor.const([[0, 0], [0, 1], [1, 0], [1, 1]], name='x')

y = Tensor.const([[0], [1], [1], [0]], name='y')

epochs: int = 2

y_hat = model(x)
print(y_hat)
for epoch in range(epochs):
    y_hat = model(x)
    loss = mse_loss(y_hat, y)
    loss.backward()
    loss.export_graphviz(f'xor_{epoch}.dot')
    optim.step()
    optim.zero_grad()
    if epoch % 1000 == 0:
        print(f'Epoch: {epoch}, Loss: {loss.item()}')

y_hat = model(x)
print(y_hat)
