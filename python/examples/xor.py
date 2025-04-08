import magnetron as mag

class XOR(mag.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = mag.Linear(2, 2)
        self.l2 = mag.Linear(2, 1)

    def forward(self, x: mag.Tensor) -> mag.Tensor:
        x = self.l1(x).tanh()
        x = self.l2(x).tanh()
        return x

model = XOR()
params = [
    model.l1.weight,
    model.l1.bias,
    model.l2.weight,
    model.l2.bias,
]
optimizer = mag.optim.SGD(params, lr=1e-1)
criterion = mag.optim.mse_loss

x = mag.Tensor.const([[0, 0], [0, 1], [1, 0], [1, 1]], name='x')
y = mag.Tensor.const([[0], [1], [1], [0]], name='y')

for epoch in range(2000):
    y_hat = model(x)
    loss = criterion(y_hat, y)
    loss.backward()
    loss.export_graphviz('meow.dot')
    optimizer.step()
    optimizer.zero_grad()
    if epoch % 100 == 0:
        print(f'Epoch: {epoch}, Loss: {loss.item()}')
    optimizer.zero_grad()

with mag.no_grad():
    y_hat = model(x)
    print(y_hat)