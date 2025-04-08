# (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

epochs: int = 2000

def xor_magnetron():
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

    for epoch in range(epochs):
        y_hat = model(x)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if epoch % 100 == 0:
            print(f'Epoch: {epoch}, Loss: {loss.item()}')
        optimizer.zero_grad()

    with mag.no_grad():
        y_hat = model(x)
        return y_hat

def xor_torch():
    import torch
    import torch.nn as nn
    import torch.optim as optim

    class XOR(nn.Module):
        def __init__(self):
            super(XOR, self).__init__()
            self.l1 = nn.Linear(2, 2)
            self.l2 = nn.Linear(2, 1)

        def forward(self, x):
            x = torch.tanh(self.l1(x))
            x = torch.tanh(self.l2(x))
            return x

    model = XOR()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.MSELoss()

    x = torch.tensor([[0, 0],
                      [0, 1],
                      [1, 0],
                      [1, 1]], dtype=torch.float32)
    y = torch.tensor([[0],
                      [1],
                      [1],
                      [0]], dtype=torch.float32)

    print(model.forward(x))

    for epoch in range(epochs):
        optimizer.zero_grad()
        y_hat = model(x)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f'Epoch: {epoch}, Loss: {loss.item()}')
        optimizer.zero_grad()

    with torch.no_grad():
        y_hat = model(x)
        return y_hat

a = xor_magnetron()
b = xor_torch()

print('Magnetron: ' + str(a.tolist()))
print('PyTorch: ' + str(b.numpy()))
