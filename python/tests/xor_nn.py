# (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

import numpy as np
from magnetron import Tensor

learning_rate = 0.1
epochs = 10000

def xor_nn_np():
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(x):
        return x * (1 - x)

    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])

    Y = np.array([[0], [1], [1], [0]])

    W1 = np.random.randn(2, 4)
    b1 = np.zeros((1, 4))
    W2 = np.random.randn(4, 1)
    b2 = np.zeros((1, 1))

    for epoch in range(epochs):
        z1 = np.matmul(X, W1) + b1
        a1 = sigmoid(z1)
        z2 = np.matmul(a1, W2) + b2
        a2 = sigmoid(z2)

        loss = np.mean((Y - a2) ** 2)

        if epoch % 1000 == 0:
            print(f'Epoch: {epoch}, loss: {loss}')

        d_a2 = -(Y - a2)
        d_z2 = d_a2 * sigmoid_derivative(a2)
        d_W2 = np.matmul(a1.T, d_z2)
        d_b2 = np.sum(d_z2)

        d_a1 = np.matmul(d_z2, W2.T)
        d_z1 = d_a1 * sigmoid_derivative(a1)
        d_W1 = np.matmul(X.T, d_z1)
        d_b1 = np.sum(d_z1)

        W2 -= learning_rate * d_W2
        b2 -= learning_rate * d_b2
        W1 -= learning_rate * d_W1
        b1 -= learning_rate * d_b1

    def predict(x):
        z1 = np.matmul(x, W1) + b1
        a1 = sigmoid(z1)
        z2 = np.matmul(a1, W2) + b2
        a2 = sigmoid(z2)
        return a2

    for input_ in X:
        output = predict(input_)
        print(output[0][0])

def tonumpy(t: Tensor):
    return np.array(t.tolist(), dtype=np.float32).reshape(t.shape)

def fromnumpy(a: np.ndarray):
    return Tensor.const(a.tolist())

def xor_nn_mag():
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(x):
        return x * (1 - x)

    X = Tensor.const([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])

    Y = Tensor.const([[0], [1], [1], [0]])

    W1 = np.random.randn(2, 4)
    b1 = np.zeros((1, 4))
    W2 = np.random.randn(4, 1)
    b2 = np.zeros((1, 1))

    W1 = fromnumpy(W1)
    b1 = fromnumpy(b1)
    W2 = fromnumpy(W2)
    b2 = fromnumpy(b2)

    for epoch in range(epochs):
        a1 = (X @ W1 + b1).sigmoid()
        a2 = (a1 @ W2 + b2).sigmoid()

        loss = (Y - a2).sqr_().mean()[0]

        if epoch % 1000 == 0:
            print(f'Epoch: {epoch}, loss: {loss}')

        d_a2 = -(Y - a2)
        d_z2 = d_a2 * sigmoid_derivative(a2)
        d_W2 = a1.T.clone() @ d_z2
        d_b2 = d_z2.sum()

        d_z1 = (d_z2 @ W2.T.clone()) * sigmoid_derivative(a1)

        ld_W1 = X.T.clone() @ d_z1
        print(f'{X.T.clone().shape} @ {d_z1.shape} = {ld_W1.shape}')

        d_z1 = tonumpy(d_z1)

        d_W1 = tonumpy(X).T @ d_z1
        print(f'{tonumpy(X).T.shape} @ {d_z1.shape} = {d_W1.shape}')

        d_z1 = fromnumpy(d_z1)
        d_W1 = fromnumpy(d_W1)

        d_b1 = d_z1.sum()

        W2 -= learning_rate * d_W2
        b2 -= learning_rate * d_b2
        W1 -= learning_rate * d_W1
        b1 -= learning_rate * d_b1

    def predict(x):
        z1 = x @ W1 + b1
        a1 = z1.sigmoid()
        z2 = a1 @ W2 + b2
        a2 = z2.sigmoid()
        return a2

    XX = [[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]]
    for input_ in XX:
        output = predict(Tensor.const([input_]))[0]
        print(output)

xor_nn_np()
xor_nn_mag()
