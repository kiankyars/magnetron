import pickle

import magnetron as mag
import magnetron.nn as nn
import numpy as np
from matplotlib import pyplot as plt


class MNIST(nn.Module):
    def __init__(self, data_file: str) -> None:
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

        with open(data_file, 'rb') as f:
            dat = pickle.load(f)

        self.fc1.weight.x = mag.Tensor.from_data(dat['fc1_w'], name='fc1_w').T
        self.fc1.bias.x = mag.Tensor.from_data(dat['fc1_b'], name='fc1_b')
        self.fc2.weight.x = mag.Tensor.from_data(dat['fc2_w'], name='fc2_w').T
        self.fc2.bias.x = mag.Tensor.from_data(dat['fc2_b'], name='fc2_b')
        self.eval()

    def forward(self, x: mag.Tensor) -> mag.Tensor:
        x = self.fc1(x).relu()
        x = self.fc2(x)
        return x.softmax()

    def predict(self, x: mag.Tensor) -> list[int]:
        y_hat: mag.Tensor = self(x)
        flat: list[float] = y_hat.tolist()
        rows, cols = y_hat.shape
        nested: list[float] = [flat[i * cols : (i + 1) * cols] for i in range(rows)]
        preds = []
        for row in nested:
            preds.append(max(range(10), key=row.__getitem__))
        return preds


def load_test_data() -> mag.Tensor:
    with open('mnist_test_images.pkl', 'rb') as f:
        test_images = pickle.load(f)
    return mag.Tensor.from_data(test_images, name='test_images')


def plot_predictions(y_hat: list[int], test_data: mag.Tensor) -> None:
    images = np.array(test_data.tolist()).reshape(-1, 28, 28)
    n_show = 20
    nrows, ncols = 4, 5
    plt.figure(figsize=(10, 8))
    for i in range(n_show):
        plt.subplot(nrows, ncols, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(f'Pred: {y_hat[i]}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    model = MNIST('mnist_mlp_weights.pkl')
    test_data: mag.Tensor = load_test_data()

    with mag.no_grad():
        y_hat: list[int] = model.predict(test_data)
        plot_predictions(y_hat, test_data)
