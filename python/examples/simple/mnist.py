import magnetron as mag
import magnetron.nn as nn
from magnetron.io import StorageStream
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import urllib.request
from tqdm import tqdm

# Download helper


def download_with_progress(url: str, filename: str) -> None:
    class TqdmBarUpdater(tqdm):
        def update_to(
            self, b: int = 1, bsize: int = 1, tsize: int | None = None
        ) -> None:
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    with TqdmBarUpdater(
        unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]
    ) as t:
        urllib.request.urlretrieve(url, filename, reporthook=t.update_to)


# Download the MNIST dataset

DATASET_URL: str = 'https://huggingface.co/datasets/mario-sieg/magnetron-mnist/resolve/main/mnist_full_e8m23.mag'
DATASET_FILE_NAME: str = 'mnist_full_e8m23.mag'

if not Path(DATASET_FILE_NAME).exists():
    print(f'Downloading dataset file f{DATASET_FILE_NAME}...')
    download_with_progress(DATASET_URL, DATASET_FILE_NAME)
    print('Download complete!')


class MNIST(nn.Module):
    def __init__(self, data_file: str) -> None:
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

        with StorageStream.open(data_file) as stream:
            self.test_images = stream['test_images']
            self.test_labels = stream['test_labels']

            self.fc1.weight.x = stream['fc1_w']
            self.fc1.bias.x = stream['fc1_b']
            self.fc2.weight.x = stream['fc2_w']
            self.fc2.bias.x = stream['fc2_b']

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
    model = MNIST('mnist_full_e8m23.mag')

    with mag.no_grad():
        y_hat: list[int] = model.predict(model.test_images)
        plot_predictions(y_hat, model.test_images)
