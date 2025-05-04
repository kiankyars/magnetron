from pathlib import Path

import magnetron as mag
import magnetron.nn as nn
import base64
import io

from django.http import JsonResponse, HttpRequest, HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import ensure_csrf_cookie
from PIL import Image
from magnetron.io import StorageStream

from .utils import post_process_image, download_with_progress
import numpy as np

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

    def predict(self, x: mag.Tensor) -> list:
        y_hat: mag.Tensor = self(x)
        flat: list = y_hat.tolist()
        rows, cols = y_hat.shape
        nested: list[float] = [flat[i * cols : (i + 1) * cols] for i in range(rows)]
        preds = []
        for row in nested:
            preds.append(max(range(10), key=row.__getitem__))
        print(preds)
        return preds


model = MNIST(DATASET_FILE_NAME)


@ensure_csrf_cookie
def index(request: HttpRequest) -> HttpResponse:
    return render(request, 'index.html')


@mag.no_grad()
def predict_digit(request: HttpRequest) -> JsonResponse:
    if request.method != 'POST':
        return JsonResponse({'error': 'POST request required'}, status=400)

    data_url: str = request.POST.get('image')
    if data_url is None:
        return JsonResponse({'error': 'No image data provided'}, status=400)

    header, encoded = data_url.split(',', 1)
    img_data: bytes = base64.b64decode(encoded)
    image: Image = Image.open(io.BytesIO(img_data)).convert('L')
    image = post_process_image(image, target_size=28, padding=4)
    arr = np.array(image, dtype=np.float32) / 255.0
    test_tensor: mag.Tensor = mag.Tensor.from_data([arr.flatten().tolist()], name='input')
    pred = model.predict(test_tensor)
    digit: int = pred[0]
    return JsonResponse({'prediction': digit})
