import magnetron as mag
import magnetron.nn as nn
import base64
import io
import pickle

from django.http import JsonResponse, HttpRequest, HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import ensure_csrf_cookie
from PIL import Image
import numpy as np


class MNIST(nn.Module):
    def __init__(self, data_file: str) -> None:
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

        with open(data_file, 'rb') as f:
            dat = pickle.load(f)

        self.fc1.weight.x = mag.Tensor.const(dat['fc1_w'], name='fc1_w').T
        self.fc1.bias.x = mag.Tensor.const(dat['fc1_b'], name='fc1_b')
        self.fc2.weight.x = mag.Tensor.const(dat['fc2_w'], name='fc2_w').T
        self.fc2.bias.x = mag.Tensor.const(dat['fc2_b'], name='fc2_b')
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


MODEL_PATH: str = 'mnist_mlp_weights.pkl'
model = MNIST(MODEL_PATH)


@ensure_csrf_cookie
def index(request: HttpRequest) -> HttpResponse:
    return render(request, 'index.html')


def resize_and_center(
    image: Image.Image, target_size: int = 28, padding: int = 4
) -> Image.Image:
    """
    Crops the image to its bounding box of non-background pixels,
    then resizes it while preserving the aspect ratio and centers it
    in a new image of size target_size x target_size.
    - A new PIL Image of size (target_size, target_size).
    """
    arr = np.array(image)
    threshold = 10
    coords = np.column_stack(np.nonzero(arr > threshold))
    if coords.size == 0:
        return image.resize((target_size, target_size))
    top, left = coords.min(axis=0)
    bottom, right = coords.max(axis=0)
    cropped = image.crop((left, top, right + 1, bottom + 1))
    cropped_width, cropped_height = cropped.size
    new_size = target_size - 2 * padding
    scale = new_size / max(cropped_width, cropped_height)
    new_width = int(cropped_width * scale)
    new_height = int(cropped_height * scale)
    resized = cropped.resize((new_width, new_height))
    new_image = Image.new('L', (target_size, target_size), 0)
    x_offset = (target_size - new_width) // 2
    y_offset = (target_size - new_height) // 2
    new_image.paste(resized, (x_offset, y_offset))
    return new_image


def predict_digit(request: HttpRequest) -> JsonResponse:
    if request.method != 'POST':
        return JsonResponse({'error': 'POST request required'}, status=400)

    data_url: str = request.POST.get('image')
    if data_url is None:
        return JsonResponse({'error': 'No image data provided'}, status=400)

    header, encoded = data_url.split(',', 1)
    img_data: bytes = base64.b64decode(encoded)
    image: Image = Image.open(io.BytesIO(img_data)).convert('L')
    image = resize_and_center(image, target_size=28, padding=4)
    arr = np.array(image, dtype=np.float32) / 255.0
    flat_arr: list[float] = arr.flatten().tolist()
    test_tensor: mag.Tensor = mag.Tensor.const([flat_arr], name='input')
    with mag.no_grad():
        pred = model.predict(test_tensor)
    digit: int = pred[0]
    return JsonResponse({'prediction': digit})
