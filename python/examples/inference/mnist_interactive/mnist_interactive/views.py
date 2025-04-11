import base64
import io
import pickle

from django.http import JsonResponse
from django.shortcuts import render
from PIL import Image, ImageOps
import numpy as np

import magnetron as mag
import magnetron.nn as nn
from django.views.decorators.csrf import ensure_csrf_cookie


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
        nested: list[float] = [flat[i * cols:(i + 1) * cols] for i in range(rows)]
        preds = []
        for row in nested:
            preds.append(max(range(10), key=row.__getitem__))
        return preds


MODEL_PATH = 'mnist_mlp_weights.pkl'
model = MNIST(MODEL_PATH)

@ensure_csrf_cookie
def index(request):
    return render(request, 'index.html')

def predict_digit(request) -> JsonResponse:
    if request.method != 'POST':
        return JsonResponse({'error': 'POST request required'}, status=400)

    data_url: str = request.POST.get('image')
    if data_url is None:
        return JsonResponse({'error': 'No image data provided'}, status=400)
    header, encoded = data_url.split(',', 1)
    img_data: bytes = base64.b64decode(encoded)
    image: Image = Image.open(io.BytesIO(img_data)).convert('L')
    image = ImageOps.invert(image)
    image = image.resize((28, 28))
    arr= np.array(image, dtype=np.float32) / 255.0
    flat_arr: list[float] = arr.flatten().tolist()
    test_tensor: mag.Tensor = mag.Tensor.const([flat_arr], name='input')
    with mag.no_grad():
        prediction = model.predict(test_tensor)
    digit: int = prediction[0]
    return JsonResponse({'prediction': digit})
