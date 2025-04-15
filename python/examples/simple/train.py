import gzip, pickle, urllib.request, random, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import magnetron as mag
from magnetron.io import StorageStream

url = 'https://github.com/MichalDanielDobrzanski/DeepLearningPython/raw/master/mnist.pkl.gz'
fname = 'mnist.pkl.gz'

if not Path(fname).exists():
    print('downloading mnist.pkl.gz …')
    urllib.request.urlretrieve(url, fname)

(train_x, train_y), (_, _), (test_images, test_labels) = pickle.load(
    gzip.open(fname, 'rb'), encoding='latin1'
)

train_x = torch.tensor(train_x, dtype=torch.float32)
train_y = torch.tensor(train_y, dtype=torch.long)
test_images = torch.tensor(test_images, dtype=torch.float32)
test_labels = torch.tensor(test_labels, dtype=torch.long)

net = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
opt = optim.Adam(net.parameters(), lr=0.001)
loss = nn.CrossEntropyLoss()

BATCH: int = 256
EPOCHS: int = 5
for epoch in range(EPOCHS):
    idx = list(range(len(train_x)))
    random.shuffle(idx)
    for i in range(0, len(idx), BATCH):
        j = idx[i : i + BATCH]
        opt.zero_grad()
        l = loss(net(train_x[j]), train_y[j])
        l.backward()
        opt.step()
    with torch.no_grad():
        pred = net(test_images).argmax(1)
        acc = (pred == test_labels).float().mean().item()
    print(f'epoch {epoch + 1}/{EPOCHS}, acc {acc:.4f}')

data = {
    'fc1_w': net[0].weight.detach().numpy().tolist(),
    'fc1_b': net[0].bias.detach().numpy().tolist(),
    'fc2_w': net[2].weight.detach().numpy().tolist(),
    'fc2_b': net[2].bias.detach().numpy().tolist(),
    'test_images': test_images.tolist(),
    'test_labels': test_labels.tolist(),
}

stream = StorageStream()
for key in data:
    tensor = mag.Tensor.from_data(data[key])
    print(f'key: {key}, shape: {tensor.shape}, dtype: {tensor.dtype}')
    stream[key] = tensor
stream.serialize('mnist_full_e8m23.mag')
