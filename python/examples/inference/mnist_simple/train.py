import gzip, pickle, urllib.request, random, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

url = ("https://github.com/MichalDanielDobrzanski/DeepLearningPython/raw/master/mnist.pkl.gz")
fname = "mnist.pkl.gz"

if not Path(fname).exists():
    print("downloading mnist.pkl.gz …")
    urllib.request.urlretrieve(url, fname)

(train_x, train_y), (_, _), (test_x, test_y) = pickle.load(gzip.open(fname, "rb"), encoding="latin1")

train_x = torch.tensor(train_x, dtype=torch.float32)
train_y = torch.tensor(train_y, dtype=torch.long)
test_x = torch.tensor(test_x,  dtype=torch.float32)
test_y = torch.tensor(test_y,  dtype=torch.long)

net  = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)
opt = optim.Adam(net.parameters(), lr=0.001)
loss = nn.CrossEntropyLoss()

BATCH: int = 256
EPOCHS: int = 5
for epoch in range(EPOCHS):
    idx = list(range(len(train_x)))
    random.shuffle(idx)
    for i in range(0, len(idx), BATCH):
        j = idx[i:i+BATCH]
        opt.zero_grad()
        l = loss(net(train_x[j]), train_y[j])
        l.backward()
        opt.step()
    with torch.no_grad():
        pred = net(test_x).argmax(1)
        acc  = (pred == test_y).float().mean().item()
    print(f"epoch {epoch+1}/{EPOCHS}, acc {acc:.4f}")

data = {
    "fc1_w": net[0].weight.detach().numpy().T.tolist(),
    "fc1_b": net[0].bias.detach().numpy().tolist(),
    "fc2_w": net[2].weight.detach().numpy().T.tolist(),
    "fc2_b": net[2].bias.detach().numpy().tolist(),
}

with open("mnist_mlp_weights.pkl", "wb") as f:
    pickle.dump(data, f)
print("✓ wrote mnist_mlp_weights.pkl")

with open("mnist_test_images.pkl", "wb") as f:
    pickle.dump(test_x.tolist(), f)
with open("mnist_test_labels.pkl", "wb") as f:
    pickle.dump(test_y.tolist(), f)
print("✓ wrote mnist_test_images.pkl / mnist_test_labels.pkl")