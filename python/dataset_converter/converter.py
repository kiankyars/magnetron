# (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

import magnetron as mag
import pickle

def convert_pickle(file_path: str) -> None:
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    results = {}
    for key in data:
        if key.endswith('w'):
            results[key] = mag.Tensor.from_data(data[key], name=key).T
        elif key.endswith('b'):
            results[key] = mag.Tensor.from_data(data[key], name=key)
    print(results)

convert_pickle('../examples/interactive/mnist_interactive/mnist_mlp_weights.pkl')
