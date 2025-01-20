# (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>
# Implements utility functions for magnetron. Requires matplotlib and numpy.

from magnetron.core import Tensor
from matplotlib import pyplot as plt
import numpy as np


def to_numpy(tensor: Tensor, deinterleave: bool = False) -> np.array:
    if deinterleave:
        w: int = tensor.width
        h: int = tensor.height
        c: int = tensor.channels
        data = np.array(tensor.data_as_f32()).reshape(c, h, w)
        data = np.transpose(data, (1, 2, 0))
        return (data * 255.0).astype(np.uint8)
    else:
        return np.array(tensor.data_as_f32(), dtype=np.float32).reshape(tensor.shape)


def plot_tensor_image(tensor: Tensor, title: str | None = None) -> None:
    if title is not None:
        plt.title(title)
    plt.imshow(to_numpy(tensor, deinterleave=True))
    plt.show()


def plot_tensor_scatter(tensor: Tensor, title: str | None = None, deinterleave: bool = False) -> None:
    data = to_numpy(tensor, deinterleave)
    match tensor.rank:
        case 2:
            x, y = np.where(data > 0)
            plt.scatter(x, y, marker='o')
            if title is not None:
                plt.title(title)
            plt.show()
        case 3:
            x, y, z = np.where(data > 0)
            fig = plt.figure()
            if title is not None:
                plt.title(title)
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(x, y, z, marker='o')
            plt.show()
        case 4:
            x, y, z, w = np.where(data > 0)
            fig = plt.figure()
            if title is not None:
                plt.title(title)
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(x, y, z, c=w, cmap='viridis', marker='o')
            plt.colorbar(scatter, label='4th Dimension')
            plt.show()
        case _:
            raise ValueError('Only 2D, 3D and 4D tensors are supported.')

