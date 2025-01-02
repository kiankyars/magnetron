# (c) 2024 Mario 'Neo' Sieg. <mario.sieg.64@gmail.com>

import gzip
import struct
import numpy as np
import magnetron as mag


def ubyte_load_data(src: str, num_samples: int) -> np.ndarray:
    with gzip.open(src) as gz:
        n = struct.unpack("I", gz.read(4))
        assert n[0] == 0x3080000
        n = struct.unpack(">I", gz.read(4))[0]
        assert n == num_samples
        crow = struct.unpack(">I", gz.read(4))[0]
        ccol = struct.unpack(">I", gz.read(4))[0]
        assert crow == ccol == 28
        res = np.frombuffer(gz.read(num_samples * crow * ccol), dtype=np.uint8)
    return res.astype(dtype=np.float32).reshape((num_samples, crow, ccol)) / 256.0


def ubyte_load_labels(src: str, num_samples: int) -> np.ndarray:
    with gzip.open(src) as gz:
        n = struct.unpack("I", gz.read(4))
        assert n[0] == 0x1080000
        n = struct.unpack(">I", gz.read(4))
        assert n[0] == num_samples
        res = np.frombuffer(gz.read(num_samples), dtype=np.uint8)
    return res.reshape(num_samples)


data = ubyte_load_data('../../../datasets/mnist/images-idx3-ubyte.gz', 60000)
mnist = mag.Tensor.const(data=data.tolist())
mnist.print(True, False)
mnist.save('mnist_images.magnetron')
