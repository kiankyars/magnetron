# (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

from magnetron import *

def test_tensor_fill_zero():
    tensor = Tensor.zeros((1, 2, 3, 4, 5, 6))
    data = tensor.to_list()
    assert len(data) == 1 * 2 * 3 * 4 * 5 * 6
    assert all([x == 0 for x in data])

def test_tensor_fill_x():
    tensor = Tensor.full((1, 2, 3, 4, 5, 6), fill_value=-22)
    data = tensor.to_list()
    assert len(data) == 1 * 2 * 3 * 4 * 5 * 6
    assert all([x == -22 for x in data])

def test_tensor_fill_uniform():
    tensor = Tensor.uniform((1, 2, 3, 4, 5, 6), interval=(-1, 1))
    data = tensor.to_list()
    assert len(data) == 1 * 2 * 3 * 4 * 5 * 6
    assert all([-1 <= x <= 1 for x in data])

def test_tensor_fill_uniform2():
    tensor = Tensor.uniform((1, 2, 3, 4, 5, 6), interval=(0, 100))
    data = tensor.to_list()
    assert len(data) == 1 * 2 * 3 * 4 * 5 * 6
    assert all([0 <= x <= 100 for x in data])

def test_tensor_fill_uniform3():
    tensor = Tensor.uniform((1, 2, 3, 4, 5, 6), interval=(-1000, -20))
    data = tensor.to_list()
    assert len(data) == 1 * 2 * 3 * 4 * 5 * 6
    assert all([-1000 <= x <= -20 for x in data])

def test_tensor_fill_normal():
    mean = 0.0
    stddev = 1
    tensor = Tensor.normal((1, 2, 3, 4, 5, 6), mean=mean, stddev=stddev)
    data = tensor.to_list()
    assert len(data) == 1 * 2 * 3 * 4 * 5 * 6
    #assert all([abs(x - mean) <= 3 * stddev for x in data]) TODO

def test_tensor_fill_const_1d():
    init = [1, 2, 3, 4]
    tensor = Tensor.const(init)
    assert tensor.shape == (4,)
    assert tensor.numel == 4
    assert tensor.rank == 1
    data = tensor.to_list()
    assert data == [1, 2, 3, 4]

def test_tensor_fill_const_2d():
    init = [
        [1, 2],
        [3, 4]
    ]
    tensor = Tensor.const(init)
    assert tensor.shape == (2, 2)
    assert tensor.numel == 4
    assert tensor.rank == 2
    data = tensor.to_list()
    assert data == [1, 2, 3, 4]

def test_tensor_fill_const_3d():
    init = [
        [
           [1, 2],
           [3, 4]
        ],
        [
            [1, 2],
            [3, 4]
        ]
    ]
    tensor = Tensor.const(init)
    assert tensor.shape == (2, 2, 2)
    assert tensor.numel == 8
    assert tensor.rank == 3
    data = tensor.to_list()
    assert data == [1, 2, 3, 4, 1, 2, 3, 4]

def test_tensor_fill_const_4d():
    init = [
        [
            [
                [1, 2],
                [3, 4]
            ],
            [
                [1, 2],
                [3, 4]
            ]
        ],
        [
            [
                [1, 2],
                [3, 4]
            ],
            [
                [1, 2],
                [3, 4]
            ]
        ]
    ]
    tensor = Tensor.const(init)
    assert tensor.shape == (2, 2, 2, 2)
    assert tensor.numel == 16
    assert tensor.rank == 4
    data = tensor.to_list()
    assert data == [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]

def test_tensor_fill_const_5d():
    init = [
        [
            [
                [
                    [1, 2],
                    [3, 4]
                ],
                [
                    [1, 2],
                    [3, 4]
                ]
            ],
            [
                [
                    [1, 2],
                    [3, 4]
                ],
                [
                    [1, 2],
                    [3, 4]
                ]
            ]
        ],
        [
            [
                [
                    [1, 2],
                    [3, 4]
                ],
                [
                    [1, 2],
                    [3, 4]
                ]
            ],
            [
                [
                    [1, 2],
                    [3, 4]
                ],
                [
                    [1, 2],
                    [3, 4]
                ]
            ]
        ]
    ]
    tensor = Tensor.const(init)
    assert tensor.shape == (2, 2, 2, 2, 2)
    assert tensor.numel == 32
    assert tensor.rank == 5
    data = tensor.to_list()
    assert data == [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]

def test_tensor_fill_const_6d():
    init = [
        [
            [
                [
                    [
                        [1, 2],
                        [3, 4]
                    ],
                    [
                        [1, 2],
                        [3, 4]
                    ]
                ],
                [
                    [
                        [1, 2],
                        [3, 4]
                    ],
                    [
                        [1, 2],
                        [3, 4]
                    ]
                ]
            ],
            [
                [
                    [
                        [1, 2],
                        [3, 4]
                    ],
                    [
                        [1, 2],
                        [3, 4]
                    ]
                ],
                [
                    [
                        [1, 2],
                        [3, 4]
                    ],
                    [
                        [1, 2],
                        [3, 4]
                    ]
                ]
            ]
        ],
        [
            [
                [
                    [
                        [1, 2],
                        [3, 4]
                    ],
                    [
                        [1, 2],
                        [3, 4]
                    ]
                ],
                [
                    [
                        [1, 2],
                        [3, 4]
                    ],
                    [
                        [1, 2],
                        [3, 4]
                    ]
                ]
            ],
            [
                [
                    [
                        [1, 2],
                        [3, 4]
                    ],
                    [
                        [1, 2],
                        [3, 4]
                    ]
                ],
                [
                    [
                        [1, 2],
                        [3, 4]
                    ],
                    [
                        [1, 2],
                        [3, 4]
                    ]
                ]
            ]
        ]
    ]
    tensor = Tensor.const(init)
    assert tensor.shape == (2, 2, 2, 2, 2, 2)
    assert tensor.numel == 64
    assert tensor.rank == 6
    data = tensor.to_list()
    assert data == [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]

