# (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

def test_import_magnetron():
    import magnetron
    assert magnetron.__version__ is not None


def test_simple_exec():
    import magnetron as mag
    a = mag.Tensor.const([1, 4, 1])
    assert a.max().scalar() == 4
