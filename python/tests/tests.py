# (c) 2024 Mario 'Neo' Sieg. <mario.sieg.64@gmail.com>

def test_import_magnetron():
    import magnetron
    assert magnetron.__version__ is not None


def test_simple_exec():
    import magnetron as mag
    a = mag.Tensor.const([1, 4, 1])
    assert a.max().scalar() == 4

from magnetron import *

def test_context_creation():
    # Test that a context can be created and defaults are correct.
    ctx = Context.active()
    assert ctx.compute_device in (ComputeDevice.CPU, ComputeDevice.CUDA)
    assert ctx.execution_mode.name in ('EAGER', 'DEFERRED')
    assert isinstance(ctx.os_name, str)
    assert isinstance(ctx.cpu_name, str)
    assert ctx.cpu_virtual_cores >= 1
    assert ctx.cpu_physical_cores >= 1

def test_tensor_creation():
    # Test creating a simple tensor
    t = Tensor.empty((3, 3))
    assert t.shape == (3, 3)
    assert t.numel == 9
    assert t.dtype == DType.F32
    assert t.is_matrix is True

def test_tensor_fill():
    # Test creating a tensor filled with a constant value
    val = 42.0
    t = Tensor.full((2, 2), fill_value=val)
    data = t.to_list()
    assert all(x == val for x in data)
    assert t.shape == (2, 2)

def test_tensor_zeros():
    # Test zero tensor
    t = Tensor.zeros((2, 3))
    data = t.to_list()
    assert all(x == 0.0 for x in data)

def test_tensor_ops():
    # Test basic arithmetic operations
    a = Tensor.const([[1, 2], [3, 4]])
    b = Tensor.const([[4, 3], [2, 1]])

    c = a + b
    assert c.rank == 2
    assert c.shape == (2, 2)
    assert c.to_list() == [5.0, 5.0, 5.0, 5.0]

    d = a - b
    assert d.to_list() == [-3.0, -1.0, 1.0, 3.0]

    e = a * 2.0
    assert e.to_list() == [2.0, 4.0, 6.0, 8.0]

def test_tensor_inplace_ops():
    # Test in-place operations
    t = Tensor.const([1, 2, 3])
    t += 1
    assert t.to_list() == [2.0, 3.0, 4.0]

    t *= 2
    assert t.to_list() == [4.0, 6.0, 8.0]

def test_tensor_unary_ops():
    # Test unary operations like neg, abs, sqrt
    t = Tensor.const([-1, -4, 9])
    neg_t = t.neg()
    assert neg_t.to_list() == [1.0, 4.0, -9.0]

    abs_t = t.abs()
    assert abs_t.to_list() == [1.0, 4.0, 9.0]

    sqrt_t = abs_t.sqrt()
    # sqrt(1)=1, sqrt(4)=2, sqrt(9)=3
    assert sqrt_t.to_list() == [1.0, 2.0, 3.0]

def test_save_and_load(tmp_path):
    t = Tensor.const([[1, 2], [3, 4]])
    file_path = tmp_path / 'test_tensor.magnetron'
    t.save(str(file_path))
    loaded = Tensor.load(str(file_path))
    assert loaded.shape == t.shape
    assert loaded.to_list() == t.to_list()
