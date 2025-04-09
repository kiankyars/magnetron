# (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

import contextlib
import faulthandler
import typing
import weakref
from dataclasses import dataclass
from enum import Enum, auto, unique
from functools import lru_cache
from os import getenv
from os.path import isfile

from magnetron._lib_loader import load_native_module

faulthandler.enable()
ffi, C = load_native_module()

MAX_DIMS: int = 6
DIM_MAX: int = (1 << 63) - 1  # INT64_MAX


class ComputeDevice:
    class CPU:
        def __init__(self, num_threads: int = 0) -> None:
            self.num_threads = num_threads

    class CUDA:
        def __init__(self, device_id: int = 0) -> None:
            self.device_id = device_id


@unique
class PRNGAlgorithm(Enum):
    MERSENNE_TWISTER = C.MAG_PRNG_MERSENNE_TWISTER
    PCG = C.MAG_PRNG_PCG


@dataclass
class DType:
    value: int
    size: int
    name: str
    is_fp: bool


e8m23 = f32 = DType(C.MAG_DTYPE_E8M23, 4, 'e8m23', True)
e5m10 = f16 = DType(C.MAG_DTYPE_E5M10, 2, 'e5m10', True)


@dataclass
class GlobalConfig:
    verbose: bool = getenv('MAGNETRON_VERBOSE', '0') == '1'
    compute_device: ComputeDevice.CPU | ComputeDevice.CUDA = ComputeDevice.CPU()


@typing.final
class Context:
    """Manages the execution context and owns all tensors and active compute devices."""

    @staticmethod
    @lru_cache(maxsize=1)
    def primary() -> 'Context':
        """Get global context singleton."""
        C.mag_set_log_mode(GlobalConfig.verbose)
        return Context(GlobalConfig.compute_device)

    def __init__(self, device: ComputeDevice.CPU | ComputeDevice.CUDA) -> None:
        descriptor: ffi.CData = ffi.new('mag_device_descriptor_t*')
        if isinstance(device, ComputeDevice.CPU):
            descriptor.type = 0
            descriptor.thread_count = abs(device.num_threads)
        elif isinstance(device, ComputeDevice.CUDA):
            descriptor.type = 1
            descriptor.cuda_device_id = abs(device.device_id)
        self._ptr = C.mag_ctx_create2(descriptor)
        self.default_dtype = e8m23

    @property
    def compute_device_name(self) -> str:
        return ffi.string(C.mag_ctx_get_compute_device_name(self._ptr)).decode('utf-8')

    @property
    def prng_algorithm(self) -> PRNGAlgorithm:
        return PRNGAlgorithm(C.mag_ctx_get_prng_algorithm(self._ptr))

    @prng_algorithm.setter
    def prng_algorithm(self, algorithm: PRNGAlgorithm) -> None:
        C.mag_ctx_set_prng_algorithm(self._ptr, algorithm.value, 0)

    def seed(self, seed: int) -> None:
        C.mag_ctx_set_prng_algorithm(self._ptr, self.prng_algorithm.value, seed)

    @property
    def os_name(self) -> str:
        return ffi.string(C.mag_ctx_get_os_name(self._ptr)).decode('utf-8')

    @property
    def cpu_name(self) -> str:
        return ffi.string(C.mag_ctx_get_cpu_name(self._ptr)).decode('utf-8')

    @property
    def cpu_virtual_cores(self) -> int:
        return C.mag_ctx_get_cpu_virtual_cores(self._ptr)

    @property
    def cpu_physical_cores(self) -> int:
        return C.mag_ctx_get_cpu_physical_cores(self._ptr)

    @property
    def cpu_sockets(self) -> int:
        return C.mag_ctx_get_cpu_sockets(self._ptr)

    @property
    def physical_memory_total(self) -> int:
        return C.mag_ctx_get_physical_memory_total(self._ptr)

    @property
    def physical_memory_free(self) -> int:
        return C.mag_ctx_get_physical_memory_free(self._ptr)

    @property
    def physical_memory_used(self) -> int:
        return abs(self.physical_memory_total - self.physical_memory_free)

    @property
    def is_numa_system(self) -> bool:
        return C.mag_ctx_is_numa_system(self._ptr)

    @property
    def total_allocated_pool_memory(self) -> int:
        return C.mag_ctx_total_allocated_pool_memory(self._ptr)

    @property
    def total_tensors_created(self) -> int:
        return C.mag_ctx_get_total_tensors_created(self._ptr)

    @property
    def total_tensors_allocated(self) -> int:
        return C.mag_ctx_get_total_tensors_allocated(self._ptr)

    def start_profiler(self) -> None:
        C.mag_ctx_profiler_start(self._ptr)

    def stop_profiler(self, export_csv_file: str | None = None) -> None:
        csv_file: ffi.CData | bytes = (
            ffi.NULL if export_csv_file is None else bytes(export_csv_file, 'utf-8')
        )
        C.mag_ctx_profiler_end(self._ptr, csv_file)

    @property
    def is_profiling(self) -> bool:
        return C.mag_ctx_profiler_is_running(self._ptr)

    def start_grad_recorder(self) -> None:
        C.mag_ctx_grad_recorder_start(self._ptr)

    def stop_grad_recorder(self) -> None:
        C.mag_ctx_grad_recorder_stop(self._ptr)

    @property
    def is_grad_recording(self) -> bool:
        return C.mag_ctx_grad_recorder_is_running(self._ptr)

    def __del__(self) -> None:
        C.mag_ctx_destroy(self._ptr)
        self._ptr = ffi.NULL


class no_grad(contextlib.ContextDecorator):
    """Disables gradient recording within a function or block."""

    def __enter__(self) -> None:
        """Disable gradient tracking by stopping the active context's recorder."""
        Context.primary().stop_grad_recorder()

    def __exit__(self, exc_type: any, exc_value: any, traceback: any) -> None:
        """Re-enable gradient tracking when exiting the context."""
        Context.primary().start_grad_recorder()


_ALLOC_DISPATCH: list[int, ffi.CData] = {
    1: C.mag_tensor_create_1d,
    2: C.mag_tensor_create_2d,
    3: C.mag_tensor_create_3d,
    4: C.mag_tensor_create_4d,
    5: C.mag_tensor_create_5d,
    6: C.mag_tensor_create_6d,
}
assert len(_ALLOC_DISPATCH) == MAX_DIMS


class Tensor:
    """A 1-6 dimensional tensor with support for automatic differentiation."""

    __slots__ = ('__weakref__', '_ctx', '_ptr')

    def __init__(self, ptr: ffi.CData | None = None) -> None:
        self._ctx = None
        self._ptr = ptr

    def __del__(self) -> None:
        if self._ptr is not None and self._ptr != ffi.NULL:
            C.mag_tensor_decref(self._ptr)
        self._ptr = ffi.NULL

    def _new(
        self,
        ctx: Context,
        *,
        shape: tuple[int, ...],
        dtype: DType = Context.primary().default_dtype,
        requires_grad: bool = False,
        name: str | None = None,
    ) -> None:
        assert 0 < len(shape) <= MAX_DIMS, f'Invalid number of dimensions: {len(shape)}'
        assert all(0 < dim <= DIM_MAX for dim in shape), 'Invalid dimension size'
        self._ctx = weakref.ref(ctx)
        self._ptr = _ALLOC_DISPATCH[len(shape)](ctx._ptr, dtype.value, *shape)
        self.requires_grad = requires_grad
        if name:
            self.name = name

    @classmethod
    def empty(
        cls,
        shape: tuple[int, ...],
        *,
        dtype: DType = Context.primary().default_dtype,
        requires_grad: bool = False,
        name: str | None = None,
    ) -> 'Tensor':
        tensor = cls(None)
        tensor._new(
            Context.primary(),
            shape=shape,
            dtype=dtype,
            requires_grad=requires_grad,
            name=name,
        )
        return tensor

    @classmethod
    def full(
        cls,
        shape: tuple[int, ...],
        *,
        fill_value: float,
        dtype: DType = Context.primary().default_dtype,
        requires_grad: bool = False,
        name: str | None = None,
    ) -> 'Tensor':
        tensor = cls(None)
        tensor._new(
            Context.primary(),
            shape=shape,
            dtype=dtype,
            requires_grad=requires_grad,
            name=name,
        )
        C.mag_tensor_fill(tensor._ptr, fill_value)
        return tensor

    @classmethod
    def const(
        cls,
        data: list[float, ...],
        *,
        dtype: DType = Context.primary().default_dtype,
        requires_grad: bool = False,
        name: str | None = None,
    ) -> 'Tensor':
        def flatten_nested_lists(nested: object) -> tuple[tuple[int, ...], list[float]]:
            if not isinstance(nested, list):
                return (), [nested]
            elif len(nested) == 0:
                return (0,), []
            else:
                shapes = []
                flattened = []
                for item in nested:
                    shape_lst, flat = flatten_nested_lists(item)
                    shapes.append(shape_lst)
                    flattened += flat
                first_shape = shapes[0]
                for s in shapes:
                    assert s == first_shape, 'All sub-lists must have the same shape'
                return (len(nested),) + first_shape, flattened

        shape, flattened_data = flatten_nested_lists(data)
        tensor = cls(None)
        tensor._new(
            Context.primary(),
            shape=shape,
            dtype=dtype,
            requires_grad=requires_grad,
            name=name,
        )
        size: int = len(flattened_data) * ffi.sizeof('float')
        C.mag_tensor_copy_buffer_from(
            tensor._ptr, ffi.new(f'float[{len(flattened_data)}]', flattened_data), size
        )
        return tensor

    @classmethod
    def zeros(
        cls,
        shape: tuple[int, ...],
        *,
        dtype: DType = Context.primary().default_dtype,
        requires_grad: bool = False,
        name: str | None = None,
    ) -> 'Tensor':
        return cls.full(
            shape, fill_value=0.0, dtype=dtype, requires_grad=requires_grad, name=name
        )

    @classmethod
    def uniform(
        cls,
        shape: tuple[int, ...],
        *,
        interval: (float, float) = (-1.0, 1.0),
        dtype: DType = Context.primary().default_dtype,
        requires_grad: bool = False,
        name: str | None = None,
    ) -> 'Tensor':
        tensor = cls(None)
        tensor._new(
            Context.primary(),
            shape=shape,
            dtype=dtype,
            requires_grad=requires_grad,
            name=name,
        )
        if interval[1] < interval[0]:
            interval = (interval[1], interval[0])
        C.mag_tensor_fill_random_uniform(tensor._ptr, interval[0], interval[1])
        return tensor

    @classmethod
    def normal(
        cls,
        shape: tuple[int, ...],
        *,
        mean: float = 0.0,
        stddev: float = 1.0,
        requires_grad: bool = False,
        name: str | None = None,
    ) -> 'Tensor':
        tensor = cls(None)
        tensor._new(
            Context.primary(),
            shape=shape,
            dtype=DType.F32,
            requires_grad=requires_grad,
            name=name,
        )
        C.mag_tensor_fill_random_normal(tensor._ptr, mean, stddev)
        return tensor

    @classmethod
    def load(cls, file_path: str) -> 'Tensor':
        assert file_path.endswith('.magnetron'), 'File must be a magnetron file'
        instance = C.mag_tensor_load(Context.primary()._ptr, bytes(file_path, 'utf-8'))
        return cls(ptr=instance)

    def print(self, print_header: bool = False, print_data: bool = True) -> None:
        C.mag_tensor_print(self._ptr, print_header, print_data)

    @property
    def name(self) -> str:
        return ffi.string(C.mag_tensor_get_name(self._ptr)).decode('utf-8')

    @name.setter
    def name(self, name: str) -> None:
        C.mag_tensor_set_name(self._ptr, bytes(name, 'utf-8'))

    @property
    def rank(self) -> int:
        return C.mag_tensor_get_rank(self._ptr)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(ffi.unpack(C.mag_tensor_get_shape(self._ptr), self.rank))

    @property
    def strides(self) -> tuple[int, ...]:
        return tuple(ffi.unpack(C.mag_tensor_get_strides(self._ptr), self.rank))

    @property
    def dtype(self) -> DType:
        return DType(C.mag_tensor_get_dtype(self._ptr))

    @property
    def data_ptr(self) -> int:
        return int(ffi.cast('uintptr_t', C.mag_tensor_get_data_ptr(self._ptr)))

    def item(self) -> float:
        return self.tolist()[0]

    def tolist(self) -> list[float]:
        ptr: ffi.CData = C.mag_tensor_to_float_array(
            self._ptr
        )  # Convert tensor dtype to float array
        unpacked: list[float] = ffi.unpack(ptr, self.numel)
        C.mag_tensor_to_float_array_free_data(ptr)  # Free allocated native float array
        return unpacked

    @property
    def data_size(self) -> int:
        return C.mag_tensor_get_data_size(self._ptr)

    @property
    def numel(self) -> int:
        return C.mag_tensor_get_numel(self._ptr)

    @property
    def is_transposed(self) -> bool:
        return C.mag_tensor_is_transposed(self._ptr)

    @property
    def is_permuted(self) -> bool:
        return C.mag_tensor_is_permuted(self._ptr)

    def is_shape_eq(self, other: 'Tensor') -> bool:
        return C.mag_tensor_is_shape_eq(self._ptr, other._ptr)

    def are_strides_eq(self, other: 'Tensor') -> bool:
        return C.mag_tensor_are_strides_eq(self._ptr, other._ptr)

    def can_broadcast(self, other: 'Tensor') -> bool:
        return C.mag_tensor_can_broadcast(self._ptr, other._ptr)

    @property
    def width(self) -> int:
        return self.shape[2]

    @property
    def height(self) -> int:
        return self.shape[1]

    @property
    def channels(self) -> int:
        return self.shape[0]

    @property
    def is_contiguous(self) -> bool:
        return C.mag_tensor_is_contiguous(self._ptr)

    @property
    def requires_grad(self) -> bool:
        return C.mag_tensor_requires_grad(self._ptr)

    @requires_grad.setter
    def requires_grad(self, require: bool) -> None:
        C.mag_tensor_set_requires_grad(self._ptr, require)

    @property
    def grad(self) -> 'Tensor':
        if not self.requires_grad:
            return None
        ptr: ffi.CData = C.mag_tensor_get_grad(self._ptr)
        if ptr is None or ptr == ffi.NULL:
            return None
        return Tensor(ptr)

    def backward(self) -> None:
        assert self.requires_grad, 'Tensor must require gradient tracking'
        assert self.rank == 1 and self.numel == 1, 'Tensor must be scalar'
        C.mag_tensor_backward(self._ptr)

    def zero_grad(self) -> None:
        assert self.requires_grad, 'Tensor must require gradient tracking'
        C.mag_tensor_zero_grad(self._ptr)

    def save(self, file_path: str) -> None:
        if not file_path.endswith('.magnetron'):
            file_path += '.magnetron'
        C.mag_tensor_save(self._ptr, bytes(file_path, 'utf-8'))

    def export_graphviz(self, file_path: str) -> None:
        C.mag_tensor_export_graphviz(self._ptr, bytes(file_path, 'utf-8'))

    def clone(self) -> 'Tensor':
        return Tensor(C.mag_clone(self._ptr))

    def view(self) -> 'Tensor':
        return Tensor(C.mag_view(self._ptr))

    def transpose(self) -> 'Tensor':
        return Tensor(C.mag_transpose(self._ptr))

    @property
    def T(self) -> 'Tensor':
        return Tensor(C.mag_transpose(self._ptr))

    def permute(self, axes: tuple[int, ...]) -> 'Tensor':
        assert len(axes) == self.rank, (
            f'Invalid number of axes, require {self.rank}, got {len(axes)}'
        )
        if len(axes) != MAX_DIMS:
            axes = axes + tuple(range(self.rank, MAX_DIMS))
        assert len(axes) == MAX_DIMS
        for i in range(MAX_DIMS):
            assert 0 <= axes[i] < MAX_DIMS
            for j in range(i + 1, MAX_DIMS):
                assert axes[i] != axes[j], f'Duplicate axis: {axes[i]}'
        return Tensor(C.mag_permute(self._ptr, *axes))

    def mean(self) -> 'Tensor':
        return Tensor(C.mag_mean(self._ptr))

    def min(self) -> 'Tensor':
        return Tensor(C.mag_min(self._ptr))

    def max(self) -> 'Tensor':
        return Tensor(C.mag_max(self._ptr))

    def sum(self) -> 'Tensor':
        return Tensor(C.mag_sum(self._ptr))

    def abs(self) -> 'Tensor':
        return Tensor(C.mag_abs(self._ptr))

    def abs_(self) -> 'Tensor':
        assert not self.requires_grad, (
            'In-place operations are not supported for gradient-tracking tensors'
        )
        return Tensor(C.mag_abs_(self._ptr))

    def neg(self) -> 'Tensor':
        return Tensor(C.mag_neg(self._ptr))

    def neg_(self) -> 'Tensor':
        assert not self.requires_grad, (
            'In-place operations are not supported for gradient-tracking tensors'
        )
        return Tensor(C.mag_neg_(self._ptr))

    def __neg__(self) -> 'Tensor':
        return self.neg()

    def log(self) -> 'Tensor':
        return Tensor(C.mag_log(self._ptr))

    def log_(self) -> 'Tensor':
        assert not self.requires_grad, (
            'In-place operations are not supported for gradient-tracking tensors'
        )
        return Tensor(C.mag_log_(self._ptr))

    def sqr(self) -> 'Tensor':
        return Tensor(C.mag_sqr(self._ptr))

    def sqr_(self) -> 'Tensor':
        assert not self.requires_grad, (
            'In-place operations are not supported for gradient-tracking tensors'
        )
        return Tensor(C.mag_sqr_(self._ptr))

    def sqrt(self) -> 'Tensor':
        return Tensor(C.mag_sqrt(self._ptr))

    def sqrt_(self) -> 'Tensor':
        assert not self.requires_grad, (
            'In-place operations are not supported for gradient-tracking tensors'
        )
        return Tensor(C.mag_sqrt_(self._ptr))

    def sin(self) -> 'Tensor':
        return Tensor(C.mag_sin(self._ptr))

    def sin_(self) -> 'Tensor':
        assert not self.requires_grad, (
            'In-place operations are not supported for gradient-tracking tensors'
        )
        return Tensor(C.mag_sin_(self._ptr))

    def cos(self) -> 'Tensor':
        return Tensor(C.mag_cos(self._ptr))

    def cos_(self) -> 'Tensor':
        assert not self.requires_grad, (
            'In-place operations are not supported for gradient-tracking tensors'
        )
        return Tensor(C.mag_cos_(self._ptr))

    def step(self) -> 'Tensor':
        return Tensor(C.mag_step(self._ptr))

    def step_(self) -> 'Tensor':
        assert not self.requires_grad, (
            'In-place operations are not supported for gradient-tracking tensors'
        )
        return Tensor(C.mag_step_(self._ptr))

    def exp(self) -> 'Tensor':
        return Tensor(C.mag_exp(self._ptr))

    def exp_(self) -> 'Tensor':
        assert not self.requires_grad, (
            'In-place operations are not supported for gradient-tracking tensors'
        )
        return Tensor(C.mag_exp_(self._ptr))

    def softmax(self) -> 'Tensor':
        return Tensor(C.mag_softmax(self._ptr))

    def softmax_(self) -> 'Tensor':
        assert not self.requires_grad, (
            'In-place operations are not supported for gradient-tracking tensors'
        )
        return Tensor(C.mag_softmax_(self._ptr))

    def sigmoid(self) -> 'Tensor':
        return Tensor(C.mag_sigmoid(self._ptr))

    def sigmoid_(self) -> 'Tensor':
        assert not self.requires_grad, (
            'In-place operations are not supported for gradient-tracking tensors'
        )
        return Tensor(C.mag_sigmoid_(self._ptr))

    def hard_sigmoid(self) -> 'Tensor':
        return Tensor(C.mag_hard_sigmoid(self._ptr))

    def hard_sigmoid_(self) -> 'Tensor':
        assert not self.requires_grad, (
            'In-place operations are not supported for gradient-tracking tensors'
        )
        return Tensor(C.mag_hard_sigmoid_(self._ptr))

    def silu(self) -> 'Tensor':
        return Tensor(C.mag_silu(self._ptr))

    def silu_(self) -> 'Tensor':
        assert not self.requires_grad, (
            'In-place operations are not supported for gradient-tracking tensors'
        )
        return Tensor(C.mag_silu_(self._ptr))

    def tanh(self) -> 'Tensor':
        return Tensor(C.mag_tanh(self._ptr))

    def tanh_(self) -> 'Tensor':
        assert not self.requires_grad, (
            'In-place operations are not supported for gradient-tracking tensors'
        )
        return Tensor(C.mag_tanh_(self._ptr))

    def relu(self) -> 'Tensor':
        return Tensor(C.mag_relu(self._ptr))

    def relu_(self) -> 'Tensor':
        assert not self.requires_grad, (
            'In-place operations are not supported for gradient-tracking tensors'
        )
        return Tensor(C.mag_relu_(self._ptr))

    def gelu(self) -> 'Tensor':
        return Tensor(C.mag_gelu(self._ptr))

    def gelu_(self) -> 'Tensor':
        assert not self.requires_grad, (
            'In-place operations are not supported for gradient-tracking tensors'
        )
        return Tensor(C.mag_gelu_(self._ptr))

    def __add__(self, other: object | int | float) -> 'Tensor':
        if not isinstance(other, Tensor):
            other = Tensor.full(self.shape, fill_value=float(other))
        return Tensor(C.mag_add(self._ptr, other._ptr))

    def __radd__(self, other: int | float) -> 'Tensor':
        other = Tensor.full(self.shape, fill_value=float(other))
        return other + self

    def __iadd__(self, other: object | int | float) -> 'Tensor':
        assert not self.requires_grad, (
            'In-place operations are not supported for gradient-tracking tensors'
        )
        if not isinstance(other, Tensor):
            other = Tensor.full(self.shape, fill_value=float(other))
        return Tensor(C.mag_adds_(self._ptr, float(other)))

    def __sub__(self, other: object | int | float) -> 'Tensor':
        if not isinstance(other, Tensor):
            other = Tensor.full(self.shape, fill_value=float(other))
        return Tensor(C.mag_sub(self._ptr, other._ptr))

    def __rsub__(self, other: int | float) -> 'Tensor':
        other = Tensor.full(self.shape, fill_value=float(other))
        return other - self

    def __isub__(self, other: object | int | float) -> 'Tensor':
        assert not self.requires_grad, (
            'In-place operations are not supported for gradient-tracking tensors'
        )
        if not isinstance(other, Tensor):
            other = Tensor.full(self.shape, fill_value=float(other))
        return Tensor(C.mag_sub_(self._ptr, other._ptr))

    def __mul__(self, other: object | int | float) -> 'Tensor':
        if not isinstance(other, Tensor):
            other = Tensor.full(self.shape, fill_value=float(other))
        return Tensor(C.mag_mul(self._ptr, other._ptr))

    def __rmul__(self, other: int | float) -> 'Tensor':
        other = Tensor.full(self.shape, fill_value=float(other))
        return other * self

    def __imul__(self, other: object | int | float) -> 'Tensor':
        assert not self.requires_grad, (
            'In-place operations are not supported for gradient-tracking tensors'
        )
        if not isinstance(other, Tensor):
            other = Tensor.full(self.shape, fill_value=float(other))
        return Tensor(C.mag_mul_(self._ptr, other._ptr))

    def __truediv__(self, other: object | int | float) -> 'Tensor':
        if not isinstance(other, Tensor):
            other = Tensor.full(self.shape, fill_value=float(other))
        return Tensor(C.mag_div(self._ptr, other._ptr))

    def __rtruediv__(self, other: int | float) -> 'Tensor':
        other = Tensor.full(self.shape, fill_value=float(other))
        return other / self

    def __itruediv__(self, other: object | int | float) -> 'Tensor':
        assert not self.requires_grad, (
            'In-place operations are not supported for gradient-tracking tensors'
        )
        if not isinstance(other, Tensor):
            other = Tensor.full(self.shape, fill_value=float(other))
            return Tensor(C.mag_div_(self._ptr, other._ptr))

    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        return Tensor(C.mag_matmul(self._ptr, other._ptr))

    def __imatmul__(self, other: 'Tensor') -> 'Tensor':
        assert not self.requires_grad, (
            'In-place operations are not supported for gradient-tracking tensors'
        )
        return Tensor(C.mag_matmul_(self._ptr, other._ptr))

    def __pow__(self, exponent: int | float) -> 'Tensor':
        return Tensor(C.mag_pows(self._ptr, float(exponent)))

    def __ipow__(self, exponent: int | float) -> 'Tensor':
        return Tensor(C.mag_pows_(self._ptr, float(exponent)))

    def __eq__(self, other: 'Tensor') -> bool:
        return C.mag_tensor_eq(self._ptr, other._ptr)

    def __str__(self) -> str:
        self.print(False, True)
        return ''

    def __getitem__(self, indices: int | tuple[int, ...]) -> float:
        if isinstance(indices, int):
            return C.mag_tensor_subscript_get_flattened(self._ptr, indices)
        elif isinstance(indices, tuple):
            idx = indices + (0,) * (MAX_DIMS - len(indices))
            return C.mag_tensor_subscript_get_multi(self._ptr, *idx)
        else:
            raise TypeError('Indices must be an int or a tuple of ints.')

    def __setitem__(self, indices: int | tuple[int, ...], value: float) -> None:
        if isinstance(indices, int):
            C.mag_tensor_subscript_set_flattened(self._ptr, indices, float(value))
        elif isinstance(indices, tuple):
            idx = indices + (0,) * (MAX_DIMS - len(indices))
            C.mag_tensor_subscript_set_multi(self._ptr, *idx, float(value))
        else:
            raise TypeError('Indices must be an int or a tuple of ints.')
