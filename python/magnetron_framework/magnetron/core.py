# (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

import faulthandler
import typing
import weakref
from dataclasses import dataclass
from enum import Enum, auto, unique
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
    MERSENNE_TWISTER = 0
    PCG = auto()


@unique
class DType(Enum):
    F32 = 0


@unique
class ColorChannels(Enum):
    AUTO = 0
    GRAY = auto()
    GRAY_A = auto()
    RGB = auto()
    RGBA = auto()

    @property
    def argument_count(self) -> int:
        return C.mag_op_get_argcount(self.value)

    @property
    def supports_inplace(self) -> bool:
        return C.mag_op_supports_inplace(self.value)

    @property
    def is_unary(self) -> bool:
        return self.argument_count == 1

    @property
    def is_binary(self) -> bool:
        return self.argument_count == 2


@unique
class GraphEvalOrder(Enum):
    FORWARD = 0
    REVERSE = 1


@unique
class ExecutionMode(Enum):
    EAGER = 0
    DEFERRED = 1


@dataclass
class GlobalConfig:
    verbose: bool = getenv('MAG_VERBOSE', '0') == '1'
    compute_device: ComputeDevice.CPU | ComputeDevice.CUDA = ComputeDevice.CPU()


@typing.final
class Context:
    _active: 'Context' = None

    @staticmethod
    def active() -> 'Context':
        if Context._active is None:
            C.mag_set_log_mode(GlobalConfig.verbose)
            Context._active = Context(GlobalConfig.compute_device)
        return Context._active

    def __init__(
        self,
        device: ComputeDevice.CPU | ComputeDevice.CUDA,
        *,
        execution_mode: ExecutionMode = ExecutionMode.EAGER,
    ) -> None:
        descriptor: ffi.CData = ffi.new('mag_device_descriptor_t*')
        if isinstance(device, ComputeDevice.CPU):
            descriptor.type = 0
            descriptor.thread_count = abs(device.num_threads)
        elif isinstance(device, ComputeDevice.CUDA):
            descriptor.type = 1
            descriptor.cuda_device_id = abs(device.device_id)
        self._ptr = C.mag_ctx_create2(descriptor)
        self.execution_mode = execution_mode

    @property
    def compute_device_name(self) -> str:
        return ffi.string(C.mag_ctx_get_compute_device_name(self._ptr)).decode('utf-8')

    @property
    def execution_mode(self) -> ExecutionMode:
        return ExecutionMode(C.mag_ctx_get_exec_mode(self._ptr))

    @execution_mode.setter
    def execution_mode(self, mode: ExecutionMode) -> None:
        C.mag_ctx_set_exec_mode(self._ptr, mode.value)

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
        C.mag_ctx_profile_start_recording(self._ptr)

    def stop_profiler(self, export_csv_file: str | None = None) -> None:
        csv_file = (
            ffi.NULL if export_csv_file is None else bytes(export_csv_file, 'utf-8')
        )
        C.mag_ctx_profile_stop_recording(self._ptr, csv_file)

    def __del__(self) -> None:
        C.mag_ctx_destroy(self._ptr)
        self._ptr = ffi.NULL


def no_grad() -> 'no_grad.Scope':
    """Temporary disable gradient computation"""

    class Scope:
        def __call__(self, func: callable) -> None:
            def f(*args: tuple[object, ...], **kwargs: dict[str, object]) -> None:
                with Scope():
                    return func(*args, **kwargs)

            return f

        def __enter__(self) -> None:
            pass

        def __exit__(
            self, exc_type: object, exc_value: object, traceback: object
        ) -> None:
            pass

    return Scope()


@typing.final
class Tensor:
    __slots__ = ('_ctx', '_ptr')

    def __init__(self, ptr: ffi.CData | None = None) -> None:
        if isinstance(ptr, ffi.CData):
            assert ptr != ffi.NULL, 'Invalid tensor pointer'
        self._ctx = None
        self._ptr = ptr

    def __del__(self) -> None:
        if (
            hasattr(self, '_ptr')
            and isinstance(self._ptr, ffi.CData)
            and self._ptr != ffi.NULL
        ):
            C.mag_tensor_decref(self._ptr)
        self._ptr = ffi.NULL

    _DISPATCH: list[int, ffi.CData] = {
        1: C.mag_tensor_create_1d,
        2: C.mag_tensor_create_2d,
        3: C.mag_tensor_create_3d,
        4: C.mag_tensor_create_4d,
        5: C.mag_tensor_create_5d,
        6: C.mag_tensor_create_6d,
    }
    assert len(_DISPATCH) == MAX_DIMS

    def _new(
        self,
        ctx: Context,
        *,
        shape: tuple[int, ...],
        dtype: DType = DType.F32,
        requires_grad: bool = False,
        name: str | None = None,
    ) -> None:
        assert 0 < len(shape) <= MAX_DIMS, f'Invalid number of dimensions: {len(shape)}'
        assert all(0 < dim <= DIM_MAX for dim in shape), 'Invalid dimension size'
        self._ctx = weakref.ref(ctx)
        self._ptr = self._DISPATCH[len(shape)](ctx._ptr, dtype.value, *shape)
        self.requires_grad = requires_grad
        if name:
            self.name = name

    @classmethod
    def empty(
        cls,
        shape: tuple[int, ...],
        *,
        dtype: DType = DType.F32,
        requires_grad: bool = False,
        name: str | None = None
    ) -> 'Tensor':
        tensor = cls(None)
        tensor._new(Context.active(), shape=shape, dtype=dtype, requires_grad=requires_grad, name=name)
        return tensor

    @classmethod
    def full(
        cls,
        shape: tuple[int, ...],
        *,
        fill_value: float,
        dtype: DType = DType.F32,
        requires_grad: bool = False,
        name: str | None = None,
    ) -> 'Tensor':
        tensor = cls(None)
        tensor._new(Context.active(), shape=shape, dtype=dtype, requires_grad=requires_grad, name=name)
        C.mag_tensor_fill(tensor._ptr, fill_value)
        return tensor

    @classmethod
    def const(
        cls,
        data: list[float, ...],
        *,
        dtype: DType = DType.F32,
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
                    flattened.extend(flat)
                first_shape = shapes[0]
                for s in shapes:
                    assert s == first_shape, 'All sub-lists must have the same shape'
                return (len(nested),) + first_shape, flattened

        shape, flattened_data = flatten_nested_lists(data)
        tensor = cls(None)
        tensor._new(Context.active(), shape=shape, dtype=dtype, requires_grad=requires_grad, name=name)
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
        dtype: DType = DType.F32,
        requires_grad: bool = False,
        name: str | None = None,
    ) -> 'Tensor':
        return cls.full(shape, fill_value=0.0, dtype=dtype, requires_grad=requires_grad, name=name)

    @classmethod
    def uniform(
        cls,
        shape: tuple[int, ...],
        *,
        interval: (float, float) = (-1.0, 1.0),
        dtype: DType = DType.F32,
        requires_grad: bool = False,
        name: str | None = None,
    ) -> 'Tensor':
        tensor = cls(None)
        tensor._new(Context.active(), shape=shape, dtype=dtype, requires_grad=requires_grad, name=name)
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
        tensor._new(Context.active(), shape=shape, dtype=DType.F32, requires_grad=requires_grad, name=name)
        C.mag_tensor_fill_random_normal(tensor._ptr, mean, stddev)
        return tensor

    @classmethod
    def load(cls, file_path: str) -> 'Tensor':
        assert file_path.endswith('.magnetron'), 'File must be a magnetron file'
        instance = C.mag_tensor_load(Context.active()._ptr, bytes(file_path, 'utf-8'))
        return cls(ptr=instance)

    @classmethod
    def load_image(
        cls,
        file_path: str,
        *,
        name: str | None = None,
        channels: ColorChannels = ColorChannels.AUTO,
        resize_to: (int, int) = (0, 0),
    ) -> 'Tensor':
        assert isfile(file_path), f'File not found: {file_path}'
        instance = C.mag_tensor_load_image(
            Context.active()._ptr,
            bytes(file_path, 'utf-8'),
            channels.value,
            resize_to[0],
            resize_to[1],
        )
        tensor = cls(instance)
        if name is not None:
            tensor.name = name
        return tensor

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
        return C.mag_tensor_rank(self._ptr)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(ffi.unpack(C.mag_tensor_shape(self._ptr), self.rank))

    @property
    def strides(self) -> tuple[int, ...]:
        return tuple(ffi.unpack(C.mag_tensor_strides(self._ptr), self.rank))

    @property
    def dtype(self) -> DType:
        return DType(C.mag_tensor_dtype(self._ptr))

    @property
    def data_ptr(self) -> int:
        return int(ffi.cast('uintptr_t', C.mag_tensor_data_ptr(self._ptr)))

    def item(self) -> float:
        assert self.is_scalar, 'Tensor must be a scalar'
        return self[0]

    def tolist(self) -> list[float]:
        assert self.dtype == DType.F32, 'Invalid data type'
        return ffi.unpack(
            ffi.cast('float*', C.mag_tensor_data_ptr(self._ptr)), self.numel
        )

    @property
    def data_size(self) -> int:
        return C.mag_tensor_data_size(self._ptr)

    @property
    def numel(self) -> int:
        return C.mag_tensor_numel(self._ptr)

    @property
    def num_rows(self) -> int:
        return C.mag_tensor_num_rows(self._ptr)

    @property
    def num_cols(self) -> int:
        return C.mag_tensor_num_cols(self._ptr)

    @property
    def is_scalar(self) -> bool:
        return C.mag_tensor_is_scalar(self._ptr)

    @property
    def is_vector(self) -> bool:
        return C.mag_tensor_is_vector(self._ptr)

    @property
    def is_matrix(self) -> bool:
        return C.mag_tensor_is_matrix(self._ptr)

    @property
    def is_volume(self) -> bool:
        return C.mag_tensor_is_volume(self._ptr)

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
        assert self.requires_grad
        return Tensor(C.mag_tensor_grad(self._ptr))

    def backward(self) -> None:
        assert self.requires_grad
        C.mag_tensor_backward(self._ptr)

    def zero_grad(self) -> None:
        raise NotImplementedError('Not implemented yet')

    def is_close(
        self, other: 'Tensor', eps: float = -1.0, print_eq_percent: bool = False
    ) -> (bool, float):
        percent_eq = ffi.new('double[1]')
        is_eq = C.mag_tensor_is_close(self._ptr, other._ptr, eps, percent_eq)
        if print_eq_percent:
            print(f'Tensors are close: {is_eq}, Percent equal: {percent_eq[0]:.2f}%')
        return is_eq, percent_eq[0]

    def draw_box(
        self, p1: (int, int), p2: (int, int), width: int = 2, rgb: int = 0xFFFFFF
    ) -> None:
        assert p2[0] > p1[0] and p2[1] > p1[1] and width > 0
        C.mag_tensor_img_draw_box(
            self._ptr, p1[0], p1[1], p2[0], p2[1], width, rgb & 0xFFFFFF
        )

    def draw_text(
        self, p: (int, int), size: int, txt: str, rgb: int = 0xFFFFFF
    ) -> None:
        C.mag_tensor_img_draw_text(
            self._ptr, p[0], p[1], size, rgb & 0xFFFFFF, bytes(txt, 'utf-8')
        )

    def save(self, file_path: str) -> None:
        if not file_path.endswith('.magnetron'):
            file_path += '.magnetron'
        C.mag_tensor_save(self._ptr, bytes(file_path, 'utf-8'))

    def save_image(self, file_path: str) -> None:
        assert self.rank == 3, 'Tensor must be a 3D image _ptr'
        assert self.channels in (1, 3, 4), 'Invalid number of color channels'
        C.mag_tensor_save_image(self._ptr, bytes(file_path, 'utf-8'))

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
        return Tensor(C.mag_abs_(self._ptr))

    def neg(self) -> 'Tensor':
        return Tensor(C.mag_neg(self._ptr))

    def neg_(self) -> 'Tensor':
        return Tensor(C.mag_neg_(self._ptr))

    def __neg__(self) -> 'Tensor':
        return self.neg()

    def log(self) -> 'Tensor':
        return Tensor(C.mag_log(self._ptr))

    def log_(self) -> 'Tensor':
        return Tensor(C.mag_log_(self._ptr))

    def sqr(self) -> 'Tensor':
        return Tensor(C.mag_sqr(self._ptr))

    def sqr_(self) -> 'Tensor':
        return Tensor(C.mag_sqr_(self._ptr))

    def sqrt(self) -> 'Tensor':
        return Tensor(C.mag_sqrt(self._ptr))

    def sqrt_(self) -> 'Tensor':
        return Tensor(C.mag_sqrt_(self._ptr))

    def sin(self) -> 'Tensor':
        return Tensor(C.mag_sin(self._ptr))

    def sin_(self) -> 'Tensor':
        return Tensor(C.mag_sin_(self._ptr))

    def cos(self) -> 'Tensor':
        return Tensor(C.mag_cos(self._ptr))

    def cos_(self) -> 'Tensor':
        return Tensor(C.mag_cos_(self._ptr))

    def step(self) -> 'Tensor':
        return Tensor(C.mag_step(self._ptr))

    def step_(self) -> 'Tensor':
        return Tensor(C.mag_step_(self._ptr))

    def exp(self) -> 'Tensor':
        return Tensor(C.mag_exp(self._ptr))

    def exp_(self) -> 'Tensor':
        return Tensor(C.mag_exp_(self._ptr))

    def softmax(self, derivative: bool = False) -> 'Tensor':
        return Tensor(
            C.mag_softmax_dv(self._ptr) if derivative else C.mag_softmax(self._ptr)
        )

    def softmax_(self, derivative: bool = False) -> 'Tensor':
        return Tensor(
            C.mag_softmax_dv_(self._ptr) if derivative else C.mag_softmax_(self._ptr)
        )

    def sigmoid(self, derivative: bool = False) -> 'Tensor':
        return Tensor(
            C.mag_sigmoid_dv(self._ptr) if derivative else C.mag_sigmoid(self._ptr)
        )

    def sigmoid_(self, derivative: bool = False) -> 'Tensor':
        return Tensor(
            C.mag_sigmoid_dv_(self._ptr) if derivative else C.mag_sigmoid_(self._ptr)
        )

    def hard_sigmoid(self) -> 'Tensor':
        return Tensor(C.mag_hard_sigmoid(self._ptr))

    def hard_sigmoid_(self) -> 'Tensor':
        return Tensor(C.mag_hard_sigmoid_(self._ptr))

    def silu(self, derivative: bool = False) -> 'Tensor':
        return C.mag_silu_dv(self._ptr) if derivative else C.mag_silu(self._ptr)

    def silu_(self, derivative: bool = False) -> 'Tensor':
        return Tensor(
            C.mag_silu_dv_(self._ptr) if derivative else C.mag_silu_(self._ptr)
        )

    def tanh(self, derivative: bool = False) -> 'Tensor':
        return Tensor(C.mag_tanh_dv(self._ptr) if derivative else C.mag_tanh(self._ptr))

    def tanh_(self, derivative: bool = False) -> 'Tensor':
        return Tensor(
            C.mag_tanh_dv_(self._ptr) if derivative else C.mag_tanh_(self._ptr)
        )

    def relu(self, derivative: bool = False) -> 'Tensor':
        return Tensor(C.mag_relu_dv(self._ptr) if derivative else C.mag_relu(self._ptr))

    def relu_(self, derivative: bool = False) -> 'Tensor':
        return Tensor(
            C.mag_relu_dv_(self._ptr) if derivative else C.mag_relu_(self._ptr)
        )

    def gelu(self, derivative: bool = False) -> 'Tensor':
        return Tensor(C.mag_gelu_dv(self._ptr) if derivative else C.mag_gelu(self._ptr))

    def gelu_(self, derivative: bool = False) -> 'Tensor':
        return Tensor(
            C.mag_gelu_dv_(self._ptr) if derivative else C.mag_gelu_(self._ptr)
        )

    def __add__(self, other: object | int | float) -> 'Tensor':
        return Tensor(
            C.mag_add(self._ptr, other._ptr)
            if isinstance(other, Tensor)
            else C.mag_adds(self._ptr, float(other))
        )

    def __radd__(self, other: int | float) -> 'Tensor':
        return Tensor(
            C.mag_add(self._ptr, other._ptr)
            if isinstance(other, Tensor)
            else C.mag_adds(self._ptr, float(other))
        )

    def __iadd__(self, other: object | int | float) -> 'Tensor':
        return Tensor(
            C.mag_add_(self._ptr, other._ptr)
            if isinstance(other, Tensor)
            else C.mag_adds_(self._ptr, float(other))
        )

    def __sub__(self, other: object | int | float) -> 'Tensor':
        return Tensor(
            C.mag_sub(self._ptr, other._ptr)
            if isinstance(other, Tensor)
            else C.mag_subs(self._ptr, float(other))
        )

    def __rsub__(self, other: int | float) -> 'Tensor':
        return Tensor.full(self.shape, fill_value=float(other)) - self

    def __isub__(self, other: object | int | float) -> 'Tensor':
        return Tensor(
            C.mag_sub_(self._ptr, other._ptr)
            if isinstance(other, Tensor)
            else C.mag_subs_(self._ptr, float(other))
        )

    def __mul__(self, other: object | int | float) -> 'Tensor':
        return Tensor(
            C.mag_mul(self._ptr, other._ptr)
            if isinstance(other, Tensor)
            else C.mag_muls(self._ptr, float(other))
        )

    def __rmul__(self, other: int | float) -> 'Tensor':
        return Tensor(
            C.mag_mul(self._ptr, other._ptr)
            if isinstance(other, Tensor)
            else C.mag_muls(self._ptr, float(other))
        )

    def __imul__(self, other: object | int | float) -> 'Tensor':
        return Tensor(
            C.mag_mul_(self._ptr, other._ptr)
            if isinstance(other, Tensor)
            else C.mag_muls_(self._ptr, float(other))
        )

    def __truediv__(self, other: object | int | float) -> 'Tensor':
        return Tensor(
            C.mag_div(self._ptr, other._ptr)
            if isinstance(other, Tensor)
            else C.mag_divs(self._ptr, float(other))
        )

    def __rtruediv__(self, other: int | float) -> 'Tensor':
        return Tensor.full(self.shape, fill_value=float(other)) / self

    def __itruediv__(self, other: object | int | float) -> 'Tensor':
        return Tensor(
            C.mag_div_(self._ptr, other._ptr)
            if isinstance(other, Tensor)
            else C.mag_divs_(self._ptr, float(other))
        )

    def __pow__(self, exponent: int | float) -> 'Tensor':
        return Tensor(C.mag_pows(self._ptr, float(exponent)))

    def __ipow__(self, exponent: int | float) -> 'Tensor':
        return Tensor(C.mag_pows_(self._ptr, float(exponent)))

    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        return Tensor(C.mag_matmul(self._ptr, other._ptr))

    def __imatmul__(self, other: 'Tensor') -> 'Tensor':
        return Tensor(C.mag_matmul_(self._ptr, other._ptr))

    def __eq__(self, other: 'Tensor') -> bool:
        return C.mag_tensor_eq(self._ptr, other._ptr)

    def __str__(self) -> str:
        self.print(False, True)
        return ''

    def __getitem__(self, indices: int | tuple[int, ...]) -> float:
        if isinstance(indices, int):
            return C.mag_tensor_get_scalar_virtual_index(self._ptr, indices)
        elif isinstance(indices, tuple):
            idx = indices + (0,) * (MAX_DIMS - len(indices))
            return C.mag_tensor_get_scalar_physical_index(self._ptr, *idx)
        else:
            raise TypeError('Indices must be an int or a tuple of ints.')

    def __setitem__(self, indices: int | tuple[int, ...], value: float) -> None:
        if isinstance(indices, int):
            C.mag_tensor_set_scalar_virtual_index(self._ptr, indices, float(value))
        elif isinstance(indices, tuple):
            idx = indices + (0,) * (MAX_DIMS - len(indices))
            C.mag_tensor_set_scalar_physical_index(self._ptr, *idx, float(value))
        else:
            raise TypeError('Indices must be an int or a tuple of ints.')
