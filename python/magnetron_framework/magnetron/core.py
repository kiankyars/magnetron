# (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

# Implements core functionality: Context, Tensors and Operations.

# To debug Python to C FFI calls:
# $ cp examples/perceptron.py tmp.py && gdb -ex r --args python3 tmp.py
# See also https://wiki.python.org/moin/DebuggingWithGdb

import faulthandler
import weakref
from dataclasses import dataclass
from os import getenv
from os.path import isfile
from magnetron._lib_loader import load_native_module
from enum import Enum, auto

# Enable faulthandler for debugging
faulthandler.enable()

ffi, C = load_native_module()  # Load the native magnetron shared library

# Common constants
MAX_DIMS: int = 6
MAX_ARG_TENSORS: int = 2
MAG_MAX_OP_PARAMS: int = 6
DIM_MAX: int = ((1 << 64) - 1) >> 1


def enable_log(enable: bool) -> None:
    """
    Set logging mode for the magnetron backend.

    Parameters
    ----------
    enable : bool
        If True, enables logging of operations and internal states.
    """
    C.mag_set_log_mode(enable)


def pack_color(r: int, g: int, b: int) -> int:
    """
    Packs three 8-bit color components (R, G, B) into a single 24-bit integer.

    Parameters
    ----------
    r : int
        Red component [0-255].
    g : int
        Green component [0-255].
    b : int
        Blue component [0-255].

    Returns
    -------
    int
        A single integer representing the packed RGB color.
    """
    return C.mag_pack_color_u8(r, g, b)


class ComputeDevice:
    """
    Compute devices available for parallel computations.
    """
    class CPU:
        """
        CPU device configuration.
        """
        def __init__(self, num_threads: int = 0):
            """
            Initializes a new CPU device configuration.

            Parameters
            ----------
            num_threads : int, optional
                Number of threads to use, 0 for automatic, by default 0.
            """

            self.num_threads = num_threads

    class CUDA:
        """
        CUDA device configuration.
        """
        def __init__(self, device_id: int = 0):
            """
            Initializes a new CUDA device configuration.

            Parameters
            ----------
            device_id : int, optional
                CUDA device ID, by default 0.
            """

            self.device_id = device_id

class PRNGAlgorithm(Enum):
    """
    Pseudorandom number generator algorithms.
    """
    MERSENNE_TWISTER = 0  # Default - Mersenne Twister
    PCG = auto()  # Permuted Congruential Generator


class DType(Enum):
    """
    Supported data types for tensors.
    """
    F32 = 0


class ColorChannels(Enum):
    """
    Desired color channels when loading images.
    """
    AUTO = 0  # Automatically determine the number of color channels
    GRAY = auto()  # Grayscale F32
    GRAY_A = auto()  # Grayscale F32 + Alpha
    RGB = auto()  # R32G32B32
    RGBA = auto()  # R32G32B32A32

    @property
    def name(self) -> str:
        """Returns the operator name as a string."""
        return ffi.string(C.mag_op_get_name(self.value)).decode('utf-8')

    @property
    def mnemonic(self) -> str:
        """Returns a short mnemonic name for the operator."""
        return ffi.string(C.mag_op_get_mnemonic(self.value)).decode('utf-8')

    @property
    def argument_count(self) -> int:
        """Returns the number of _ptr arguments required by this operator."""
        return C.mag_op_get_argcount(self.value)

    @property
    def supports_inplace(self) -> bool:
        """Checks if the operator supports in-place modifications."""
        return C.mag_op_supports_inplace(self.value)

    @property
    def is_unary(self) -> bool:
        """Checks if the operator is a unary operator."""
        return self.argument_count == 1

    @property
    def is_binary(self) -> bool:
        """Checks if the operator is a binary operator."""
        return self.argument_count == 2


class GraphEvalOrder(Enum):
    """
    Order in which the computation graph should be evaluated.
    Applies to deferred execution mode only.
    """
    FORWARD = 0  # Evaluate graph left-to-right
    REVERSE = 1  # Evaluate graph right-to-left


class ExecutionMode(Enum):
    """
    Execution modes for the magnetron context.
    """
    EAGER = 0  # Execute operations immediately (dynamic graph)
    DEFERRED = 1  # Build computation graph and execute later (static graph)


@dataclass
class GlobalConfig:
    verbose: bool = (getenv('MAG_VERBOSE', '0') == '1')
    compute_device: ComputeDevice.CPU | ComputeDevice.CUDA = ComputeDevice.CPU()


class Context:
    """
    Manages the magnetron context and _ptr lifecycles, including device selection,
    memory allocation, and execution mode.

    A global active context is created automatically when needed.
    """

    _active: 'Context' = None  # Global context

    @staticmethod
    def active() -> 'Context':
        """
        Returns the active global context, creating one if it does not exist.

        Returns
        -------
        Context
            The currently active context.
        """
        if Context._active is None:
            enable_log(GlobalConfig.verbose)
            Context._active = Context(GlobalConfig.compute_device)
        return Context._active

    def __init__(self, device: ComputeDevice.CPU | ComputeDevice.CUDA, *, execution_mode: ExecutionMode = ExecutionMode.EAGER):
        """
        Initializes a new magnetron context.

        Parameters
        ----------
        device : ComputeDevice
            The compute device (CPU or CUDA).
        execution_mode : ExecutionMode, optional
            The execution mode (eager or deferred), by default EAGER.
        """
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
        """
        Returns the name of the active compute device.

        Returns
        -------
        str
            Name of the device, e.g. "CPU" or "NVIDIA GPU".
        """
        return ffi.string(C.mag_ctx_get_compute_device_name(self._ptr)).decode('utf-8')

    @property
    def execution_mode(self) -> ExecutionMode:
        """
        Returns the execution mode of the context.

        Returns
        -------
        ExecutionMode
            The current execution mode (EAGER or DEFERRED).
        """
        return ExecutionMode(C.mag_ctx_get_exec_mode(self._ptr))

    @execution_mode.setter
    def execution_mode(self, mode: ExecutionMode):
        """
        Sets the execution mode of the context.

        Parameters
        ----------
        mode : ExecutionMode
            Desired mode (EAGER or DEFERRED).
        """
        C.mag_ctx_set_exec_mode(self._ptr, mode.value)

    @property
    def prng_algorithm(self) -> PRNGAlgorithm:
        """
        Returns the PRNG algorithm currently used.

        Returns
        -------
        PRNGAlgorithm
            The PRNG algorithm in use.
        """
        return PRNGAlgorithm(C.mag_ctx_get_prng_algorithm(self._ptr))

    @prng_algorithm.setter
    def prng_algorithm(self, algorithm: PRNGAlgorithm):
        """
        Sets the PRNG algorithm and re-seeds it.

        Parameters
        ----------
        algorithm : PRNGAlgorithm
            The desired PRNG algorithm.
        """
        C.mag_ctx_set_prng_algorithm(self._ptr, algorithm.value, 0)

    @property
    def os_name(self) -> str:
        """
        Returns the operating system name.

        Returns
        -------
        str
            The OS name (e.g., "Linux").
        """
        return ffi.string(C.mag_ctx_get_os_name(self._ptr)).decode('utf-8')

    @property
    def cpu_name(self) -> str:
        """
        Returns the CPU name/brand string.

        Returns
        -------
        str
            The CPU name.
        """
        return ffi.string(C.mag_ctx_get_cpu_name(self._ptr)).decode('utf-8')

    @property
    def cpu_virtual_cores(self) -> int:
        """
        Returns the number of virtual CPU cores (logical processors).

        Returns
        -------
        int
            Count of virtual CPU cores.
        """
        return C.mag_ctx_get_cpu_virtual_cores(self._ptr)

    @property
    def cpu_physical_cores(self) -> int:
        """
        Returns the number of physical CPU cores.

        Returns
        -------
        int
            Count of physical CPU cores.
        """
        return C.mag_ctx_get_cpu_physical_cores(self._ptr)

    @property
    def cpu_sockets(self) -> int:
        """
        Returns the number of CPU sockets present on the system.

        Returns
        -------
        int
            CPU socket count.
        """
        return C.mag_ctx_get_cpu_sockets(self._ptr)

    @property
    def physical_memory_total(self) -> int:
        """
        Returns the total physical memory (RAM) in bytes.

        Returns
        -------
        int
            Total system memory in bytes.
        """
        return C.mag_ctx_get_physical_memory_total(self._ptr)

    @property
    def physical_memory_free(self) -> int:
        """
        Returns the amount of free physical memory (RAM) in bytes.

        Returns
        -------
        int
            Free memory in bytes.
        """
        return C.mag_ctx_get_physical_memory_free(self._ptr)

    @property
    def physical_memory_used(self) -> int:
        """
        Returns the amount of used physical memory (RAM) in bytes.

        Returns
        -------
        int
            Used memory in bytes.
        """
        return abs(self.physical_memory_total - self.physical_memory_free)

    @property
    def is_numa_system(self) -> bool:
        """
        Checks if the system uses Non-Uniform Memory Access (NUMA).

        Returns
        -------
        bool
            True if NUMA is supported, False otherwise.
        """
        return C.mag_ctx_is_numa_system(self._ptr)

    @property
    def total_allocated_pool_memory(self) -> int:
        """
        Returns the total allocated memory (pool) by this context.

        Returns
        -------
        int
            Total allocated memory in bytes.
        """
        return C.mag_ctx_total_allocated_pool_memory(self._ptr)

    @property
    def total_tensors_created(self) -> int:
        """
        Returns the total number of tensors created (including views).

        Returns
        -------
        int
            Count of all tensors created.
        """
        return C.mag_ctx_get_total_tensors_created(self._ptr)

    @property
    def total_tensors_allocated(self) -> int:
        """
        Returns the total number of allocated tensors (not including views).

        Returns
        -------
        int
            Count of allocated tensors.
        """
        return C.mag_ctx_get_total_tensors_allocated(self._ptr)

    def start_profiler(self) -> None:
        """
        Starts recording profiling information for operations.
        Profiling must be stopped to produce a report. Slightly reduces performance.
        """
        C.mag_ctx_profile_start_recording(self._ptr)

    def stop_profiler(self, export_csv_file: str | None = None) -> None:
        """
        Stops recording profiling information and generates a report.

        Parameters
        ----------
        export_csv_file : str, optional
            Path to export a CSV profiling report. If None, only an internal report is generated.
        """
        csv_file = ffi.NULL if export_csv_file is None else bytes(export_csv_file, 'utf-8')
        C.mag_ctx_profile_stop_recording(self._ptr, csv_file)

    def __del__(self):
        """
        Destructor that releases context resources.
        """
        C.mag_ctx_destroy(self._ptr)
        self._ptr = ffi.NULL


class Tensor:
    """
    Represents a _ptr in the magnetron library. Supports various operations and transformations.
    """

    def __init__(self, ptr: ffi.CData | None = None) -> None:
        """
        Internal constructor. Do not call directly. Use static methods or operators.

        Parameters
        ----------
        ptr : ffi.CData or None
            Internal tensor instance (C data pointer).
        """
        self._ctx = None
        self._ptr = ptr

    def __del__(self) -> None:
        """Releases _ptr resources upon object destruction."""
        if isinstance(self._ptr, ffi.CData) and self._ptr != ffi.NULL:
            C.mag_tensor_decref(self._ptr)
        self._ptr = ffi.NULL

    _DISPATCH = {
        1: C.mag_tensor_create_1d,
        2: C.mag_tensor_create_2d,
        3: C.mag_tensor_create_3d,
        4: C.mag_tensor_create_4d,
        5: C.mag_tensor_create_5d,
        6: C.mag_tensor_create_6d
    }
    assert len(_DISPATCH) == MAX_DIMS

    def _new(self, ctx: Context, *, shape: tuple[int, ...], dtype: DType = DType.F32,
             name: str | None = None) -> None:
        """
        Internal helper to create a new _ptr instance.

        Parameters
        ----------
        ctx : Context
            The magnetron context where this _ptr belongs.
        shape : tuple[int, ...]
            Dimensions of the _ptr.
        dtype : DType, optional
            Data type of the _ptr, by default DType.F32.
        name : str or None, optional
            A friendly name for the _ptr, by default None.
        """
        assert 0 < len(shape) <= MAX_DIMS, f'Invalid number of dimensions: {len(shape)}'
        assert all(0 < dim <= DIM_MAX for dim in shape), 'Invalid dimension size'
        self._ctx = weakref.ref(ctx)
        self._ptr = self._DISPATCH[len(shape)](ctx._ptr, dtype.value, *shape)
        self.name = f'Tensor {self.shape}' if name is None else name

    @classmethod
    def empty(cls, shape: tuple[int, ...], *, dtype: DType = DType.F32, name: str | None = None) -> 'Tensor':
        """
        Creates an empty _ptr with uninitialized data.

        Parameters
        ----------
        shape : tuple[int, ...]
            The shape of the _ptr.
        dtype : DType, optional
            Data type of the _ptr, by default F32.
        name : str or None, optional
            A friendly name for the _ptr, by default None.

        Returns
        -------
        Tensor
            The newly created empty _ptr.
        """
        tensor = cls(None)
        tensor._new(Context.active(), shape=shape, dtype=dtype, name=name)
        return tensor

    @classmethod
    def full(cls, shape: tuple[int, ...], *, fill_value: float, dtype: DType = DType.F32,
             name: str | None = None) -> 'Tensor':
        """
        Creates a _ptr filled with a given constant value.

        Parameters
        ----------
        shape : tuple[int, ...]
            The shape of the _ptr.
        fill_value : float
            The constant value to fill.
        dtype : DType, optional
            Data type, by default F32.
        name : str or None, optional
            A friendly name for the _ptr, by default None.

        Returns
        -------
        Tensor
            The filled _ptr.
        """
        tensor = cls(None)
        tensor._new(Context.active(), shape=shape, dtype=dtype, name=name)
        C.mag_tensor_fill(tensor._ptr, fill_value)
        return tensor

    @classmethod
    def const(cls, data, *, dtype: DType = DType.F32,
              name: str | None = None) -> 'Tensor':
        """
        Creates a _ptr from a nested Python list or a single scalar.

        Parameters
        ----------
        data : scalar or nested list
            Data to create the _ptr from.
        dtype : DType, optional
            Data type, by default F32.
        name : str or None, optional
            A friendly name for the _ptr, by default None.

        Returns
        -------
        Tensor
            The constructed _ptr.
        """

        def flatten_nested_lists(nested):
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
        tensor._new(Context.active(), shape=tuple(shape), dtype=dtype, name=name)
        size: int = len(flattened_data) * ffi.sizeof('float')
        C.mag_tensor_copy_buffer_from(tensor._ptr, ffi.new(f'float[{len(flattened_data)}]', flattened_data), size)
        return tensor

    @classmethod
    def zeros(cls, shape: tuple[int, ...], *, dtype: DType = DType.F32,
              name: str | None = None) -> 'Tensor':
        """
        Creates a _ptr filled with zeros.

        Parameters
        ----------
        shape : tuple[int, ...]
            The shape of the _ptr.
        dtype : DType, optional
            Data type, by default F32.
        name : str or None, optional
            A friendly name for the _ptr, by default None.

        Returns
        -------
        Tensor
            The zero-filled _ptr.
        """
        return cls.full(shape, fill_value=0.0, dtype=dtype, name=name)

    @classmethod
    def uniform(cls, shape: tuple[int, ...], *, interval: (float, float) = (-1.0, 1.0), dtype: DType = DType.F32,
                name: str | None = None) -> 'Tensor':
        """
        Creates a _ptr filled with random uniform values within a given interval.

        Parameters
        ----------
        shape : tuple[int, ...]
            The shape of the _ptr.
        interval : (float, float), optional
            Min and max values for uniform distribution, by default (-1.0, 1.0).
        dtype : DType, optional
            Data type, by default F32.
        name : str or None, optional
            A friendly name for the _ptr, by default None.

        Returns
        -------
        Tensor
            The _ptr filled with random values.
        """
        tensor = cls(None)
        tensor._new(Context.active(), shape=shape, dtype=dtype, name=name)
        if interval[1] < interval[0]:
            interval = (interval[1], interval[0])
        C.mag_tensor_fill_random_uniform(tensor._ptr, interval[0], interval[1])
        return tensor

    @classmethod
    def normal(cls, shape: tuple[int, ...], *, mean: float, stddev: float) -> 'Tensor':
        """
        Creates a _ptr filled with random values from a normal distribution.

        Parameters
        ----------
        shape : tuple[int, ...]
            The shape of the _ptr.
        mean : float
            Mean of the normal distribution.
        stddev : float
            Standard deviation of the normal distribution.

        Returns
        -------
        Tensor
            The _ptr filled with normally distributed values.
        """
        tensor = cls(None)
        tensor._new(Context.active(), shape=shape, dtype=DType.F32)
        C.mag_tensor_fill_random_normal(tensor._ptr, mean, stddev)
        return tensor

    @classmethod
    def load(cls, file_path: str) -> 'Tensor':
        """
        Loads a _ptr from a binary magnetron file.

        Parameters
        ----------
        file_path : str
            Path to the .magnetron file.

        Returns
        -------
        Tensor
            The loaded _ptr.
        """
        assert file_path.endswith('.magnetron'), 'File must be a magnetron file'
        instance = C.mag_tensor_load(Context.active()._ptr, bytes(file_path, 'utf-8'))
        return cls(ptr=instance)

    @classmethod
    def load_image(cls, file_path: str, *,
                   name: str | None = None,
                   channels=ColorChannels.AUTO,
                   resize_to: (int, int) = (0, 0)) -> 'Tensor':
        """
        Loads an image from a file and creates a _ptr.

        Parameters
        ----------
        file_path : str
            Path to the image file.
        name : str or None, optional
            A friendly name for the _ptr, by default None.
        channels : ColorChannels, optional
            Desired color channels to load, by default AUTO.
        resize_to : (int, int), optional
            If not (0,0), resize the image to (width, height), by default (0,0).

        Returns
        -------
        Tensor
            The created image _ptr.
        """
        assert isfile(file_path), f'File not found: {file_path}'
        instance = C.mag_tensor_load_image(Context.active()._ptr, bytes(file_path, 'utf-8'), channels.value,
                                           resize_to[0], resize_to[1])
        tensor = cls(instance)
        if name is not None:
            tensor.name = name
        return tensor

    def print(self, print_header: bool = False, print_data: bool = True) -> None:
        """
        Prints the _ptr metadata and optionally its data.

        Parameters
        ----------
        print_header : bool, optional
            If True, prints a header line, by default False.
        print_data : bool, optional
            If True, prints the _ptr data, by default True.
        """
        C.mag_tensor_print(self._ptr, print_header, print_data)

    @property
    def name(self) -> str:
        """
        Returns the name of the _ptr.

        Returns
        -------
        str
            The _ptr's name.
        """
        return ffi.string(C.mag_tensor_get_name(self._ptr)).decode('utf-8')

    @name.setter
    def name(self, name: str) -> None:
        """
        Sets a name for the _ptr.

        Parameters
        ----------
        name : str
            The new name for the _ptr.
        """
        C.mag_tensor_set_name(self._ptr, bytes(name, 'utf-8'))

    @property
    def rank(self) -> int:
        """
        Returns the rank (number of dimensions) of the _ptr.

        Returns
        -------
        int
            Number of dimensions.
        """
        return C.mag_tensor_rank(self._ptr)

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Returns the shape (dimensions) of the _ptr.

        Returns
        -------
        tuple[int, ...]
            The dimensions of the _ptr.
        """
        return tuple(ffi.unpack(C.mag_tensor_shape(self._ptr), self.rank))

    @property
    def strides(self) -> tuple[int, ...]:
        """
        Returns the strides of the _ptr.

        Returns
        -------
        tuple[int, ...]
            The strides for each dimension.
        """
        return tuple(ffi.unpack(C.mag_tensor_strides(self._ptr), self.rank))

    @property
    def dtype(self) -> DType:
        """
        Returns the data type of the _ptr.

        Returns
        -------
        DType
            The data type, e.g. DType.F32.
        """
        return DType(C.mag_tensor_dtype(self._ptr))

    @property
    def data_ptr(self) -> int:
        """
        Returns the pointer to the _ptr's data buffer.

        Returns
        -------
        int
            Memory address of the _ptr data.
        """
        return int(ffi.cast('uintptr_t', C.mag_tensor_data_ptr(self._ptr)))

    def to_list(self) -> list[float]:
        """
        Returns the _ptr data as a Python list of floats.

        Returns
        -------
        list[float]
            A flat list containing all _ptr elements.
        """
        assert self.dtype == DType.F32, 'Invalid data type'
        return ffi.unpack(ffi.cast('float*', C.mag_tensor_data_ptr(self._ptr)), self.numel)

    def scalar(self) -> float:
        """
        Returns the scalar value of a 0D _ptr.

        Returns
        -------
        float
            The single scalar value.
        """
        assert self.dtype == DType.F32, 'Invalid data type'
        return ffi.unpack(ffi.cast('float*', C.mag_tensor_data_ptr(self._ptr)), 1)[0]

    @property
    def data_size(self) -> int:
        """
        Returns the size of the _ptr buffer in bytes.

        Returns
        -------
        int
            Data size in bytes.
        """
        return C.mag_tensor_data_size(self._ptr)

    @property
    def numel(self) -> int:
        """
        Returns the number of elements in the _ptr.

        Returns
        -------
        int
            Number of elements.
        """
        return C.mag_tensor_numel(self._ptr)

    @property
    def num_rows(self) -> int:
        """
        Returns the number of rows in a 2D _ptr (matrix).

        Returns
        -------
        int
            Number of rows.
        """
        return C.mag_tensor_num_rows(self._ptr)

    @property
    def num_cols(self) -> int:
        """
        Returns the number of columns in a 2D _ptr (matrix).

        Returns
        -------
        int
            Number of columns.
        """
        return C.mag_tensor_num_cols(self._ptr)

    @property
    def is_scalar(self) -> bool:
        """
        Checks if the _ptr is a scalar (0D).

        Returns
        -------
        bool
            True if scalar, otherwise False.
        """
        return C.mag_tensor_is_scalar(self._ptr)

    @property
    def is_vector(self) -> bool:
        """
        Checks if the _ptr is a vector (1D).

        Returns
        -------
        bool
            True if vector, otherwise False.
        """
        return C.mag_tensor_is_vector(self._ptr)

    @property
    def is_matrix(self) -> bool:
        """
        Checks if the _ptr is a matrix (2D).

        Returns
        -------
        bool
            True if matrix, otherwise False.
        """
        return C.mag_tensor_is_matrix(self._ptr)

    @property
    def is_volume(self) -> bool:
        """
        Checks if the _ptr is a volume (3D or higher).

        Returns
        -------
        bool
            True if volume, otherwise False.
        """
        return C.mag_tensor_is_volume(self._ptr)

    @property
    def is_transposed(self) -> bool:
        """
        Checks if the _ptr is transposed.

        Returns
        -------
        bool
            True if transposed, otherwise False.
        """
        return C.mag_tensor_is_transposed(self._ptr)

    @property
    def is_permuted(self) -> bool:
        """
        Checks if the _ptr is permuted.

        Returns
        -------
        bool
            True if permuted, otherwise False.
        """
        return C.mag_tensor_is_permuted(self._ptr)

    def is_shape_eq(self, other: 'Tensor') -> bool:
        """
        Checks if two tensors have the same shape.

        Parameters
        ----------
        other : Tensor
            Another _ptr to compare shape.

        Returns
        -------
        bool
            True if shapes match, otherwise False.
        """
        return C.mag_tensor_is_shape_eq(self._ptr, other._ptr)

    def are_strides_eq(self, other: 'Tensor') -> bool:
        """
        Checks if two tensors have identical strides.

        Parameters
        ----------
        other : Tensor
            Another _ptr to compare strides.

        Returns
        -------
        bool
            True if strides match, otherwise False.
        """
        return C.mag_tensor_are_strides_eq(self._ptr, other._ptr)

    def can_broadcast(self, other: 'Tensor') -> bool:
        """
        Checks if `other` _ptr can be broadcasted to the shape of this _ptr.

        Parameters
        ----------
        other : Tensor
            Another _ptr.

        Returns
        -------
        bool
            True if broadcastable, otherwise False.
        """
        return C.mag_tensor_can_broadcast(self._ptr, other._ptr)

    @property
    def width(self) -> int:
        """
        Returns the width of an image _ptr (assumes layout: CxHxW).

        Returns
        -------
        int
            Width dimension.
        """
        return self.shape[2]

    @property
    def height(self) -> int:
        """
        Returns the height of an image _ptr (assumes layout: CxHxW).

        Returns
        -------
        int
            Height dimension.
        """
        return self.shape[1]

    @property
    def channels(self) -> int:
        """
        Returns the number of channels in an image _ptr (assumes layout: CxHxW).

        Returns
        -------
        int
            Number of channels.
        """
        return self.shape[0]

    @property
    def is_contiguous(self) -> bool:
        """
        Checks if the _ptr is contiguous in memory.

        Returns
        -------
        bool
            True if contiguous, otherwise False.
        """
        return C.mag_tensor_is_contiguous(self._ptr)

    def is_close(self, other: 'Tensor', eps: float = -1.0, print_eq_percent: bool = False) -> (bool, float):
        """
        Checks if the _ptr is close to another _ptr within a given epsilon.

        Parameters
        ----------
        other : Tensor
            Another _ptr to compare to.
        eps : float, optional
            Epsilon tolerance. If -1.0, uses a default internal value.
        print_eq_percent : bool, optional
            If True, prints the percentage of equal elements.

        Returns
        -------
        (bool, float)
            A tuple (close_status, eq_percent).
            close_status is True if the tensors are close.
            eq_percent is the percentage of approximately equal elements.
        """
        percent_eq = ffi.new('double[1]')
        is_eq = C.mag_tensor_is_close(self._ptr, other._ptr, eps, percent_eq)
        if print_eq_percent:
            print(f'Tensors are close: {is_eq}, Percent equal: {percent_eq[0]:.2f}%')
        return is_eq, percent_eq[0]

    def draw_box(self, p1: (int, int), p2: (int, int), width: int = 2, rgb: int = 0xffffff):
        """
        Draws a rectangular box on an image _ptr.

        Parameters
        ----------
        p1 : (int, int)
            Top-left corner coordinates (x, y).
        p2 : (int, int)
            Bottom-right corner coordinates (x, y).
        width : int, optional
            Line width of the box, by default 2.
        rgb : int, optional
            24-bit RGB color (e.g. 0xFFFFFF for white), by default 0xffffff.
        """
        assert p2[0] > p1[0] and p2[1] > p1[1] and width > 0
        C.mag_tensor_img_draw_box(self._ptr, p1[0], p1[1], p2[0], p2[1], width, rgb & 0xffffff)

    def draw_text(self, p: (int, int), size: int, txt: str, rgb: int = 0xffffff):
        """
        Draws text on an image _ptr.

        Parameters
        ----------
        p : (int, int)
            Position to draw text (x, y).
        size : int
            Font size.
        txt : str
            The text to draw.
        rgb : int, optional
            24-bit RGB color, by default white (0xffffff).
        """
        C.mag_tensor_img_draw_text(self._ptr, p[0], p[1], size, rgb & 0xffffff, bytes(txt, 'utf-8'))

    def save(self, file_path: str) -> None:
        """
        Saves the _ptr to a binary magnetron file.

        Parameters
        ----------
        file_path : str
            File path to save the _ptr. Appends '.magnetron' if not present.
        """
        if not file_path.endswith('.magnetron'):
            file_path += '.magnetron'
        C.mag_tensor_save(self._ptr, bytes(file_path, 'utf-8'))

    def save_image(self, file_path: str) -> None:
        """
        Saves a 3D image _ptr as a JPG image file.

        Parameters
        ----------
        file_path : str
            File path to save the image.

        Raises
        ------
        AssertionError
            If _ptr is not 3D or channel count is not supported.
        """
        assert self.rank == 3, 'Tensor must be a 3D image _ptr'
        assert self.channels in (1, 3, 4), 'Invalid number of color channels'
        C.mag_tensor_save_image(self._ptr, bytes(file_path, 'utf-8'))

    def clone(self) -> 'Tensor':
        """
        Creates a new _ptr with the same data as this one (deep copy).

        Returns
        -------
        Tensor
            A cloned _ptr.
        """
        return Tensor(C.mag_clone(self._ptr))

    def view(self) -> 'Tensor':
        """
        Creates a view of the _ptr that shares underlying data (shallow copy).

        Returns
        -------
        Tensor
            A view _ptr.
        """
        return Tensor(C.mag_view(self._ptr))

    def transpose(self) -> 'Tensor':
        """
        Transposes the _ptr (swaps the last two dimensions).

        Returns
        -------
        Tensor
            A transposed _ptr.
        """
        return Tensor(C.mag_transpose(self._ptr))

    def permute(self, axes: tuple[int, ...]) -> 'Tensor':
        """
        Permutes the dimensions of the _ptr.

        Parameters
        ----------
        axes : tuple[int, ...]
            A tuple specifying the permutation of axes.

        Returns
        -------
        Tensor
            A _ptr with permuted dimensions.
        """
        assert len(axes) == MAX_DIMS, f'Invalid number of axes: {axes}'
        for i in range(MAX_DIMS):
            assert 0 <= axes[i] < MAX_DIMS
            for j in range(i + 1, MAX_DIMS):
                assert axes[i] != axes[j], f'Duplicate axis: {axes[i]}'
        return Tensor(C.mag_permute(self._ptr, *axes))

    def mean(self) -> 'Tensor':
        """Computes the mean of all elements in the _ptr."""
        return Tensor(C.mag_mean(self._ptr))

    def min(self) -> 'Tensor':
        """Computes the minimum value in the _ptr."""
        return Tensor(C.mag_min(self._ptr))

    def max(self) -> 'Tensor':
        """Computes the maximum value in the _ptr."""
        return Tensor(C.mag_max(self._ptr))

    def sum(self) -> 'Tensor':
        """Computes the sum of all elements in the _ptr."""
        return Tensor(C.mag_sum(self._ptr))

    def abs(self) -> 'Tensor':
        """Computes element-wise absolute value."""
        return Tensor(C.mag_abs(self._ptr))

    def abs_(self) -> 'Tensor':
        """In-place element-wise absolute value."""
        return Tensor(C.mag_abs_(self._ptr))

    def neg(self) -> 'Tensor':
        """Computes element-wise negation."""
        return Tensor(C.mag_neg(self._ptr))

    def neg_(self) -> 'Tensor':
        """In-place element-wise negation."""
        return Tensor(C.mag_neg_(self._ptr))

    def __neg__(self) -> 'Tensor':
        """Overloads unary negation: -X."""
        return self.neg()

    def log(self) -> 'Tensor':
        """Computes element-wise natural logarithm."""
        return Tensor(C.mag_log(self._ptr))

    def log_(self) -> 'Tensor':
        """In-place element-wise natural logarithm."""
        return Tensor(C.mag_log_(self._ptr))

    def sqr(self) -> 'Tensor':
        """Computes element-wise square of values."""
        return Tensor(C.mag_sqr(self._ptr))

    def sqr_(self) -> 'Tensor':
        """In-place element-wise square of values."""
        return Tensor(C.mag_sqr_(self._ptr))

    def sqrt(self) -> 'Tensor':
        """Computes element-wise square root."""
        return Tensor(C.mag_sqrt(self._ptr))

    def sqrt_(self) -> 'Tensor':
        """In-place element-wise square root."""
        return Tensor(C.mag_sqrt_(self._ptr))

    def sin(self) -> 'Tensor':
        """Computes element-wise sine."""
        return Tensor(C.mag_sin(self._ptr))

    def sin_(self) -> 'Tensor':
        """In-place element-wise sine."""
        return Tensor(C.mag_sin_(self._ptr))

    def cos(self) -> 'Tensor':
        """Computes element-wise cosine."""
        return Tensor(C.mag_cos(self._ptr))

    def cos_(self) -> 'Tensor':
        """In-place element-wise cosine."""
        return Tensor(C.mag_cos_(self._ptr))

    def heaviside_step(self) -> 'Tensor':
        """Computes element-wise Heaviside step function."""
        return Tensor(C.mag_step(self._ptr))

    def heaviside_step_(self) -> 'Tensor':
        """In-place element-wise Heaviside step function."""
        return Tensor(C.mag_step_(self._ptr))

    def softmax(self, derivative: bool = False) -> 'Tensor':
        """
        Applies softmax or its derivative on the _ptr.

        Parameters
        ----------
        derivative : bool, optional
            If True, computes softmax derivative instead of softmax, by default False.

        Returns
        -------
        Tensor
            The transformed _ptr.
        """
        return Tensor(C.mag_softmax_dv(self._ptr) if derivative else C.mag_softmax(self._ptr))

    def softmax_(self, derivative: bool = False) -> 'Tensor':
        """In-place softmax or softmax derivative."""
        return Tensor(C.mag_softmax_dv_(self._ptr) if derivative else C.mag_softmax_(self._ptr))

    def sigmoid(self, derivative: bool = False) -> 'Tensor':
        """
        Applies sigmoid or its derivative on the _ptr.

        Parameters
        ----------
        derivative : bool, optional
            If True, computes sigmoid derivative, by default False.

        Returns
        -------
        Tensor
            The transformed _ptr.
        """
        return Tensor(C.mag_sigmoid_dv(self._ptr) if derivative else C.mag_sigmoid(self._ptr))

    def sigmoid_(self, derivative: bool = False) -> 'Tensor':
        """In-place sigmoid or sigmoid derivative."""
        return Tensor(C.mag_sigmoid_dv_(self._ptr) if derivative else C.mag_sigmoid_(self._ptr))

    def hard_sigmoid(self) -> 'Tensor':
        """Applies hard sigmoid to the _ptr."""
        return Tensor(C.mag_hard_sigmoid(self._ptr))

    def hard_sigmoid_(self) -> 'Tensor':
        """In-place hard sigmoid."""
        return Tensor(C.mag_hard_sigmoid_(self._ptr))

    def silu(self, derivative: bool = False) -> 'Tensor':
        """
        Applies SiLU (Sigmoid-Weighted Linear Unit) or its derivative.

        Parameters
        ----------
        derivative : bool, optional
            If True, applies SiLU derivative, by default False.

        Returns
        -------
        Tensor
            The transformed _ptr.
        """
        return C.mag_silu_dv(self._ptr) if derivative else C.mag_silu(self._ptr)

    def silu_(self, derivative: bool = False) -> 'Tensor':
        """In-place SiLU or SiLU derivative."""
        return Tensor(C.mag_silu_dv_(self._ptr) if derivative else C.mag_silu_(self._ptr))

    def tanh(self, derivative: bool = False) -> 'Tensor':
        """
        Applies hyperbolic tangent or its derivative.

        Parameters
        ----------
        derivative : bool, optional
            If True, computes tanh derivative, by default False.

        Returns
        -------
        Tensor
            The transformed _ptr.
        """
        return Tensor(C.mag_tanh_dv(self._ptr) if derivative else C.mag_tanh(self._ptr))

    def tanh_(self, derivative: bool = False) -> 'Tensor':
        """In-place tanh or tanh derivative."""
        return Tensor(C.mag_tanh_dv_(self._ptr) if derivative else C.mag_tanh_(self._ptr))

    def relu(self, derivative: bool = False) -> 'Tensor':
        """
        Applies ReLU (Rectified Linear Unit) or its derivative.

        Parameters
        ----------
        derivative : bool, optional
            If True, computes ReLU derivative, by default False.

        Returns
        -------
        Tensor
            The transformed _ptr.
        """
        return Tensor(C.mag_relu_dv(self._ptr) if derivative else C.mag_relu(self._ptr))

    def relu_(self, derivative: bool = False) -> 'Tensor':
        """In-place ReLU or ReLU derivative."""
        return Tensor(C.mag_relu_dv_(self._ptr) if derivative else C.mag_relu_(self._ptr))

    def gelu(self, derivative: bool = False) -> 'Tensor':
        """
        Applies GELU (Gaussian Error Linear Unit) or its derivative.

        Parameters
        ----------
        derivative : bool, optional
            If True, computes GELU derivative, by default False.

        Returns
        -------
        Tensor
            The transformed _ptr.
        """
        return Tensor(C.mag_gelu_dv(self._ptr) if derivative else C.mag_gelu(self._ptr))

    def gelu_(self, derivative: bool = False) -> 'Tensor':
        """In-place GELU or GELU derivative."""
        return Tensor(C.mag_gelu_dv_(self._ptr) if derivative else C.mag_gelu_(self._ptr))

    def __add__(self, other: object | int | float) -> 'Tensor':
        """Element-wise addition with another _ptr or scalar."""
        return Tensor(C.mag_add(self._ptr, other._ptr) if isinstance(other, Tensor) else C.mag_adds(self._ptr,
                                                                                                    float(other)))

    def __iadd__(self, other: object | int | float) -> 'Tensor':
        """In-place element-wise addition."""
        return Tensor(C.mag_add_(self._ptr, other._ptr) if isinstance(other, Tensor) else C.mag_adds_(self._ptr,
                                                                                                      float(other)))

    def __sub__(self, other: object | int | float) -> 'Tensor':
        """Element-wise subtraction with another _ptr or scalar."""
        return Tensor(C.mag_sub(self._ptr, other._ptr) if isinstance(other, Tensor) else C.mag_subs(self._ptr,
                                                                                                    float(other)))

    def __isub__(self, other: object | int | float) -> 'Tensor':
        """In-place element-wise subtraction."""
        return Tensor(C.mag_sub_(self._ptr, other._ptr) if isinstance(other, Tensor) else C.mag_subs_(self._ptr,
                                                                                                      float(other)))

    def __mul__(self, other: object | int | float) -> 'Tensor':
        """Element-wise multiplication with another _ptr or scalar."""
        return Tensor(C.mag_mul(self._ptr, other._ptr) if isinstance(other, Tensor) else C.mag_muls(self._ptr,
                                                                                                    float(other)))

    def __imul__(self, other: object | int | float) -> 'Tensor':
        """In-place element-wise multiplication."""
        return Tensor(C.mag_mul_(self._ptr, other._ptr) if isinstance(other, Tensor) else C.mag_muls_(self._ptr,
                                                                                                      float(other)))

    def __truediv__(self, other: object | int | float) -> 'Tensor':
        """Element-wise division with another _ptr or scalar."""
        return Tensor(C.mag_div(self._ptr, other._ptr) if isinstance(other, Tensor) else C.mag_divs(self._ptr,
                                                                                                    float(other)))

    def __itruediv__(self, other: object | int | float) -> 'Tensor':
        """In-place element-wise division."""
        return Tensor(C.mag_div_(self._ptr, other._ptr) if isinstance(other, Tensor) else C.mag_divs_(self._ptr,
                                                                                                      float(other)))

    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        """Matrix multiplication with another _ptr: A @ B."""
        return Tensor(C.mag_matmul(self._ptr, other._ptr))

    def __imatmul__(self, other: 'Tensor') -> 'Tensor':
        """In-place matrix multiplication: A @= B."""
        return Tensor(C.mag_matmul_(self._ptr, other._ptr))

    def __eq__(self, other: 'Tensor') -> bool:
        """
        Checks if two tensors have identical data and shape.

        Parameters
        ----------
        other : Tensor
            Another _ptr to compare.

        Returns
        -------
        bool
            True if equal, otherwise False.
        """
        return C.mag_tensor_eq(self._ptr, other._ptr)

    def __str__(self) -> str:
        """String representation prints the _ptr header and data."""
        self.print(True, True)
        return ''

    def __getitem__(self, indices: int | tuple[int, ...]) -> float:
        """Retrieve a value via virtual or physical indexing."""
        if isinstance(indices, int):
            return C.mag_tensor_get_scalar_virtual_index(self._ptr, indices)
        elif isinstance(indices, tuple):
            idx = indices + (0,) * (6 - len(indices))
            return C.mag_tensor_get_scalar_physical_index(
                self._ptr,
                idx[0],
                idx[1],
                idx[2],
                idx[3],
                idx[4],
                idx[5],
            )
        else:
            raise TypeError("Indices must be an int or a tuple of ints.")

    def __setitem__(self, indices: int | tuple[int, ...], value: float) -> None:
        """Set a value via virtual or physical indexing."""
        if isinstance(indices, int):
            C.mag_tensor_set_scalar_virtual_index(self._ptr, indices, float(value))
        elif isinstance(indices, tuple):
            idx = indices + (0,) * (6 - len(indices))
            C.mag_tensor_set_scalar_physical_index(
                self._ptr,
                idx[0],
                idx[1],
                idx[2],
                idx[3],
                idx[4],
                idx[5],
                float(value),
            )
        else:
            raise TypeError("Indices must be an int or a tuple of ints.")
