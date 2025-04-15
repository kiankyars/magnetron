# (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

from pathlib import Path
from magnetron import Tensor, Context
from magnetron._bootstrap import load_native_module

_ffi, _C = load_native_module()

class StorageStream:
    """Reads and writes Magnetron (.mag) storage files."""
    def __init__(self, handle: _ffi.CData | None = None) -> None:
        self._ctx = Context.primary()
        if handle is not None:
            assert handle != _ffi.NULL
            self._ptr = handle
        else:
            self._ptr = _C.mag_storage_stream_new(self._ctx.native_ptr)

    def __del__(self) -> None:
        _C.mag_storage_stream_close(self._ptr)
        self._ptr = None

    @property
    def native_ptr(self) -> _ffi.CData:
        return self._ptr

    @classmethod
    def open(cls, file_path: Path | str) -> 'StorageStream':
        """Opens a Magnetron storage file for reading."""
        if isinstance(file_path, str):
            file_path = Path(file_path)
        handle = _C.mag_storage_stream_deserialize(Context.primary().native_ptr, str(file_path).encode('utf-8'))
        if handle == _ffi.NULL:
            raise RuntimeError(f'Failed to open storage stream from file: {file_path}')
        return cls(handle)

    def put(self, key: str, tensor: Tensor) -> None:
        """Adds a tensor with unique key to the storage stream."""
        assert tensor.native_ptr is not None and tensor.native_ptr != _ffi.NULL
        key_utf8: bytes = key.encode('utf-8')
        if _C.mag_storage_stream_get_tensor(self._ptr, key_utf8) != _ffi.NULL:
            raise RuntimeError(f'Tensor with key {key} already exists in storage stream')
        if not _C.mag_storage_stream_put_tensor(self._ptr, key_utf8, tensor.native_ptr):
            raise RuntimeError('Failed to put tensor into storage stream')

    def get(self, key: str) -> Tensor | None:
        """Retrieves a tensor from the storage stream."""
        handle = _C.mag_storage_stream_get_tensor(self._ptr, key.encode('utf-8'))
        if handle == _ffi.NULL:
            return None
        return Tensor(handle)

    def serialize(self, file_path: Path) -> None:
        if not _C.mag_storage_stream_serialize(self._ptr, str(file_path).encode('utf-8')):
            raise RuntimeError(f'Failed to serialize storage stream to file: {file_path}')
