from pathlib import Path
from magnetron._ffi_cdecl_generated import __MAG_CDECLS
import sys

MAG_LIBS = [
    ('win32', 'magnetron.dll'),
    ('linux', 'libmagnetron.so'),
    ('darwin', 'libmagnetron.dylib'),
]

def load_native_module():
    platform = sys.platform
    lib_name = next((lib for os, lib in MAG_LIBS if platform.startswith(os)), None)
    assert lib_name, f"Unsupported platform: {platform}"

    # Locate the library in the package directory
    pkg_path = Path(__file__).parent
    lib_path = pkg_path / lib_name
    assert lib_path.exists(), f"magnetron shared library not found: {lib_path}"

    # Load the library using cffi
    from cffi import FFI
    ffi = FFI()
    ffi.cdef(__MAG_CDECLS)  # Define the C declarations
    lib = ffi.dlopen(str(lib_path))  # Load the shared library
    return ffi, lib