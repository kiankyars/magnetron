# (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>
# To install locally, use: pip3 install . --force-reinstall

import os
import subprocess
import multiprocessing
import shutil
import sys

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

BUILD_RELEASE: bool = True
CMAKE_ROOT: str = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # Root directory of the CMake project
NUM_JOBS: int = max(multiprocessing.cpu_count() - 1, 1)  # Use all but one core


def get_dll_extension() -> str:
    platform = sys.platform
    if platform.startswith('linux'):
        return '.so'
    elif platform == 'darwin':
        return '.dylib'
    elif platform == 'win32':
        return '.dll'
    else:
        raise RuntimeError(f'Unsupported platform: {platform}. Please report this issue to the author.')


class BuildException(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class CMakeBuildExtension(Extension):
    def __init__(self, name, root_dir: str = ''):
        super().__init__(name, sources=[])
        self.root_dir = os.path.abspath(root_dir)


class CMakeBuildExecutor(build_ext):
    def initialize_options(self):
        super().initialize_options()

    def run(self):
        # Ensure CMake is available
        try:
            print(subprocess.check_output(['cmake', '--version']))
        except OSError:
            raise BuildException('CMake must be installed to build the magnetron binaries from source. Please install CMake and try again.')
        super().run()
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        # 1) Prepare build directory
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        # 2) Tell CMake to put the .so into build_lib/magnetron/
        lib_output_dir = os.path.abspath(os.path.join(self.build_lib, 'magnetron'))

        cmake_args = [
            '-DMAGNETRON_ENABLE_CUDA=OFF',
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={lib_output_dir}',
            f'-DCMAKE_BUILD_TYPE={"Release" if BUILD_RELEASE else "Debug"}',
        ]
        build_args = [
            '--target',
            'magnetron',
            f'-j{NUM_JOBS}',
            '-v',
        ]

        # 3) Run CMake configure + build
        subprocess.check_call(['cmake', ext.root_dir] + cmake_args, cwd=self.build_temp)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

        # 4) Copy/rename from CMake output into the Python extension path
        #    CMake put it at lib_output_dir/libmagnetron.so
        built_lib = os.path.join(lib_output_dir, 'libmagnetron' + get_dll_extension())
        if not os.path.isfile(built_lib):
            raise BuildException(f'Expected built library not found at {built_lib}')

        #    setuptools wants something like build/lib/magnetron.cpython-311-x86_64-linux-gnu.so
        dest_path = self.get_ext_fullpath(ext.name)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copyfile(built_lib, dest_path)


# Read install_requires from requirements.txt if present
lib_folder = os.path.dirname(os.path.realpath(__file__))
requirement_path = os.path.join(lib_folder, 'requirements.txt')
install_requires = []
if os.path.isfile(requirement_path):
    with open(requirement_path) as f:
        install_requires = f.read().splitlines()

# setuptools entry point
setup(
    name='magnetron',
    version='0.1.0',
    author='Mario Sieg',
    author_email='mario.sieg.64@gmail.com',
    description='A lightweight machine learning library with GPU support.',
    long_description='A lightweight machine learning library with GPU support.',
    packages=[
        'magnetron',
        'magnetron.nn',
    ],
    package_dir={'': 'src'},
    package_data={'magnetron': ['*.dylib', '*.so', '*.dll']},
    include_package_data=True,
    ext_modules=[CMakeBuildExtension('magnetron', root_dir=CMAKE_ROOT)],
    cmdclass={'build_ext': CMakeBuildExecutor},
    zip_safe=False,
    install_requires=install_requires,
)
