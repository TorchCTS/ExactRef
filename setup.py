"""Build exactref's C extension, ``exactref._c``: one abi3 module for CPython 3.10 and newer (not free-threaded
Python, which the 3.10 limited API does not support)."""

import glob

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class BuildExt(build_ext):
    """The extension's own optimization level, whatever the building Python's flags: at -O2
    (Ubuntu's Python) GCC leaves the kernels' term loops about ten times slower than at -O3. Other
    than MSVC, compilers hide the C's functions (the module's init stays exported): with -fPIC GCC
    neither inlines nor binds directly a function another library could interpose, which costs the
    kernels 3 to 9 percent on Linux."""

    def build_extensions(self):
        flags = ["/O2"] if self.compiler.compiler_type == "msvc" else ["-O3", "-fvisibility=hidden"]
        for ext in self.extensions:
            ext.extra_compile_args = flags
        super().build_extensions()


setup(
    ext_modules=[
        Extension(
            "exactref._c",
            sources=sorted(glob.glob("csrc/*.c")),
            depends=sorted(glob.glob("csrc/*.h")),
            define_macros=[("Py_LIMITED_API", "0x030A0000")],
            py_limited_api=True,
        )
    ],
    cmdclass={"build_ext": BuildExt},
    options={"bdist_wheel": {"py_limited_api": "cp310"}},
)
