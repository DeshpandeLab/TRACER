"""Build hook for Cython extensions.

Package metadata lives in pyproject.toml. This file exists solely to
compile the .pyx files into C extensions at install time, because the
pyproject-only setuptools backend does not yet handle ext_modules cleanly.
"""
import platform
from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

# OpenMP flags: on macOS clang needs `-Xpreprocessor -fopenmp` + libomp
# (brew-installed at /opt/homebrew/opt/libomp). Elsewhere `-fopenmp`
# is enough. The _cy_reassign kernel needs OpenMP for its prange.
omp_extra_includes: list[str] = []
omp_extra_lib_dirs: list[str] = []
if platform.system() == "Darwin":
    omp_compile_args = ["-Xpreprocessor", "-fopenmp"]
    omp_link_args = ["-lomp"]
    # brew libomp on Apple Silicon
    _brew_libomp = "/opt/homebrew/opt/libomp"
    import os.path as _osp
    if _osp.isdir(_brew_libomp):
        omp_extra_includes.append(f"{_brew_libomp}/include")
        omp_extra_lib_dirs.append(f"{_brew_libomp}/lib")
else:
    omp_compile_args = ["-fopenmp"]
    omp_link_args = ["-fopenmp"]

ext_specs = [
    {"name": "_cy_prune",    "src": "src/tracer/_cy_prune.pyx",    "openmp": False},
    {"name": "_cy_spatial",  "src": "src/tracer/_cy_spatial.pyx",  "openmp": False},
    {"name": "_cy_reassign", "src": "src/tracer/_cy_reassign.pyx", "openmp": True},
]

extensions = []
for spec in ext_specs:
    mod = f"tracer.{spec['name']}"
    if spec["openmp"]:
        extensions.append(Extension(
            mod, [spec["src"]],
            extra_compile_args=omp_compile_args,
            extra_link_args=omp_link_args,
            include_dirs=[np.get_include(), *omp_extra_includes],
            library_dirs=omp_extra_lib_dirs,
        ))
    else:
        extensions.append(Extension(
            mod, [spec["src"]],
            include_dirs=[np.get_include()],
        ))

extensions = cythonize(
    extensions,
    language_level=3,
    compiler_directives={
        "boundscheck": False,
        "wraparound": False,
        "cdivision": True,
        "nonecheck": False,
    },
)

setup(ext_modules=extensions)
