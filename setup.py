from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy
import os


# Compiler arguments
extra_compile_args = ["-O3", "-ffast-math", "-march=native"]


def read_version():
    version_file = os.path.join("src", "bayesbay", "_version.py")
    with open(version_file, "r") as f:
        lines = f.read()
    for line in lines.split("\n"):
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


ext_modules = [
    Extension(
        name="bayesbay._utils_1d",
        sources=["src/bayesbay/_utils_1d.pyx"],
        extra_compile_args=extra_compile_args,
        language="c++",
    )
]

setup(
    name="bayesbay",
    version=read_version(),
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.22",
        "matplotlib>=3.0.0",
    ],
    ext_modules=cythonize(ext_modules, language_level="3"),
    include_dirs=[numpy.get_include()],
)
