from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy
import os
import platform

# Compiler arguments
extra_compile_args = ["-O3", "-ffast-math", "-march=native", "-fopenmp"]

# Platform-specific adjustments
platform_name = platform.system()
extra_link_args = ["-fopenmp"]


def read_version():
    version_file = os.path.join("src", "bayesbridge", "_version.py")
    with open(version_file, "r") as f:
        lines = f.read()
    for line in lines.split("\n"):
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


ext_modules = [
    Extension(
        name="bayesbridge._utils_bayes",
        sources=["src/bayesbridge/_utils_bayes.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++",
    )
]

setup(
    name="bayesbridge",
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
