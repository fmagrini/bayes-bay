from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy
import os
import platform

# Compiler arguments
extra_compile_args = ["-O3", "-ffast-math", "-march=native", "-fopenmp"]

# Platform-specific adjustments
platform_name = platform.system()
if platform_name.lower() == "darwin":
    src_paths = ["/usr/local", "/opt/homebrew"]
    for src in src_paths:
        gcc_path = os.path.join(src, "Cellar/gcc")
        if not os.path.exists(gcc_path):
            continue
        versions = os.listdir(gcc_path)
        version = max(versions, key=lambda i: int(i.split(".")[0]))
        version_int = version.split(".")[0]
        path = os.path.join(gcc_path, "%s/lib/gcc/%s" % (version, version_int))
        os.environ["CC"] = "gcc-%s" % version_int
        os.environ["CXX"] = "g++-%s" % version_int
        extra_link_args = ["-Wl,-rpath,%s" % path]
else:
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
        name="bayesbridge._utils_1d",
        sources=["src/bayesbridge/_utils_1d.pyx"],
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
