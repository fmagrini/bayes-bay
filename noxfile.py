"""
nox -s build
nox -s tests
nox -s lint
nox -s format_check
nox -s format
nox -s docs
"""

import nox

@nox.session
def build(session):
    session.install("setuptools", "wheel", "Cython", "numpy")
    session.run("python", "setup.py", "sdist", "bdist_wheel")

@nox.session(python=["3.7", "3.8", "3.9", "3.10"])
def tests(session):
    session.install("setuptools", "wheel", "Cython", "numpy", "pytest")
    session.install(".")
    session.run("python", "-m", "pytest", "tests")

@nox.session(python="3.10")
def lint(session):
    session.install("flake8")
    session.run("flake8", "src", "tests")

@nox.session(python="3.10")
def format_check(session):
    session.install("black")
    session.run("black", "--check", "src", "tests")
    
@nox.session(python="3.10")
def format(session):
    session.install("black")
    session.run("black", "src", "tests")

# @nox.session(python="3.10")
# def docs(session):
#     session.install(".")
#     session.install("sphinx")
#     session.run("sphinx-build", "-b", "html", "docs/source", "docs/build")
