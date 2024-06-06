# type: ignore

import os

from setuptools import find_packages, setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


# Function to read the requirements.txt file
def read_requirements(name):
    with open(os.path.join("requirements", name)) as f:
        return f.read().splitlines()


setup(
    name="cardio_rl",
    packages=find_packages(),
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "exp": read_requirements("requirements-exp.txt"),
        "dev": read_requirements("requirements-dev.txt"),
        "cpu": read_requirements("requirements-cpu.txt"),
        "gpu": read_requirements("requirements-gpu.txt"),
    },
    description="Cardio RL. In development...",
    author="Manus McAuliffe",
    url="https://github.com/mmcaulif/GymCardio",
    author_email="mmcaulif@tcd.ie",
    license="MIT",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    version="0.0.4",
    python_requires=">=3.7",
    # PyPI package information.
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
