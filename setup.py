import os
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="cardio_rl",
    packages=find_packages(),
    install_requires=["numpy>=1.21.0", "gymnasium>=0.28.0", "tensorboard", "tqdm"],
    extras_require={
        "exp": [
            # Optional dependencies for testing go here
            "hydra-core",
            "rich",
        ]
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
