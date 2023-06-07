import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = 'CardioRL',
    version = '0.0.0',
    author = 'Manus McAuliffe',
    author_email = "mmcaulif@tcd.ie",
    description = ("CardioRL. In development."),    
    license = 'MIT',
    url = 'https://github.com/mmcaulif/GymCardio',
    packages=['src'],
    long_description=read('README.md'),
    classifiers=['Development Status :: 3 - Alpha'],
)