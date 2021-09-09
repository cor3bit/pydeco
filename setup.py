import os
import setuptools


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setuptools.setup(
    name='PyDeCo',
    version='0.0,1',
    author='Nick Korbit',
    description='Decentralized Control Algorithms written in JAX.',
    long_description=read('README.md'),
    packages=setuptools.find_packages(exclude=['examples', 'tests']),
    install_requires=[
        'jax',
    ],
)
