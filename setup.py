import os
import setuptools


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setuptools.setup(
    name='PyDeco',
    version='0.0.2',
    author='Nick Korbit',
    description='Experiments in Decentralized and Distributed Control Algorithms.',
    long_description=read('README.md'),
    packages=setuptools.find_packages(exclude=[
        'artifacts',
        'examples',
        'experiments',
        'tests',
    ]),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'seaborn',
    ],
)
