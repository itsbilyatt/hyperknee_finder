from setuptools import setup


# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='HyperkneeFinder',
    version='0.0.1',
    description='Its about a tool for optimizing two inter-dependent parameters.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/itsbilyatt/hyperknee_finder',
    author='Vincenzo Lavorini',
    author_email='vincenzo.lavorini@protonmail.ch',
    #license='BSD 2-clause',
    packages=['hyperkneefinder'],
    install_requires=[
                      'numpy',
                      'matplotlib',
                      'scikit-learn'
                      ],

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)
