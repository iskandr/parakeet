#!/usr/bin/env python

from setuptools import setup, Extension, find_packages
import os 
import sys

import parakeet.version

setup(
    name="parakeet",
    description="Runtime compiler for numerical Python.",
    long_description=open(os.path.join(os.path.dirname(__file__), 'README.md')).read(),
    classifiers=['Development Status :: 3 - Alpha',
                 'Topic :: Software Development :: Libraries',
                 'License :: OSI Approved :: BSD License',
                 'Intended Audience :: Developers',
                 'Programming Language :: Python :: 2.7',
                 ],
    author="Alex Rubinsteyn",
    author_email="alexr@cs.nyu.edu",
    license="BSD",
    version=parakeet.version.__version__,
    url=parakeet.version.__website__, 
    download_url = 'https://github.com/iskandr/parakeet/releases', 
    packages=find_packages() + ['parakeet.test', 'parakeet.benchmarks', 'parakeet.examples'], 
    package_dir={ 
                  'parakeet.benchmarks' : './benchmarks', 
                  'parakeet.examples' : './examples', 
                  'parakeet.test' : './test' 
                },
    install_requires=[
      'numpy>=1.7',       
      'dsltools',
      #'appdirs', 
      # LLVM is optional as long as you use the C backend 
      # 'llvmpy',
      # SciPy is required for some of the tests but due to popular complaints
      # I dropped it as a dependency for Parakeet itself
      # 'scipy',  
    ])
