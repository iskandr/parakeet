#!/usr/bin/env python

from setuptools import setup, Extension, find_packages

import sys

import parakeet 

setup(
    name="parakeet",
    description="Runtime compiler for flight and amusement.",
    long_description='''
Parakeet
=========

An adorable bird that will make your children laugh and spontaneously combust.

''',
    classifiers=['Development Status :: 3 - Alpha',
                 'Topic :: Software Development :: Libraries',
                 'License :: OSI Approved :: BSD License',
                 'Intended Audience :: Developers',
                 'Programming Language :: Python :: 2.7',
                 ],
    author="Alex Rubinsteyn",
    author_email="alexr@cs.nyu.edu",
    license="BSD",
    version=parakeet.__version__,
    url=parakeet.__website__, 
    packages=find_packages() + ['parakeet.test', 'parakeet.benchmarks', 'parakeet.examples'], 
    #[ 'parakeet', 
    #           'parakeet.analysis', 
    #           'parakeet.frontend', 
    #           'parakeet.lib', 
    #           'parakeet.transforms', 
    #           'parakeet.type_inference', 
    #           'parakeet.examples', 
    #           'parakeet.test' ],
    
    package_dir={ 
                  'parakeet.benchmarks' : './benchmarks', 
                  'parakeet.examples' : './examples', 
                  'parakeet.test' : './test' 
                },
    requires=[

      'numpy', 
      'scipy',
      'treelike'
      # LLVM is optional as long as you use the C backend 
      # 'llvmpy'   
    ])
