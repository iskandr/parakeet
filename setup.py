#!/usr/bin/env python

from setuptools import setup, Extension

import sys


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
    version="0.14",
    url="http://github.com/iskandr/parakeet",
    packages=[ 'parakeet', 'tests', 'data' , 'examples'],
    package_dir={ '' : '.' },
    requires=[
      'llvmpy', 
      'numpy', 
      'scipy',
      'loopjit', 
      'shapely',
      'ndtypes',
      'treelike'
    ])
