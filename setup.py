#!/usr/bin/env python

from setuptools import setup, Extension

import sys

# what I *should* do is either make our use of pthreads work on Mac OS X or
# at least do the OS configuation in some structured way that I'm sure setuptools
# allows. But instead, I'm just hackishly excluding the extension if the OS seems to 
# be a Mac. 

extension_modules = []
if sys.platform != 'darwin':
  runtime_ext = Extension('parakeet.runtime._runtime', 
        sources = [
          './parakeet/runtime/thread_pool.c',
          './parakeet/runtime/job.c',
        ], 
        extra_link_args = ['-Wl', '-Bsymbolic'])
  extension_modules.append(runtime_ext)
  

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
    author="Alex Rubinsteyn & Eric Hielscher",
    author_email="alexr@cs.nyu.edu",
    license="BSD",
    version="0.1",
    url="http://github.com/iskandr/parakeet",
    # scripts = ['scripts/cloudp'],
    packages=[ 'parakeet', 'tests', 'data' ],
    package_dir={ '' : '.' },
    requires=[
      'llvmpy', 
      'numpy', 
      'scipy',
    ],
    ext_modules = extension_modules)
