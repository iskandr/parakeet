#!/usr/bin/env python

from setuptools import setup, find_packages
import os 

import parakeet.package_info

readme_filename = os.path.join(os.path.dirname(__file__), 'README.md')
with open(readme_filename, 'r') as f:
  readme = f.read()
  
try:
  import pypandoc
  readme = pypandoc.convert(readme, to='rst', format='md')
except:
  print "Conversion of long_description from markdown to reStructuredText failed, skipping..."

setup(
    name="parakeet",
    description="Runtime compiler for numerical Python.",
    long_description=readme,
    classifiers=['Development Status :: 3 - Alpha',
                 'Topic :: Software Development :: Libraries',
                 'License :: OSI Approved :: BSD License',
                 'Intended Audience :: Developers',
                 'Programming Language :: Python :: 2.7',
                 ],
    author="Alex Rubinsteyn",
    author_email="alexr@cs.nyu.edu",
    license="BSD",
    version=parakeet.package_info.__version__,
    url=parakeet.package_info.__website__, 
    download_url = 'https://github.com/iskandr/parakeet/releases', 
    packages=find_packages() + ['parakeet.test', 'parakeet.benchmarks', 'parakeet.examples'], 
    package_dir={ 
                  'parakeet.benchmarks' : './benchmarks', 
                  'parakeet.test' : './test',
                  'parakeet.examples' : './examples', 
                },
    install_requires=[
      'numpy>=1.7',       
      'dsltools',
      #'appdirs', 
      # LLVM is optional as long as you use the C backend 
      # 'llvmpy',
    ])
