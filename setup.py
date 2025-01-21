#!/usr/bin/env python

import runpy
from distutils.core import setup


__version__ = runpy.run_path("cellseg1/__version__.py")["__version__"]


setup(
    name='cellseg1',
    description='CellSeg1: Robust Cell Segmentation with One Training Image',
    version=__version__,
    author='',
    author_email='',
    url='https://github.com/caroteu/cellseg1',
    packages=['cellseg1'],
)
