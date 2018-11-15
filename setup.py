"""
This file is part of the rgf_grape python package.
Copyright (C) 2017-2018 S. Boutin
For details of the rgf_grape algorithm and applications see:
S. Boutin, J. Camirand Lemyre, and I. Garate, Majorana bound state engineering 
via efficient real-space parameter optimization, ArXiv 1804.03170 (2018).
"""

import numpy
from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

optim_path = 'rgf_grape/optimization/'
optim_files = [optim_path + '_functionsDefinitions.pyx',
               optim_path + 'fctDefs.cpp']

rgf_path = 'rgf_grape/rgf/'
rgf_files = [rgf_path+'_rgf_solver.pyx', rgf_path+'rgf_utils.cpp']

extensions = [Extension("rgf_grape.optimization._functionsDefinitions",
              optim_files,
                        include_dirs=['.', numpy.get_include()],
                        language="c++",
                        extra_compile_args=["-O3", "-Wall"],
                        extra_link_args=['-g']
                        ),
              Extension("rgf_grape.rgf._rgf_solver", rgf_files,
                        include_dirs=['.', numpy.get_include()],
                        language="c++",
                        extra_compile_args=["-O3", "-Wall"],
                        extra_link_args=['-g']
                        )]

setup(
    ext_modules=cythonize(extensions)
)
