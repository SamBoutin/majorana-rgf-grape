"""
This file is part of the rgf_grape python package.
Copyright (C) 2017-2018 S. Boutin
For details of the rgf_grape algorithm and applications see:
S. Boutin, J. Camirand Lemyre, and I. Garate, Majorana bound state engineering 
via efficient real-space parameter optimization, ArXiv 1804.03170 (2018).
"""

import rgf_grape.utilities
rgf_grape.utilities.print_git_version()
rgf_grape.utilities.init_numpy_seed()

import rgf_grape.rgf
import rgf_grape.optimization
import rgf_grape.independent_spins

try:
	import rgf_grape.data_analysis
except ImportError:
	print("Data analysis functions not available.")
	print("Are Pandas, holoviews and matplotlib installed?")