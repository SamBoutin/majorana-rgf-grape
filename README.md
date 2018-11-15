# majorana-rgf-grape
Efficient real-space parameter optimization for Majorana bound state engineering

## Description
RGF-GRAPE: algorithm for the efficient gradient-based optimization of real-space 
parameter profiles for the engineering of robust Majorana bound states.

S. Boutin, J. Camirand Lemyre, and I. Garate.  
Majorana bound state engineering via efficient real-space parameter optimization.  
ArXiv 1804.03170 (2018)

Up-to-date code can be found on GitHub:
https://github.com/SamBoutin/majorana-rgf-grape

## Dependencies : 
The lead self-energies are computed using the Kwant package
(see https://kwant-project.org/ for installation)
Other dependencies (cython, numpy, etc.) should be the same as Kwant.

Some Jupyter notebooks, which gives examples of data analysis additionally uses
Pandas, Holoviews and matplotlib packages for data visualization.

## Build
Note: Only Python > 3.5 on a linux platform has been tested.

To build the cython rgf_solver: 
$ python3 setup.py build_ext --inplace

To clean temporary files after build:
$ python3 setup.py clean --all

## Examples
Examples allowing to reproduce the results of the manuscript are presented in 
the examples folder.

To run them, the rgf_grape module is expected to be in the Python path:
$ export PYTHONPATH="$PYTHONPATH:/path/to/rgf_grape/module"

### Additional comments
* The definition of the performance index differs by a sign from the article,
since here we minimize the performance index, while in the text we maximize the
index.
* As this is a research code, all features are not necessarily compatible with
one another (or remains untested). Proceed with caution when considering use cases 
that differ from the examples.
* The RGF-GRAPE package in written so that a single code should be written for a
given type of problem. Any small variations such as parameters can then be passed
through a parameter file read at runtime. In general, the parameter files should
follow the syntax of the ConfigParser python package from which our parameter
object is derived.
