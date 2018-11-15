import rgf_grape
import sys
import numpy as np

if __name__ == '__main__':
    if len(sys.argv) > 1:
        parameter_file = sys.argv[1]
    else:
        parameter_file = 'params.ini'
    name = 'texture energies'
    vals = np.linspace(0,0.045,91)
    rgf_grape.independent_spins.optim.sweep_parameter(name, vals, parameter_file)
