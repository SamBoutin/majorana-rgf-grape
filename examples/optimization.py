import rgf_grape
import sys

if __name__ == '__main__':
    if len(sys.argv) > 1:
        parameter_file = sys.argv[1]
    else:
        parameter_file = 'params.ini'
    rgf_grape.independent_spins.optim.run(parameter_file)
