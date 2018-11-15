"""
This file is part of the rgf_grape python package.
Copyright (C) 2017-2018 S. Boutin
For details of the rgf_grape algorithm and applications see:
S. Boutin, J. Camirand Lemyre, and I. Garate, Majorana bound state engineering 
via efficient real-space parameter optimization, ArXiv 1804.03170 (2018).
"""

import subprocess
import os
import pickle
import numpy as np
from datetime import datetime


def git_version():
    '''
    Taken from numpy/setup.py
    '''
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                               env=env).communicate()[0]
        return out

    try:
        cwd = os.getcwd()
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        out = _minimal_ext_cmd(['git', 'rev-parse', '--short', 'HEAD'])
        GIT_REVISION = out.strip().decode('ascii')
        os.chdir(cwd)
    except OSError:
        GIT_REVISION = "Unknown"

    return GIT_REVISION


def print_git_version():
    print('Program started on {}'.format(datetime.now()))
    print('Current git version {}'.format(git_version()))


def get_path(file):
    '''
    Calling get_path(__file__) returns the path of the file from which the
    function is called.
    '''
    return os.path.dirname(os.path.abspath(file))


def save_to_file(fname, obj, append=False):
    if append:
        opt = 'ab'
    else:
        opt = 'wb'
    with open(fname, opt) as output:
        pickle.dump(obj, output)


def load_file(fname):
    with open(fname, 'rb') as infile:
        objs = []
        while 1:
            try:
                objs.append(pickle.load(infile))
            except EOFError:
                break
        if len(objs)==1:
            return objs[0]
    return objs


def init_numpy_seed(seed=None, save_to_file=False):
    if seed is None:
        seed = int(np.floor(np.random.rand(1)[0]*1e8))
        print('Randomly chosen seed = ', seed)
    if save_to_file:
        np.savetxt('seed.dat', [[seed]])
    r = np.random.seed(seed)