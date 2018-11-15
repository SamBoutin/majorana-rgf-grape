"""
This file is part of the rgf_grape python package.
Copyright (C) 2017-2018 S. Boutin
For details of the rgf_grape algorithm and applications see:
S. Boutin, J. Camirand Lemyre, and I. Garate, Majorana bound state engineering 
via efficient real-space parameter optimization, ArXiv 1804.03170 (2018).

parameters.py: 
Small wrapper around the ConfigParser python class for the reading and 
writing parameter files.
"""

import sys
import json
from configparser import ConfigParser
import numpy as np

class Parameters(ConfigParser):
    def __init__(self):
        ConfigParser.__init__(self)

    def _load_config(self, file):
        self.read(file)

    def print_config(self, output=sys.stdout):
        self.write(output)

    def saveas(self, file_name):
        with open(file_name, 'w') as cfgfile:
            self.print_config(cfgfile)

    def get_list(self, section, option, default=None):
        string = self[section].get(option)
        # If this is not a list or a string turns it into one.
        if string is None:
            string = '[]'
            if default is not None:
                string = '[{}]'.format(default)
        if string[0] is not '[':
            string = '[{}]'.format(string)
        # Use lower case version of string since JSON requires booleans to be
        # true/false while python uses True/False.
        return json.loads(string.lower())

    def get_complex(self, section, option):
        ls = self.get_list(section, option)
        if len(ls)==1:
            return ls[0]
        if len(ls)==2:
            return ls[0] + 1.j*ls[1]
        print('Trying to return a list of more then two elements to a complex number.')
        sys.exit(-1)

    def _getfloat(self, section, index):
        return eval(self.get(section, index))

    def set(self, section, option, value):
        super(ConfigParser, self).set(section.upper(), option, str(value))

    def verify_list_length(self, arr, n):
        if len(arr) != n:
            if len(arr) == 1:
                arr *= n
            else:
                print('Error in config file:')
                print('List length does not match what is expected')
                sys.exit()
        return arr