"""
This file is part of the rgf_grape python package.
Copyright (C) 2017-2018 S. Boutin
For details of the rgf_grape algorithm and applications see:
S. Boutin, J. Camirand Lemyre, and I. Garate, Majorana bound state engineering 
via efficient real-space parameter optimization, ArXiv 1804.03170 (2018).
"""

import sys
import numpy as np

import rgf_grape
from . import independent_spins_wire as indep_spins
import rgf_grape.majorana.optimization_parameters as mp


class Parameters(mp.MajoranaParameters):
    def __init__(self, file_name=None, load_default=True, create_all=True):
        path = rgf_grape.utilities.get_path(__file__)
        mp.MajoranaParameters.__init__(self, file_name, load_default, 
                                    create_all, default_path=path)
        if create_all:
            self.generate()

    def generate(self, cb_name='opt'):
        self.wire = indep_spins.WireIndependentSpins(self)
        self.optimization_params = self._optimization()
        self.optimizer_opts = self._optimizer_options()
        self._initial_vector()
        self.cost_function_args = self._cost_function_args()
        self._set_task()
        self._callback(cb_name)

    def _set_texture_options(self):
        cfg_opt = self['OPTIMIZATION']
        nb = self.wire.nb_textures
        
        d = {}
        l_theta = self.get_list('OPTIMIZATION', 'optMagneticTextures theta')
        l_phi = self.get_list('OPTIMIZATION', 'optMagneticTextures phi')
        d['optThetas'] = self.verify_list_length(l_theta, nb)
        d['optPhis'] = self.verify_list_length(l_phi, nb)

        periods = self.get_list('OPTIMIZATION', 'texture_periods')
        if periods == []:
            periods = [self.wire.N]
        d['periods'] = self.verify_list_length(periods, nb)

        nb_field = np.sum([p*t for (p,t) in zip(periods, d['optThetas'])])
        nb_field += np.sum([p*t for (p,t) in zip(periods, d['optPhis'])])

        nb = sum(d['optThetas'])
        nb += sum(d['optPhis'])
        d['optField'] = (nb > 0)
        return d, nb_field

    def _optimization(self):
        params = {}
        # texture options
        opts, nb_field = self._set_texture_options()
        params['texture_opts'] = opts
        params['nb_field'] = nb_field

        # Mu options
        params['mu_opts'] = self._set_mu_options()
        params['nb_mu'] = params['mu_opts']['nb_mu']

        # Delta options
        params_delta = self._set_delta_options()
        params['delta_opts'] = params_delta
        params['nb_delta'] = params['delta_opts']['period']
        
        params['period'] = np.max([params['nb_delta'], params['nb_mu'], params['nb_field']//2])

        # Update functions :
        self.update_function = indep_spins.updateAll
        self.filter_function = indep_spins.filterGradAll
        params.update(mp.MajoranaParameters._optimization(self))

        return params

    def _initial_vector(self):
        x0_field, bnd_field = self._initial_vector_texture()
        x0_mu, bnd_mu = self._initial_vector_mu()
        x0_delta, bnd_delta = self._initial_vector_delta()

        self.x0 = np.concatenate((x0_field, x0_mu, x0_delta))
        self.bounds = bnd_field + bnd_mu + bnd_delta

    def _initial_vector_texture(self):
        nb = self.wire.nb_textures
        if nb == 0:
            return np.array([]), []
        x_init = []
        for i in range(nb):
            x_init += [self._initial_texture(i)]
        opts = self.optimization_params['texture_opts']
        tmp_args = {'wire': self.wire, 
                    'texture_opts': 
                        {'optPhis': [True]*nb,
                        'optThetas': [True]*nb,
                        'fixed_phases':None,
                        'periods': opts['periods']
                        },
                    'calculatePenalty': False
                    }
        x_init = np.concatenate(x_init)
        indep_spins.updateTextures(x_init, tmp_args)

        # For optimization, we keep only the textures to be optimized:
        x0 = np.array([])
        fixed_phases = []
        optPhis = opts['optPhis']
        optThetas = opts['optThetas']
        for i, (t, f) in enumerate(zip(optThetas, optPhis)):
            ns = opts['periods'][i]
            if t and f:
                x0 = np.append(x0, x_init[i*2*ns:(2*i+1)*ns])
                x0 = np.append(x0, x_init[(i*2+1)*ns:(2*i+2)*ns])
            elif t: #(and not f)
                x0 = np.append(x0, x_init[i*2*ns:(2*i+1)*ns])
                fixed_phases.append(x_init[(i*2+1)*ns:(2*i+2)*ns])
            elif f: # (and not t)
                fixed_phases.append(x_init[i*2*ns:(2*i+1)*ns])
                x0 = np.append(x0, x_init[(i*2+1)*ns:(2*i+2)*ns])
            else: # (not t) and (not f)
                fixed_phases.append(x_init[i*2*ns:(2*i+1)*ns])
                fixed_phases.append(x_init[(i*2+1)*ns:(2*i+2)*ns])
        opts['fixed_phases'] = fixed_phases
        bnds = [(None, None)]*x0.size

        return x0, bnds

    def _initial_texture(self, index):
        section = 'TEXTURE {}'.format(index)
        opts = self.optimization_params['texture_opts']
        initType = self[section].get('initType')
        N = opts['periods'][index]
        if initType == 'spiral' or initType == 'spiral+noise':
            # Spiral parameters :
            spiralPeriod = self[section].getint('period', N)
            phase = self._getfloat(section, 'phase')
            theta = [np.pi / 2.] * N
            phi = [x * 2 * np.pi / spiralPeriod + phase for x in range(0, N)]
            x0 = np.array([theta + phi]).reshape(2 * N)
        if initType == 'spiral+noise':
            noise_amp = self._getfloat(section, 'noise amplitude')
            x0 += (np.random.rand(2 * N) - 0.5)*4.*np.pi*noise_amp

        if initType == 'rnd':
            noise_amp = self._getfloat(section, 'noise amplitude')
            x0 = (np.random.rand(2 * N) - 0.5)*4.*np.pi*noise_amp
            x0[:N] += self._getfloat(section, 'theta')
            x0[N:] += self._getfloat(section, 'phi')
        if initType == 'rndPhi':
            noise_amp = self._getfloat(section, 'noise amplitude')
            x0 = (np.random.rand(2 * N) - 0.5)*4.*np.pi*noise_amp
            x0[:N] *= 0.
            x0[:N] += self._getfloat(section, 'theta')
            x0[N:] += self._getfloat(section, 'phi')
        return x0
