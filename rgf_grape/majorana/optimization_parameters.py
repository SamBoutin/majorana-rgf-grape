"""
This file is part of the rgf_grape python package.
Copyright (C) 2017-2018 S. Boutin
For details of the rgf_grape algorithm and applications see:
S. Boutin, J. Camirand Lemyre, and I. Garate, Majorana bound state engineering 
via efficient real-space parameter optimization, ArXiv 1804.03170 (2018).
"""

import sys
import os
import numpy as np

import rgf_grape
from rgf_grape import pauliMatrices as pauli
from rgf_grape.optimization.callback import CallbackHelper
from rgf_grape.optimization import functionsDefinitions as fctDef
from rgf_grape.parameters import Parameters

class MajoranaParameters(Parameters):
    def __init__(self, file_name=None, load_default=True, create_all=True, default_path=None):
        Parameters.__init__(self)
        # First reads default parameter file
        # Second, reads local parameter file.
        # This allows to only specify in the local file parameters that differ
        # from defaults. The local file overwrite default if there is a
        # conflict.
        if load_default:
            if default_path is None:
                default_path = rgf_grape.utilities.get_path(__file__)
            default_path += '/defaults.ini'
            self._load_config(default_path)
        if file_name is not None:
            self._load_config(file_name)
        self.set_rng_seed()

    def set_rng_seed(self):
        seed = self['DEFAULT'].getint('seed')
        if seed is not None:
            rgf_grape.utilities.init_numpy_seed(seed)

    def _optimization(self):
        cfg_opt = self['OPTIMIZATION']
        params = {}
        params['calculatePenalty'] = cfg_opt.getboolean('calculatePenalty')
        params['penalty_weight'] = cfg_opt.getfloat('penalty_weight')
        scale_penalty = cfg_opt.getboolean('scale_penalty_by_chain_length')
        if scale_penalty:
            nb_sites = self.wire.N
            params['penalty_weight'] /= nb_sites

        params['useBH'] = cfg_opt.getboolean('basinhopping optimization')
        return params

    def _set_task(self):
        # Default options:
        self.cost_function = fctDef.performance_index
        self.grad_function = fctDef.rgf_grape_grad

        if 'TASK' not in self.sections():
            return
        fd = False
        if self['TASK'].getboolean('exact function OBC',False):
            self.cost_function = fctDef.exact_phi_OBC
            fd = True

        if self['TASK'].getboolean('exact function PBC',False):
            self.cost_function = fctDef.exact_phi_PBC
            fd = True

        if self['TASK'].getboolean('rgf_grad_naive',False):
            self.grad_function = fctDef.rgf_grad_naive

        if self['TASK'].getboolean('rgf_grad_v1',False):
            self.grad_function = fctDef.rgf_grad_v1

        if self['TASK'].getboolean('finite difference',False) or fd:
            self.grad_function = None
            self.filter_function = lambda x,y,z: y
            self.cost_function_args['eps_FD'] = self.optimizer_opts['eps']
        

        if self['TASK'].getboolean('multiple mu',False):
            self.set_task_multiple_mu()

    def set_task_multiple_mu(self):
        span = self['TASK'].getfloat('span mu',0.001)
        nb = self['TASK'].getfloat('nb mu',0)
        mu_array = np.linspace(-span, span, 2*nb+1)
        def multiple_mu(x, f):
            mu0 = np.array(x['wire'].mu[1:-1])
            A_vals=[]
            B_vals =[]
            for mu in mu_array:
                x['wire'].mu[1:-1] = mu0+mu
                x['sol'] = x['wire'].make_RGF_solver()
                A, B = f(x)
                A_vals.append(A)
                B_vals.append(B)
            x['wire'].mu[1:-1] = mu0
            x['sol'] = x['wire'].make_RGF_solver()
            return np.mean(np.array(A_vals),axis=0), np.array(B_vals).mean()
        
        f = self.cost_function
        self.cost_function = lambda x: multiple_mu(x, f)
        g = self.grad_function
        self.grad_function = lambda x: multiple_mu(x, g)

    def _set_mu_options(self):
        cfg_opt = self['OPTIMIZATION']
        d = {}
        d['optMuN'] = cfg_opt.getboolean('optMuN')
        d['optMuSc'] = cfg_opt.getboolean('optMuSc')
        d['optMuProfile'] = cfg_opt.getboolean('optMuProfile')
        d['optMu'] = d['optMuN'] or d['optMuSc'] or d['optMuProfile']
        
        if not d['optMu']:
            return {'nb_mu':0, 'optMu':False}
        
        d['mu_min'] = cfg_opt.getfloat('mu_min')
        d['mu_max'] = cfg_opt.getfloat('mu_max')
        d['mu_scaling'] = cfg_opt.getfloat('mu_scaling')
        
        if d['mu_min'] is not None:
            d['mu_min'] /=d['mu_scaling']
        if d['mu_max'] is not None:
            d['mu_max'] /=d['mu_scaling']
        if d['optMuSc'] and d['optMuN']:
            d['nb_mu'] = 2
        else:
            d['nb_mu'] = 1
        
        if d['optMuProfile']:
            d['nb_mu'] = cfg_opt.getint('mu_period', self.wire.N)
            d['optMuInit'] = cfg_opt.get('optMuInit','FromWire')
            delta = self.wire.delta.mean()
            d['mu_init_min'] = cfg_opt.getfloat('mu_init_min', -delta)
            d['mu_init_max'] = cfg_opt.getfloat('mu_init_max', delta)

        return d

    def _initial_vector_mu(self):
        x0 = []
        params = self.optimization_params['mu_opts']
        if params['nb_mu'] == 0:
            return np.array([]), []
        if params['optMuProfile']:
            mu_min = params['mu_min']
            mu_max = params['mu_max']
            init = params['optMuInit']
            if init == 'Random':
                imin = params['mu_init_min']
                imax = params['mu_init_max']
                x0 = np.random.rand(self.wire.N)*(imax - imin)+ imin
            elif init == 'FromWire':
                x0 = self.wire.mu[1:-1]
            else:
                print('Unknown option, for chemical potential profile init.')
                print('Using default option "FromWire"')
                x0 = self.wire.mu[1:-1]
            x0 = x0[:params['nb_mu']]
        else:
            if params['optMuSc']:
                mu = self.wire.mu_sc
                x0.append(mu)
            if params['optMuN']:
                mu = self.wire.mu_N
                x0.append(mu)
        x0 = np.array(x0)
        mu_scaling = params['mu_scaling']
        mu_min = params.get('mu_min')
        mu_max = params.get('mu_max')
        return x0/mu_scaling, x0.size*[(mu_min, mu_max)]

    def _set_delta_options(self):
        cfg_opt = self['OPTIMIZATION']
        params = {}
        params['opt'] = cfg_opt.getboolean('optDelta')
        params['period'] =  0
        if params['opt']:
            nL = self.wire.nL
            nsc = self.wire.nsc
            params['period'] = cfg_opt.getint('delta_period',nsc)
            params['nb_translation'] = nsc//params['period']
            assert params['period']*params['nb_translation'] == nsc
            params['slice'] = slice(nL+1, nsc+nL+1)
            params['scaling'] = cfg_opt.getfloat('delta_scaling')
            params['min'] = cfg_opt.getfloat('delta_min')
            params['max'] = cfg_opt.getfloat('delta_max')
        return params

    def _initial_vector_delta(self):
        opts = self.optimization_params['delta_opts']
        if not opts.get('opt'):
            return np.array([]), []
        sl = opts['slice']
        scaling = opts['scaling']
        dmin = opts['min']/scaling
        dmax = opts['max']/scaling
        period = opts['period']
        bounds = [(dmin,dmax)]*period
        x0 = self.wire.delta[sl][:period]
        return x0/scaling, bounds

    def _optimizer_options(self):
        opt_cfg = self['OPTIMIZER']
        opts = {}
        opts['iprint'] = opt_cfg.getint('iprint')
        opts['disp'] = opt_cfg.getboolean('disp')
        opts['maxiter'] = opt_cfg.getint('maxiter', 15000)
        opts['ftol'] = opt_cfg.getfloat('ftol', 2.2e-9)
        opts['gtol'] = opt_cfg.getfloat('gtol', 1e-5)
        opts['eps'] = opt_cfg.getfloat('eps', 1e-8)

        if self.optimization_params['useBH']:
            bh_kwargs = {}
            bh_kwargs['stepsize'] = opt_cfg.getfloat('bh stepsize', 2*np.pi)
            bh_kwargs['disp'] = opt_cfg.getboolean('bh display', True)
            bh_kwargs['interval'] = opt_cfg.getfloat('bh interval', 5)
            bh_kwargs['niter'] = opt_cfg.getint('bh nb iterations', 100)
            self.bh_kwargs = bh_kwargs
        return opts

    def _cost_function_args(self):
        cfg_opt = self['OPTIMIZATION']
        args = {}

        args['E0'] = self.get_complex('OPTIMIZATION', 'E0')
        ctls = []
        if self.optimization_params['nb_field'] > 0:
            ctls += [pauli.sxt0, pauli.syt0, pauli.szt0]
        if self.optimization_params['nb_mu'] > 0:
            ctls += [np.array(pauli.s0tz, dtype=np.complex128)]
        if self.optimization_params['nb_delta'] > 0:
            ctls += [np.array(pauli.s0tx, dtype=np.complex128)]
        args['controls'] = [np.kron(np.eye(self.wire.M), ctl) for ctl in ctls]
        
        N = self.wire.N
        last = self['OPTIMIZATION'].getint('last site avg', N//2+1)
        args['nN'] = cfg_opt.getint('nN',0)
        n0 = self.wire.nL-args['nN']
        sitesL = range(n0+1, last)
        args['sitesToAvg_L'] = sitesL
        args['sitesToAvg_R'] = range(N- sitesL[0]+1, N- sitesL[-1], -1)
        
        args.update(self.optimization_params)
        args['wire'] = self.wire
        args['results'] = {}    
        return args

    def _generate_cb_args(self):
        args = {}
        en = self.get_list('CALLBACK', 'conductance_energy_span')
        delta = self.wire.delta.mean()
        args['energies'] = np.linspace(en[0], delta*en[1], en[2])+en[3]
        args['dims'] = self.x0.size
        args['x_init'] = self.x0
        args['E0'] = self.get_complex('OPTIMIZATION', 'E0')
        args['results'] = self.cost_function_args['results']
        args['wire'] = self.wire
        return args

    def _callback(self, name='phi'):
        args = self._generate_cb_args()
        cb_opts = {}

        import errno
        cb_opts['out_folder'] = self.get('CALLBACK', 'out_folder')
        try:
            os.makedirs(cb_opts['out_folder'])
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

        cfg_cb = self['CALLBACK']
        cb_opts['stopping_criterion_phi'] = cfg_cb.getfloat('stopping_criterion_phi', -1e8)
        cb_opts['stopping_criterion_detr'] = cfg_cb.getfloat('stopping_criterion_detr', -1e8)
        cb_opts['print_diag_obc'] = cfg_cb.getboolean('diag_obc')
        cb_opts['nb_eigvecs'] = cfg_cb.getint('nb_eigvecs',2)
        cb_opts['print_cond'] = cfg_cb.getboolean('print conductance')
        cb_opts['print_dos'] = cfg_cb.getboolean('print dos')
        cb_opts['print_ldos'] = cfg_cb.getboolean('print ldos')
        cb_opts['verbose'] = cfg_cb.getboolean('verbose')
        cb_opts['bh_verbose'] = cfg_cb.getboolean('basin hopping verbose')
        useBH = self.optimization_params['useBH']
        self.callback = CallbackHelper(args, cb_opts, name, bh=useBH)