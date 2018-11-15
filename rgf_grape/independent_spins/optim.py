"""
This file is part of the rgf_grape python package.
Copyright (C) 2017-2018 S. Boutin
For details of the rgf_grape algorithm and applications see:
S. Boutin, J. Camirand Lemyre, and I. Garate, Majorana bound state engineering 
via efficient real-space parameter optimization, ArXiv 1804.03170 (2018).
"""

import rgf_grape
from rgf_grape.optimization.wireOptimizer import WireOptimizer
from .parameters import Parameters


def evaluate_function(parameter_file=None, x0=None):
    params = Parameters(parameter_file)
    if x0 is not None:
        params.x0 = x0

    wOpt = WireOptimizer.from_config(params)
    print('init func = ', wOpt.optFunc(params.x0, wOpt.args))
    params.callback.__call__(params.x0)
    return params, wOpt


def run(parameter_file=None, x0=None):
    print('Optimization with fixed spins amplitude')
    params, wOpt = evaluate_function(parameter_file, x0)
    x0 = params.x0
    cb = params.callback
    useBH = params.optimization_params['useBH']
    if useBH:
        kwargs = {'cbBH': cb.basinhopping_callback,
                  'bh_kwargs': params.bh_kwargs}
    else:
        kwargs = {}
    res = wOpt.minimize(
            params.x0, params.bounds, wOpt.args, opts=params.optimizer_opts, 
            callback=cb, basinhopping = useBH, **kwargs)

    print(res)
    params = Parameters(parameter_file, create_all=False)
    params.set('CALLBACK', 'diag_obc', True)
    params.set('CALLBACK', 'print conductance', True)
    params.set('CALLBACK', 'print dos', True)
    params.set('CALLBACK', 'print ldos', True)
    params.set('CALLBACK', 'nb_eigvecs', 4)
    params.generate('final')
    wOpt = WireOptimizer.from_config(params)
    wOpt.optFunc(params.x0, wOpt.args)
    params.callback.__call__(params.x0)
    wOpt.optFunc(res.x, wOpt.args)
    print('x0 = ', params.x0)
    print('res.x = ', res.x)
    params.callback.__call__(res.x)


def sweep_parameter(param_name, values, parameter_file=None):
    for i, x in enumerate(values):
        print(param_name, ' = ', x)
        params = Parameters(parameter_file, create_all=False)
        params.set('CALLBACK', 'out_folder', '{}_{}'.format(param_name, i))
        params.set('WIRE', param_name, x)
        params.generate()

        wOpt = WireOptimizer.from_config(params)
        wOpt.optFunc(params.x0, wOpt.args)
        params.callback.__call__(params.x0)


def sweep_parameter_opt(param_name, values, parameter_file=None):
    for i, x in enumerate(values):
        print(param_name, ' = ', x)
        params = Parameters(parameter_file, create_all=False)
        out = '{}_{}'.format(param_name, i).replace(' ', '_')
        params.set('CALLBACK', 'out_folder', out)
        params.set('WIRE', param_name, x)
        params.generate()
        params.saveas(out+'/params.ini')
        runFD(out+'/params.ini')
