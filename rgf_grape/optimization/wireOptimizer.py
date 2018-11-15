"""
This file is part of the rgf_grape python package.
Copyright (C) 2017-2018 S. Boutin
For details of the rgf_grape algorithm and applications see:
S. Boutin, J. Camirand Lemyre, and I. Garate, Majorana bound state engineering 
via efficient real-space parameter optimization, ArXiv 1804.03170 (2018).
"""

import numpy as np
import scipy.optimize

class StopOptimizingException(Exception):
    pass

class TakeStepBH(object):
    def __init__(self, stepsize=1, scaling=None, bounds=None, dim=1):
        self.stepsize = stepsize
        self.dim =dim
        if scaling is None:
            self.scaling = np.ones(dim)
        else:
            assert np.array(scaling).size == dim
            self.scaling = scaling
        if bounds is None:
            self.lb = -np.inf*np.ones(dims)
            self.ub = np.inf*np.ones(dims)
        else:   
            assert len(bounds) == dim
            self.lb = np.array([(b[0] if b[0] is not None else -np.inf) for b in bounds])
            self.ub = np.array([(b[1] if b[1] is not None else np.inf) for b in bounds])

    def __call__(self, x):
        s = self.stepsize
        assert self.dim == x.size
        rnd = self.scaling*np.random.uniform(-s, s, self.dim)
        new_x = x+rnd
        mask_lb = new_x < self.lb
        new_x[mask_lb] = self.lb[mask_lb]*(1+np.sign(self.lb[mask_lb])*1e-6)
        mask_ub = new_x >    self.ub
        new_x[mask_ub] = self.ub[mask_ub]*(1-np.sign(self.ub[mask_ub])*1e-6)
        return new_x


class WireOptimizer(object):
    def __init__(self, wire, cost, update, grad, filter, args={}):
        self.wire = wire
        self.cost = cost
        self.grad = grad
        if 'controls' not in args:
            args['controls'] = []
        self.update = update
        self.filter = filter
        self.args = args

    @classmethod
    def from_config(cls, params):
        return cls(
            params.wire,
            cost=params.cost_function,
            grad=params.grad_function,
            update=params.update_function,
            filter=params.filter_function,
            args=params.cost_function_args
        )

    def optFunc(self, x, args):
        args = self.updateArgs(x, args)
        val, detr = self.cost(args)
        args['results']['fun'] = val
        return val

    def optGrad(self, x, args):
        args = self.updateArgs(x, args)
        grad, val = self.grad(args)
        return self.filter(x, grad, args)

    def optFuncAndGrad(self, x, args):
        args = self.updateArgs(x, args)
        grad, val = self.grad(args)
        args['results']['fun'] = val
        gradFilter = self.filter(args['x'], grad, args)
        return val, gradFilter

    def updateArgs(self, x, args):
        args['x'] = x
        args['wOpt'] = self
        self.update(x, args)
        args['sol'] = self.wire.make_RGF_solver()
        return args

    def minimize(
            self, x0, bnd=None, args={}, opts={'iprint': 1},
            callback=None, optMethod='l-bfgs-b', 
            basinhopping=False, cbBH=None, bh_kwargs={},
            constraints=()
    ):
        args = self.updateArgs(x0, args)
        kwargs = {
            'args': args, 'method': optMethod,
            'bounds': bnd, 'options': opts, 'callback': callback
        }
        if self.grad is None:
            fun = self.optFunc
        else:
            kwargs['jac'] = True
            fun = self.optFuncAndGrad
        if len(constraints)>0:
            kwargs['method'] = 'SLSQP'
            kwargs['constraints'] = constraints
        if basinhopping:
            step = bh_kwargs['stepsize']
            takeStep = TakeStepBH(step, bounds=bnd, dim=x0.size)
            try:
                res = scipy.optimize.basinhopping(
                    fun, x0, minimizer_kwargs=kwargs,
                    take_step=takeStep,
                    callback=cbBH, **bh_kwargs
                )
            except StopOptimizingException:
                res = callback.args['fun']
        else:
            try:
                res = scipy.optimize.minimize(fun, x0, **kwargs)
            except StopOptimizingException:
                res = callback.args['fun']
        return res

    def test_gradient(self, x0, filters=None, grads=None, args={}):
        print('testing gradient calculation')
        res = []
        fb = self.filter
        gb = self.grad
        if filters is None:
            filters = [lambda x,y,z: y, self.filter]
        if grads is None:
            grads = [finiteDiff, self.grad]
        for f, g in zip(filters, grads):
            print('In test grad loop')
            self.grad = g
            self.filter = f
            res += [self.optGrad(x0, args)]
        assert len(res[0]) == len(res[1])
        for i, (g1, g2) in enumerate(zip(res[0], res[1])):
            print(i, g1, g2, g1 / g2)
        self.filter = fb
        self.grad = gb
