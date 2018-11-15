"""
This file is part of the rgf_grape python package.
Copyright (C) 2017-2018 S. Boutin
For details of the rgf_grape algorithm and applications see:
S. Boutin, J. Camirand Lemyre, and I. Garate, Majorana bound state engineering 
via efficient real-space parameter optimization, ArXiv 1804.03170 (2018).
"""

import numpy as np
from rgf_grape.rgf.rgf_solver import RGF_solver
import rgf_grape.pauliMatrices as pauli
import scipy.linalg as spla

class MajoranaWire(object):
    def __init__(self, config):
        cfg_wire = config['WIRE']
        params = {}

        self.params = params
        self.N = cfg_wire.getint('N')
        self.nL = cfg_wire.getint('nL')
        self.nR = cfg_wire.getint('nR')
        self.nsc = self.N - self.nL - self.nR

        self.M = cfg_wire.getint('M',1)
        self.tx = cfg_wire.getfloat('t')
        self.ty = self.tx
        self.initializeHoppings()

        self.mu_lead = cfg_wire.getfloat('mu lead')
        self.mu_sc = cfg_wire.getfloat('mu sc')
        self.mu_N = cfg_wire.getfloat('mu normal region')
        self.create_step_constant_mu_array()
        self.delta = cfg_wire.getfloat('delta')*np.ones(self.N+2)

        self.initializeBarrier(config)
        self.initializeAlpha(config)
        self.initLeads(params)

        self.uniformField = np.array(config.get_list('WIRE', 'uniform field'))
        self._sol=None
        self.proj = [[1, 1, 0, 0] * self.M, [1, 1, 0, 0] * self.M]
        
    def create_step_constant_mu_array(self):
        self.mu = [self.mu_lead] + [self.mu_N]*self.nL
        self.mu += [self.mu_sc]*self.nsc 
        self.mu +=[self.mu_N]*self.nR + [self.mu_lead]
        self.mu = np.array(self.mu)

    def initializeHoppings(self):
        if self.M > 1:
            self.epsZero = 2 * self.tx + 2 * self.ty
        else:
            self.epsZero = 2 * self.tx

    def initializeAlpha(self, config):
        alpha = config.get_list('WIRE', 'alpha', 0.)
        print(alpha)
        if len(alpha)>1:
            self.alpha_x = alpha[0]
            self.alpha_y = alpha[1]
        else:
            self.alpha_x = alpha[0]
            self.alpha_y = alpha[0]

    def initializeBarrier(self, config):
        self.gammaL = config['WIRE'].getfloat('gamma_L')
        self.gammaR = config['WIRE'].getfloat('gamma_R')

    def initLeads(self, params):
        if 'fieldFreeLeads' in params.keys():
            self.fieldFreeLeads = True
            self.fieldInLeads = params['fieldFreeLeads']
        else:
            self.fieldFreeLeads = False

    def local_zeeman_field(self, site):
        field = 1.*self.uniformField
        return field

    def generateOnSite_h(self, mu_, delta_, b_, _alphay):
        h0 = np.zeros((4 * self.M, 4 * self.M), dtype=np.complex128)
        for i in range(self.M):
            sl = slice(4 * i, 4 * (i + 1))
            h0[sl, sl] = (
                (self.epsZero - mu_)*pauli.s0tz
                + delta_*pauli.s0tx + b_[0]*pauli.sxt0
                + b_[1]*pauli.syt0 + b_[2]*pauli.szt0
            )
        for i in range(self.M - 1):
            sl1 = slice(4 * i, 4 * (i + 1))
            sl2 = slice(4 * (i + 1), 4 * (i + 2))
            # Hopping from site y=i to site y=i+1
            h0[sl2, sl1] = -self.ty * pauli.s0tz + 1.j * _alphay * pauli.sxtz
            # Hopping from site y=i+1 to site y=i
            h0[sl1, sl2] = -self.ty * pauli.s0tz - 1.j * _alphay * pauli.sxtz
        return h0

    def hOnSite_lead(self, s):
        return self.generateOnSite_h(
            self.mu[s], 0, self.uniformField, self.alpha_y)

    def hOnSite_normal(self, s):
        b = self.local_zeeman_field(s)
        return self.generateOnSite_h(self.mu[s], 0, b, self.alpha_y)

    def hOnSite_sc(self, s):
        b = self.local_zeeman_field(s)
        return self.generateOnSite_h(
            self.mu[s], self.delta[s], b, self.alpha_y)

    def hop(self, _alpha_x):
        hCoupl = np.zeros((4 * self.M, 4 * self.M), dtype=np.complex128)
        for i in range(self.M):
            # Hopping from x = j to x = j+1
            sl = slice(4 * i, 4 * (i + 1))
            hCoupl[sl, sl] = -self.tx*pauli.s0tz - 1.j*_alpha_x*pauli.sytz
        return hCoupl

    def createArrays(self):
        N = self.N
        M = self.M
        nL = self.nL
        nR = self.nR
        h_arr = np.zeros((N+2, M*4, M*4),
                         dtype=np.complex128)
        h_arr[0] = self.hOnSite_lead(0)
        for i in range(1, nL+1):
            h_arr[i] = self.hOnSite_normal(i)
        for i in range(nL+1, N-nR + 1):
            h_arr[i] = self.hOnSite_sc(i)
        for i in range(N-nR+1, N+1):
            h_arr[i] = self.hOnSite_normal(i)
        h_arr[N + 1] = self.hOnSite_lead(N + 1)
        coupl_arr = np.tile(self.hop(self.alpha_x), (N + 1, 1, 1))
        return h_arr, coupl_arr

    def correctLeads(self, h_arr, coupl_arr):
        if self.fieldFreeLeads:
            bFree = [0, 0, self.fieldInLeads]
            h_arr[0] = self.generateOnSite_h(self.mu[0], 0, bFree, 0)
            h_arr[self.N + 1] = h_arr[0]
            coupl_arr[0] = self.hop(0)
            coupl_arr[self.N] = coupl_arr[0]
        return h_arr, coupl_arr

    def addBarrier(self, coupl_arr):
        # coupling from site nL to site nL+1
        coupl_arr[self.nL] = self.gammaL * self.hop(self.alpha_x)
        # coupling from site N-NR-1 to site N-NR
        coupl_arr[self.N - self.nR] = self.gammaR * self.hop(self.alpha_x)
        return coupl_arr

    def make_RGF_solver(self):
        h_arr, coupl_arr = self.createArrays()
        h_arr, coupl_arr = self.correctLeads(h_arr, coupl_arr)
        coupl_arr = self.addBarrier(coupl_arr)
        
        if self._sol is None:
            self._sol = RGF_solver(h_arr, coupl_arr, projectors=self.proj)
        else:
            self._sol.update(h_arr, coupl_arr)
        return self._sol

    def make_OBC_hamiltonian(self):
        # Consider only the sc region with open boundary conditions.
        M = 4*self.M
        N = self.N
        nsc = self.nsc
        ham = np.zeros((M*nsc, M*nsc), dtype=np.complex128)
        # Assumes constant hopping!
        hop = self.hop(self.alpha_x)
        for i in range(nsc):
            ham[i*M:(i+1)*M, i*M:(i+1)*M] = self.hOnSite_sc(i + self.nL + 1)
        for i in range(nsc-1):
            # Hop from right to left
            ham[i*M:(i+1)*M, (i+1)*M:(i+2)*M] = np.conj(hop.T)
            # Hop from left to right
            ham[(i+1)*M:(i+2)*M, i*M:(i+1)*M] = hop
        return ham

    def make_PBC_hamiltonian(self):
        # Consider only the sc region with periodic boundary conditions.
        ham = self.make_OBC_hamiltonian()
        M = 4*self.M

        # Assumes constant hopping!
        hop = self.hop(self.alpha_x)
        # Pbc :
        # Hop from right to left (bottom left corner of ham)
        ham[-M:, :M] = np.conj((hop).T)
        # Hop from left to right (top right corner of ham)
        ham[:M, -M:] = hop
        return ham

    def calculate_PBC_gap(self):
        ham = self.make_PBC_hamiltonian()
        eigs = spla.eigh(ham, check_finite=False, eigvals_only=True)
        nb = eigs.size//2
        return 0.5*(eigs[nb]-eigs[nb-1])

    def calculate_OBC_gap(self):
        ham = self.make_OBC_hamiltonian()
        eigs = spla.eigh(ham, check_finite=False, eigvals_only=True)
        nb = eigs.size//2
        return 0.5*(eigs[nb+1]-eigs[nb-2])


def updateMu(x, args):
    args_mu = args['mu_opts']
    if args_mu['optMuProfile']:
        nb_trans = args['wire'].N//args['nb_mu']
        x = np.concatenate([x]*nb_trans)
        args['wire'].mu[1:-1] = args_mu['mu_scaling']*x
        
        w = args['penalty_weight']
        calc_p = args['calculatePenalty']
        args['penalty'] += smoothing_penalty(x, w, calc_p, nb_trans)*args_mu['mu_scaling']
    else:
        if args_mu['optMuSc']:
            args['wire'].mu_sc = args_mu['mu_scaling']*x[0]
        if args_mu['optMuN']:
            args['wire'].mu_N = args_mu['mu_scaling']*x[-1]
        args['wire'].create_step_constant_mu_array()


def filterGradMu(x, grad, args):
    args_mu = args['mu_opts']
    if args_mu['optMuProfile']:
        nb_trans = args['wire'].N//args['nb_mu']
        ns = args['nb_mu']
        grad_mu =[ np.sum(grad[i::ns]) for i in range(ns)]
        grad_mu = -np.array(grad_mu)
        if args['calculatePenalty']:
            grad_mu[1:] += 2.*(x[1:]- x[:-1])*args['penalty_weight']
            grad_mu[:-1] += 2.*(x[:-1]- x[1:])*args['penalty_weight']
        
            # Boundary term
            if nb_trans>1:
                grad_mu[0] += -2.*(x[-1]-x[0])*args['penalty_weight']
                grad_mu[-1] += 2.*(x[-1]-x[0])*args['penalty_weight']
    else:
        N = args['wire'].N
        nbL = args['wire'].nL
        nbR = args['wire'].nR
        assert grad.size == N
        grad_N = -np.sum(grad[:nbL]) - np.sum(grad[N - nbR:])
        grad_sc = -np.sum(grad[nbL:(N - nbR)])
        
        grad = []
        if args_mu['optMuSc']:
            grad += [grad_sc]
        if args_mu['optMuN']:
            grad += [grad_N]
        assert len(grad) == args_mu['nb_mu']
        grad_mu = np.array(grad)
    return args_mu['mu_scaling']*grad_mu


def smoothing_penalty(x, weight, calculate, nb_trans):
    if calculate:
        penalty = np.sum((x[1:]-x[:-1])**2)
        if nb_trans>1:
            penalty += (x[-1]-x[0])**2
        return penalty*weight
    else:
        return 0


def updateDelta(x, wire, args, args_delta):
    nb_trans = args_delta['nb_translation']
    x = np.tile(x, nb_trans)
    sl = args_delta['slice']
    wire.delta[sl] = args_delta['scaling']*x

    w = args['penalty_weight']
    calc_p = args['calculatePenalty']
    args['penalty'] += smoothing_penalty(x, w, calc_p, nb_trans)


def filterGradDelta(x, grad, wire, args):
    args_delta = args['delta_opts']
    nb_trans = args_delta['nb_translation']
    x = np.tile(x, nb_trans )
    nbL = wire.nL
    nsc = wire.nsc
    
    ns = args_delta['period']
    grad_sc = grad[nbL:(nsc+nbL)]*args_delta['scaling']
    if args['calculatePenalty']:
        grad_sc[1:] += 2.*(x[1:]- x[:-1])*args['penalty_weight']
        grad_sc[:-1] += 2.*(x[:-1]- x[1:])*args['penalty_weight']

    # If finite translation invariance, sum over equivalent sites.
    grad_sc =np.array([ np.sum(grad[i::ns]) for i in range(ns)])
    return grad_sc