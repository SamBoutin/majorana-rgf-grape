"""
This file is part of the rgf_grape python package.
Copyright (C) 2017-2018 S. Boutin
For details of the rgf_grape algorithm and applications see:
S. Boutin, J. Camirand Lemyre, and I. Garate, Majorana bound state engineering 
via efficient real-space parameter optimization, ArXiv 1804.03170 (2018).
"""

import time
import numpy as np
from rgf_grape import ldos_utilities as ldos_utils
from rgf_grape.optimization import functionsDefinitions as fctDef
from rgf_grape.optimization import wireOptimizer as wireOptimizer
from rgf_grape.utilities import save_to_file

###############################################################################
# Callbacks
###############################################################################
class CallbackHelper(object):
    def __init__(self, args, cb_opts={}, name='run_', bh=False):
        self.bh = bh
        self.args = args
        self.cb_opts = cb_opts
        self.name = name
        self.start = time.time()
        self.counter = 0
        self.prev = 0
        self.dims = args['dims']
        self.path = cb_opts['out_folder']+'/'

        # File openings
        self.f = self.filename('iter', 0)
        self.print_header_iter_file()
        self.open_files(name, 0)
        
        if bh:
            self.bh_prev = 0
            self.bh_counter = 0
            self.fbh = self.filename('iter', 0, True)
            self.print_header_bh_file()
            self.bh_best = None

    def filename(self, task, iter, bh=False):
        if bh:
            name = 'bh_'+self.name
        else:
            name = self.name
        return self.path+'{}_{}_{}.pkl'.format(name, iter, task)

    def open_files(self, name, iter):
        if self.cb_opts.get('print_cond', False):
            self.fc = self.filename('cond', iter)
            save_to_file(self.fc, [None], False)
        if self.cb_opts.get('print_dos', False):
            self.fd = self.filename('dos', iter)
            save_to_file(self.fd, [None], False)
        if self.cb_opts.get('print_ldos', False):
            self.f_ldos = self.filename('ldos', iter)
            save_to_file(self.f_ldos, [None], False)
        if self.cb_opts.get('print_diag_obc', False):
            self.f_diag_eig = self.filename('diag_eig', iter)
            save_to_file(self.f_diag_eig, [None], False)
            self.f_diag_vec = self.filename('diag_vec', iter)
            save_to_file(self.f_diag_vec, [None], False)

    def close_files(self):
        save_to_file(self.f, [None], True)
        if self.cb_opts.get('print_cond', False):
            save_to_file(self.fc, [None], True)
        if self.cb_opts.get('print_dos', False):
            save_to_file(self.fd, [None], True)
        if self.cb_opts.get('print_ldos', False):
            save_to_file(self.f_ldos, [None], True)
        if self.cb_opts.get('print_diag_obc', False):
            save_to_file(self.f_diag_eig, [None], True)
            save_to_file(self.f_diag_vec, [None], True)

    def print_header_iter_file(self):
        title = ['iter', 'Phi', 'iter_time', 'hgap', 'det_r']
        title += ['x{}'.format(i) for i in range(self.dims)]
        save_to_file(self.f, title, False)

    def print_header_bh_file(self):
        title = ['iter', 'iter_opt', 'Phi', 'accept', 'iter_time', 'penalty_weight']
        title += ['x{}'.format(i) for i in range(self.dims)]
        save_to_file(self.fbh, title)

    def print_x(self, xk, gap=np.nan, detr=np.nan):
        elapsed = time.time() - self.start
        data = [self.counter, self.args['results']['fun'], elapsed - self.prev]
        data += [gap, detr]
        data += xk.tolist()
        save_to_file(self.f, data, True)
        self.prev = elapsed

    def __call__(self, xk=None):
        if self.cb_opts.get('verbose', True):
            # print('#Callback: total time elapsed = {:.2e}'.format(elapsed))
            print(self.counter, self.args['results']['fun'], self.args.get('penalty',0.))

        self.counter += 1
        
        hgap = self.args['results'].get('hgap_L', np.nan)
        detr = self.args['results'].get('detr_L', np.nan)
        self.print_x(xk, hgap, detr)

        if self.cb_opts.get('print_cond', False):
            gap, detr = self.getGap()

        if self.cb_opts.get('print_ldos', False):
            self.print_ldos()
        if self.cb_opts.get('print_dos', False):
            self.print_dos()
        if self.cb_opts.get('print_diag_obc', False):
            self.diag_obc()

        if self.cb_opts['stopping_criterion_phi'] is not None :
            if self.args['results']['fun'] < self.cb_opts['stopping_criterion_phi']:
                raise wireOptimizer.StopOptimizingException()
        if self.cb_opts['stopping_criterion_detr'] is not None :
            if self.args['results']['detr_L'] < self.cb_opts['stopping_criterion_detr']:
                raise wireOptimizer.StopOptimizingException()


    def getGap(self, bh_iter=False):
        sol = self.args['wire'].make_RGF_solver()
        energies = self.args['energies']
        cond = []
        for en in energies:
            cond.append(sol.conductance_BdeG('L', en + 1e-8))
        rMat = sol.rMatrix_majorana('L')
        detr = fctDef.getDetr(rMat, sol)

        save_to_file(self.fc, np.real(cond), True)

        return None, detr

    def diag_obc(self):
        N = self.args['wire'].N
        nb = self.args['wire'].nsc

        h = self.args['wire'].make_OBC_hamiltonian()
        eigvals, eigvecs = np.linalg.eigh(h)
        save_to_file(self.f_diag_eig, eigvals, True)
        nbv = self.cb_opts['nb_eigvecs']
        save_to_file(
            self.f_diag_vec,
            [range(2*nb-nbv, 2*nb+nbv), eigvecs[:, 2*nb-nbv:2*nb+nbv].T], True
        )

    def print_dos(self):
        sol = self.args['wire'].make_RGF_solver()
        dos = []
        for en in self.args['energies']:
            dos.append(np.sum(ldos_utils.ldos_vs_X(sol, en, BdeG=True)))
        save_to_file(self.fd, np.real(dos), True)

    def print_ldos(self):
        sol = self.args['wire'].make_RGF_solver()
        ldos = ldos_utils.ldos_vs_X(sol, self.args['E0'], BdeG=True)
        save_to_file(self.f_ldos, ldos.real, True)

    def basinhopping_callback(self, xk, f, accept):
        elapsed = time.time() - self.start
        self.bh_counter += 1
        if self.cb_opts.get('bh_verbose', False):
            sent = 'Basin hopping callback: '
            print('{}\t{}\t{}'.format(sent, self.counter, elapsed))
            print(f, accept)

        dataline = [self.bh_counter, self.counter, f, int(accept)]
        dataline += [elapsed - self.bh_prev]
        dataline += [self.args.get('penalty_weight',0)]
        dataline += xk.tolist()
        save_to_file(self.fbh, dataline, True)
        self.bh_prev = elapsed

        self.close_files()
      
        if self.bh_best is None:
            self.bh_best = f
        elif f <= self.bh_best:
            self.bh_best = f
    

    def __del__(self):
        self.close_files()
