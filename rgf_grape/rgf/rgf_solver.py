"""
This file is part of the rgf_grape python package.
Copyright (C) 2017-2018 S. Boutin
For details of the rgf_grape algorithm and applications see:
S. Boutin, J. Camirand Lemyre, and I. Garate, Majorana bound state engineering 
via efficient real-space parameter optimization, ArXiv 1804.03170 (2018).
"""
""" rgf_solver.py: Class implementing the recursive Green function algorithm

    Calculate retarded Green functions of quasi-1D system using the recursive
    Green functions technique.
    Useful references for understanding the technique :
        1) Prof. M. Wimmer's Thesis (2008)
        "Quantum transport in nanostructures: From computational concepts to
        spintronics in graphene and magnetic tunnel junctions"
        2) Lewenkopf et al. 2013
        "The recursive Green's function method for graphene"
"""

import numpy as np
import numpy.matlib as npm

from rgf_grape.rgf.lead_solver import lead_solver
from rgf_grape.rgf.block_diag_lead_solver import lead_solver_phSym
import rgf_grape.ldos_utilities as ldos_utils
import rgf_grape.pauliMatrices as pauli

# The RGF implementation is much faster when using the cython implementation.
# However, an additional pure_python version is provided in cases where cython
# cannot be used.
try:
    from rgf_grape.rgf import _rgf_solver as _rgf
except ImportError:
    USE_CYTHON = False
    print('Using pure python implementation of the RGF solver.')
    print('Performances can be very poor in this case.')
    print('We recommand compiling the cython code. See README for details.')
else:
    USE_CYTHON = True


class RGF_solver:
    def __init__(self, h, coupl, projectors=[None, None], cython_on=True):
        '''
        We consider N+2 indices (0 to N+1)
            0 : last site of the left lead
            1,2, ... N sites of the chain
            N+1 : first site of the right lead
        #Note : For now, we assume that all matrices are of the same size (MxM)
            i.e. the number of orbitals per site.
        #Inputs :
        h :     list of length N+2 of on-site Hamiltonians
            h[0],h[N+1] is the left, rigth lead on-site hamiltonian and will
            be used to calculate the lead Green function.
        coupl :list of length N+1 of coupling Hamiltonians
        coupl[i] is the hopping from site i to site i+1
        (hopping from left to rigth).
        coupl[0] will be use to calculate the left lead properties
        coupl[N] will be use to calculate the right lead properties
        projetctors :
            For BdeG Hamiltonians with normal leads, it is useful for the
            calculation of the scattering matrix to know which degree of
            freedoms are holes or electrons. For these systems, we can specify 
            a projector on the elctronic dofs for the left and the right leads.
        '''
        self.use_cython(USE_CYTHON)
        self.thresh = 1.e-12
        assert len(coupl)+1 == len(h)
        N = len(h)-2
        M = h[0].shape[0]
        self.N = N
        self.M = M
        self.id_M = npm.identity(M, dtype=np.complex128)

        # For each array, we add a bool to specify if it as been calculated
        # yet.
        self.lastParams = {}
        self.resetIsCalculated()
        self.initializeListOfMatrices()
        self.h = np.array(h, dtype=np.complex128)
        self.coupl = np.array(coupl, dtype=np.complex128)
        self.coupl_dag = np.ascontiguousarray(
                        np.transpose(np.conjugate(self.coupl), [0, 2, 1]))
        # self.h = [np.matrix(h0) for h0 in h]
        # self.coupl = [np.matrix(c) for c in coupl]

        self.projL = projectors[0]
        self.projR = projectors[1]
        self._set_up_lead_solver()
        self.initInOut()

    def _set_up_lead_solver(self):
        self.proj = {'L': self.projL, 'R': self.projR}
        if self.projL is not None:
            self.ls_L = lead_solver_phSym(self.h[0], self.coupl[0], 
                                          projList=self.projL)
        else:
            self.ls_L = lead_solver(self.h[0], self.coupl[0])

        if self.projR is not None:
            self.ls_R = lead_solver_phSym(self.h[-1], self.coupl_dag[-1], 
                                          projList=self.projR)
        else:
            self.ls_R = lead_solver(self.h[-1], self.coupl_dag[-1])
        self.ls = {'L': self.ls_L, 'R': self.ls_R}

    def update(self, h, coupl):
        assert len(coupl)+1 == len(h)

        h0 = np.linalg.norm(h[0]-self.h[0])
        h1 = np.linalg.norm(h[-1]-self.h[-1])
        c0 = np.linalg.norm(coupl[0]-self.coupl[0])
        c1 = np.linalg.norm(coupl[-1]-self.coupl[-1])
        
        self.h = np.array(h, dtype=np.complex128)
        self.coupl = np.array(coupl, dtype=np.complex128)
        self.coupl_dag = np.ascontiguousarray(
                        np.transpose(np.conjugate(self.coupl), [0, 2, 1]))
        
        if h0+h1+c0+c1> self.thresh:
            self._set_up_lead_solver()
            self.resetIsCalculated()
            self.initializeListOfMatrices()
            self.initInOut()
        else:
            self.resetIsCalculated(False)
            self.initializeListOfMatrices(False)

    def use_cython(self, cython_on):
        self.use_cython_lapack = cython_on
        if cython_on:
            self._gf = _rgf._gf
            self._calculate_gnn_R = self._calculate_gnn_R_cython
            self._calculate_gnn_L = self._calculate_gnn_L_cython
        else:
            self._gf = self._gf_pure_python
            self._calculate_gnn_R = self._calculate_gnn_R_pure_python
            self._calculate_gnn_L = self._calculate_gnn_L_pure_python

    def hopRightFrom(self, n):
        if n >= 0 and n <= self.N:
            return self.coupl[n]
        if n < 0:
            return self.coupl[0]
        if n > self.N:
            return self.coupl[-1]

    def hopLeftTo(self, n):
        if n >= 0 and n <= self.N:
            return self.coupl_dag[n]
        if n < 0:
            return self.coupl_dag[0]
        if n > self.N:
            return self.coupl_dag[-1]

    def ham(self, cpt):
        if 0 <= cpt <= self.N+1:
            return self.h[cpt]
        if cpt < 0:
            return self.h[0]
        if cpt > self.N+1:
            return self.h[-1]
        assert False, 'End of function without a return statement.'

    def isCalculated(self, params, key, up=True):
        if (self.lastParams[key] != params):
            if up:
                self.lastParams[key] = params
            return False
        return True

    def resetIsCalculated(self, reset_leads=True):
        if reset_leads : 
            keys = ['gL', 'gR', 'gamL', 'gamR','SL', 'SR']
            lp = {key: None for key in keys}
            self.lastParams.update(lp)
            self.resetInOut
        keys = ['nn_L', 'nn_R', 'Nn_R', '0n_L', 'nn_ret']
        lp = {key: None for key in keys}
        self.lastParams.update(lp)


    ###########################################################################
    # In/Out modes interface :
    ###########################################################################
    def setInOutModesType(self, side, basis):
        if self.proj[side] is None:
            return
        if side == 'L':
            ls = self.ls_L
        else:
            ls = self.ls_R
        if basis.lower() == 'maj':
            ls.propModesIn = ls.propModesIn_majorana
            ls.propModesOut = ls.propModesOut_majorana
        elif basis.lower() == 'eh':
            ls.propModesIn = ls.propModesIn_eh
            ls.propModesOut = ls.propModesOut_eh
        else:
            assert False, "Unknown in-out mode basis given"
        self.inOutBasis[side] = basis

    def initInOut(self):
        sides = ['L', 'R']
        opts = ['maj', 'eh']
        self._phiIn = {k1: {k2: None for k2 in sides} for k1 in opts}
        self._phiOutDag = {k1: {k2: None for k2 in sides} for k1 in opts}
        self._velIn = {k1: {k2: None for k2 in sides} for k1 in opts}
        self._velOut = {k1: {k2: None for k2 in sides} for k1 in opts}
        self.inOutBasis = {'L': 'eh', 'R': 'eh'}
        self.resetInOut()
        self.setInOutModesType('L', 'eh')
        self.setInOutModesType('R', 'eh')

    def resetInOut(self):
        self.lastParams["inOut_L_maj"] = None
        self.lastParams["inOut_R_maj"] = None
        self.lastParams["inOut_L_eh"] = None
        self.lastParams["inOut_R_eh"] = None

    def initializeListOfMatrices(self, reset_leads=True):
        N = self.N
        M = self.M
        if reset_leads:
            self.Gnn_L = np.zeros((N+2, M, M), dtype=np.complex128)
            self.Gnn_R = np.zeros((N+2, M, M), dtype=np.complex128)
        else:
            self.Gnn_L[1:] *=0.
            self.Gnn_R[:-1] *=0.
        self.Gnn_ret = np.zeros((N+2, M, M), dtype=np.complex128)
        self.G0n_L = np.zeros((N+2, M, M), dtype=np.complex128)
        self.GNn_R = np.zeros((N+2, M, M), dtype=np.complex128)

    ###########################################################################
    # Left/Right leads :
    ###########################################################################
    def calculate_gL(self, E):
        if self.isCalculated(E, 'gL'):
            return self.Gnn_L[0]
        self.Gnn_L[0] = self.ls_L.surface_GF(E)
        return self.Gnn_L[0]

    def calculate_gR(self, E):
        if self.isCalculated(E, 'gR'):
            return self.Gnn_R[-1]
        self.Gnn_R[-1] = self.ls_R.surface_GF(E)
        return self.Gnn_R[-1]

    def leadSelfEnergy_L(self, E):
        if self.isCalculated(E, 'SL'):
            return self.sigmaL
        self.sigmaL = self.ls_L.selfenergy(E)
        return self.sigmaL

    def leadSelfEnergy_R(self, E):
        if self.isCalculated(E, 'SR'):
            return self.sigmaR
        self.sigmaR = self.ls_R.selfenergy(E)
        return self.sigmaR

    def gammaLeft(self, E):
        if self.isCalculated(E, 'gamL'):
            return self.gamL
        sL = self.leadSelfEnergy_L(E)
        self.gamL = 1j*(sL - sL.T.conj())
        return self.gamL

    def gammaRight(self, E):
        if self.isCalculated(E, 'gamR'):
            return self.gamR
        sR = self.leadSelfEnergy_R(E)
        self.gamR = 1j*(sR - sR.T.conj())
        return self.gamR

    ###########################################################################
    # Interface methods: calls either pure python or cython wrapper
    ###########################################################################
    def calculate_gnn_R(self, E):
        if self.isCalculated(E, 'nn_R'):
            return self.Gnn_R
        return self._calculate_gnn_R(E)

    def calculate_gnn_L(self, E):
        if self.isCalculated(E, 'nn_L'):
            return self.Gnn_L
        return self._calculate_gnn_L(E)

    ###########################################################################
    # Cython wrappers for RGF algorithm
    ###########################################################################
    def _calculate_gnn_R_cython(self, E):
        self.calculate_gR(E)  # sets self.Gnn_R[-1]
        _rgf.calculate_gnn_R(self.N, self.M, E, self.coupl_dag,
                             self.h, self.Gnn_R)
        return self.Gnn_R

    def _calculate_gnn_L_cython(self, E):
        self.calculate_gL(E)  # sets self.Gnn_L[0]
        _rgf.calculate_gnn_L(self.N, self.M, E, self.coupl, self.h, self.Gnn_L)
        return self.Gnn_L
    
    ###########################################################################
    # Pure python RGF methods
    ###########################################################################
    def _gf_pure_python(self, E_id, h, sigma):
        # used only if cython is off
        try:
            res = np.linalg.inv(E_id - h - sigma)
        except np.linalg.linalg.LinAlgError as err:
            if 'Singular matrix' in err.args:
                E = E_id[0, 0]
                if (E.imag == 0):
                    new_E = E + 1e-12j
                else:
                    new_E = E.real + 10j*E.imag
                print("The matrix inversion for the Green function calculation is singular at E = {:e}. Adding small imaginary part {:e} for the calculation.".format(E, new_E.imag))
                return self._gf(new_E*self.id_M, h, sigma)
            else:
                raise
        return res

    def _calculate_gnn_R_pure_python(self, E):
        self.calculate_gR(E)  # sets self.Gnn_R[-1]
        # Iterating from right to left
        for i in range(0, self.N+1)[::-1]:
            self.Gnn_R[i] = self._gf(E, self.ham(i),
                self.hopLeftTo(i) @ self.Gnn_R[i+1] @ self.hopRightFrom(i)
            )
        return self.Gnn_R

    def _calculate_gnn_L_pure_python(self, E):
        self.calculate_gL(E)  # sets self.Gnn_L[0]
        # Iterating from left to right:
        for i in range(1, self.N+2):
            self.Gnn_L[i] = self._gf(E, self.ham(i),
                self.hopRightFrom(i-1)@self.Gnn_L[i-1]@self.hopLeftTo(i-1)
            )
        return self.Gnn_L

    ###########################################################################
    # Retarded Green's fucntions
    ###########################################################################
    def calculate_gnn_ret(self, E, n):
        if self.isCalculated(E, 'nn_ret', up=False):
            return self.Gnn_ret[n]
        return self._calculate_gnn_ret(E, n)

    def _calculate_gnn_ret(self, E, n):
        if n <= 1:
            sigmaLeft = self.leadSelfEnergy_L(E)
        else:
            gnn_L = self.calculate_gnn_L(E)
            sigmaLeft = self.hopRightFrom(n-1) @ gnn_L[n-1] @ self.hopLeftTo(n-1)
        if n >= self.N:
            sigmaRight = self.leadSelfEnergy_R(E)
        else:
            gnn_R = self.calculate_gnn_R(E)
            sigmaRight = self.hopLeftTo(n) @ gnn_R[n+1] @ self.hopRightFrom(n)
        return self._gf(E*self.id_M, self.ham(n), sigmaLeft+sigmaRight)

    def calculateAll_gnn_ret(self, E):
        if self.isCalculated(E, 'nn_ret'):
            return self.Gnn_ret
        for i in range(self.N+2):
            self.Gnn_ret[i] = self._calculate_gnn_ret(E, i)
        return self.Gnn_ret

    ###########################################################################
    # 'Transport' Green's functions
    ###########################################################################
    # Careful: for simplicity, we write g_Nn_R, but actually this is G_N+1,n^R
    def calculate_gNn_R(self, E):
        if self.isCalculated(E, 'Nn_R'):
            return self.GNn_R
        self.calculate_gnn_R(E)
        self.GNn_R[self.N+1] = self.Gnn_R[self.N+1]
        for i in range(0, self.N+1)[::-1]:
            self.GNn_R[i] = \
                self.GNn_R[i+1] @self.hopRightFrom(i) @self.Gnn_R[i]
        return self.GNn_R

    def calculate_g0n_L(self, E):
        if self.isCalculated(E, '0n_L'):
            return self.G0n_L
        self.calculate_gnn_L(E)
        self.G0n_L[0] = self.Gnn_L[0]
        for i in range(1, self.N+2):
            self.G0n_L[i] = self.G0n_L[i-1] @self.hopLeftTo(i-1) @self.Gnn_L[i]
        return self.G0n_L

    def calculate_g0n_ret(self, E, n):
        gnn = self.calculate_gnn_ret(E, n)
        if n == 0:
            return gnn
        g0n_L = self.calculate_g0n_L(E)
        return g0n_L[n-1] @ self.hopLeftTo(n-1) @ gnn

    def calculate_gNn_ret(self, E, n):
        gnn = self.calculate_gnn_ret(E, n)
        if n == (self.N+1):
            return gnn
        gNn_R = self.calculate_gNn_R(E)
        return gNn_R[n+1] @ self.hopRightFrom(n) @ gnn

    def conductance(self, E):
        gam_L = self.gammaLeft(E)
        gam_R = self.gammaRight(E)
        gLR = self.calculate_g0n_ret(E, 0, self.N+1)
        return np.trace(gam_L.dot(gLR).dot(gam_R).dot(gLR.T.conj()))

    def conductance_BdeG(self, l, E):
        rMat = self.rMatrix(l, E)
        n_e = self.ls[l].nbModes[0]
        n_h = self.ls[l].nbModes[1]
        if rMat.size > 0:
            assert n_e+n_h == rMat.shape[0]
        else:
            assert n_e+n_h == 0
        if n_e > 0:
            r_ee = rMat[0:n_e, 0:n_e]
            R_ee = np.trace(r_ee @ (r_ee.T.conj()))
        else:
            R_ee = 0
        if n_h > 0 and n_e > 0:
            r_he = rMat[n_e:, 0:n_e]
            R_he = np.trace(r_he @ (r_he.T.conj()))
        else:
            R_he = 0
            R_eh = 0
        
        res = n_e - R_ee + R_he
        return res

    def sortAndCalculateInOutModes(self, l, E):
        _t = self.inOutBasis[l]
        if self.isCalculated(E, 'inOut_'+l+'_'+_t):
            return self._phiIn[_t][l], self._phiOutDag[_t][l], self._velIn[_t][l], self._velOut[_t][l]
        if l == 'L':
            phiIn, velIn = self.ls_L.propModesIn(E)
            phiOut, velOut = self.ls_L.propModesOut(E)
        elif l == 'R':
            phiIn, velIn = self.ls_R.propModesIn(E)
            phiOut, velOut = self.ls_R.propModesOut(E)
        self._phiIn[_t][l] = phiIn
        self._phiOutDag[_t][l] = phiOut.T.conj()
        self._velIn[_t][l] = velIn
        self._velOut[_t][l] = velOut

        return phiIn, self._phiOutDag[_t][l], velIn, velOut

    def _rMat(self, l, E, g_ret, derivative=False):
        if l == 'L':
            gam = self.gammaLeft(E)
        elif l == 'R':
            gam = self.gammaRight(E)
        else:
            print('l = ', l, ' expected L or R')
            exit(-1)
        phiIn, phiOutDag, velIn, velOut = self.sortAndCalculateInOutModes(l, E)
        if phiIn.shape[0] == 0 or phiOutDag.shape[0] == 0:
            return np.array([])
        res = 1.j*(gam @ g_ret @ gam)
        if not derivative:
            res = res - gam
        res2 = phiOutDag  @  res @ phiIn
        return res2

    def rMatrix(self, l, E):
        if l == 'L':
            gnn = self.calculate_gnn_ret(E, 0)
        elif l == 'R':
            gnn = self.calculate_gnn_ret(E, self.N+1)
        else:
            print('l = ', l, ' expected L or R')
            exit(-1)
        return self._rMat(l, E, gnn)

    def _rMat_majorana(self, l, g_ret, derivative=False):
        self.setInOutModesType(l, 'maj')
        rmat = self._rMat(l, 0, g_ret, derivative)
        self.setInOutModesType(l, 'eh')
        return rmat

    def rMatrix_majorana(self, l, E0=0):
        self.setInOutModesType(l, 'maj')
        rmat = self.rMatrix(l, E0)
        self.setInOutModesType(l, 'eh')
        return rmat

    def tMatrix(self, l, E):
        if l == 'LR':
            g_ret = self.calculate_g0n_ret(E, self.N+1)
            gamOut = self.gammaLeft(E)
            phiOutDag = self.sortAndCalculateInOutModes('L', E, self.projL)[1]
            gamIn = self.gammaRight(E)
            phiIn= self.sortAndCalculateInOutModes('R', E, self.projR)[0]
        elif l == 'RL':
            g_ret = self.calculate_gNn_ret(E, 0)
            gamOut = self.gammaRight(E)
            phiOutDag = self.sortAndCalculateInOutModes('R', E, self.projR)[1]
            gamIn = self.gammaLeft(E)
            phiIn = self.sortAndCalculateInOutModes('L', E, self.projL)[0]
        else:
            print('l = ', l, ' expected LR or RL')
        if phiIn.shape[0] == 0 or phiOutDag.shape[0] == 0:
            return np.array([])
        res = 1.j*(gamOut @ g_ret @ gamIn)
        res2 = phiOutDag @  res @ phiIn
        return res2

    def sMatrix(self, E):
        rLL = self.rMatrix('L', E)
        rRR = self.rMatrix('R', E)
        tLR = self.tMatrix('LR', E)
        tRL = self.tMatrix('RL', E)
        sMat = [[rLL, tLR], [tRL, rRR]]
        return sMat

    def sMatrix_majorana(self):
        self.setInOutModesType('L', 'Majorana')
        self.setInOutModesType('R', 'Majorana')
        rLL = self.rMatrix('L', 0)
        rRR = self.rMatrix('R', 0)
        tLR = self.tMatrix('LR', 0)
        tRL = self.tMatrix('RL', 0)
        sMat = [[rLL, tLR], [tRL, rRR]]
        self.sMatrixIsUnitary(sMat)
        self.setInOutModesType('L', 'eh')
        self.setInOutModesType('R', 'eh')
        return sMat