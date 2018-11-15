"""
This file is part of the rgf_grape python package.
Copyright (C) 2017-2018 S. Boutin
For details of the rgf_grape algorithm and applications see:
S. Boutin, J. Camirand Lemyre, and I. Garate, Majorana bound state engineering 
via efficient real-space parameter optimization, ArXiv 1804.03170 (2018).
"""

import numpy as np
import rgf_grape.ldos_utilities as ldos_utils
from copy import deepcopy

###############################################################################
# GRAPE_SOLVER class
###############################################################################
'''
Solver inspired by the GRAPE optimal control algorithm.
Contains some of the general functions necessary for the calculation of
analytical derivatives of recursive Green's functions.
'''

class Grape_optimizer:
    def __init__(self, rgfSolv, listOfControls, E=0):
        self.sol = rgfSolv
        self.E = E

        self.gnnL = deepcopy(self.sol.calculate_gnn_L(E))
        self.gnnR = deepcopy(self.sol.calculate_gnn_R(E))
        self.gnnRet = deepcopy(self.sol.calculateAll_gnn_ret(E))

        self.u_gnnL = 0.*self.gnnL
        self.gnnL_ud = 0.*self.gnnL
        self.ud_gnnR = 0.*self.gnnL
        self.gnnR_u = 0.*self.gnnL
        for i in range(self.gnnL.shape[0]):
            self.u_gnnL[i] = rgfSolv.hopRightFrom(i)@ self.gnnL[i]
            self.gnnL_ud[i] = self.gnnL[i] @ rgfSolv.hopLeftTo(i)
            self.ud_gnnR[i] = rgfSolv.hopLeftTo(i-1)@ self.gnnR[i]
            self.gnnR_u[i] = self.gnnR[i] @ rgfSolv.hopRightFrom(i-1)

        self.controls = np.array(listOfControls)
        self.NK = len(self.controls)
        self.lastParams = {'DGR1': None, 'DGLN': None}
        M = self.sol.M
        N = self.sol.N
        self.DL = np.zeros((self.NK,M,M), dtype=np.complex128)
        self.DR = np.zeros((self.NK,M,M), dtype=np.complex128)
        self.D_sigmaL = np.zeros((self.NK,M,M), dtype=np.complex128)
        self.D_sigmaR = np.zeros((self.NK,M,M), dtype=np.complex128)
        self.center = np.zeros((self.NK,M,M), dtype=np.complex128)
    
    def isCalculated(self, params,key, up=True):
        if (self.lastParams[key] !=params) :
            if up :
                self.lastParams[key] = params
            return False
        return True

    def init_DgLN(self):
        if self.isCalculated(self.E, 'DGLN') : 
            return self.D_gLN_L,self.D_gLN_R
        M = self.sol.M; N = self.sol.N
        D_gLN_L = np.zeros((N,M,M), dtype=np.complex128) 
        D_gLN_R = np.zeros((N,M,M), dtype=np.complex128) 
        D_gLN_L[N-1] = np.array(self.gnnL[N], dtype=np.complex128)
        D_gLN_R[N-1] = np.array(self.gnnL[N], dtype=np.complex128)
        for m in range(1,N)[::-1] :
            D_gLN_L[m-1]= D_gLN_L[m] @ self.u_gnnL[m]
            D_gLN_R[m-1]= self.gnnL_ud[m] @ D_gLN_R[m]
        self.D_gLN_L = D_gLN_L
        self.D_gLN_R = D_gLN_R

    def D_gnn_L(self,n,j):
        '''
        Calculate the derivative of G^L_nn with regard to the prefix of 
        control k at site j
        '''
        if j>n : 
            return np.zeros((self.NK,self.sol.M,self.sol.M), dtype=np.complex128)
        if n==self.sol.N :
            self.init_DgLN()
            prod1 = self.D_gLN_L[j-1] 
            prod2 = self.D_gLN_R[j-1]
        else :
            prod1 = np.array(self.gnnL[n])
            prod2 = np.array(self.gnnL[n])
            # For loop from n-1 to j included (j<n).
            # No loop for n==j
            for m in range(j,n)[::-1] : 
                prod1 = prod1 @ self.u_gnnL[m]
                prod2 = self.gnnL_ud[m] @ prod2
        return np.einsum('bc,acd,de',prod1, self.controls, prod2)

    def init_DgR1(self):
        if self.isCalculated(self.E, 'DGR1') : 
            return self.D_gR1_L,self.D_gR1_R
        M = self.sol.M; N = self.sol.N
        D_gR1_L = np.zeros((N,M,M), dtype=np.complex128) 
        D_gR1_R = np.zeros((N,M,M), dtype=np.complex128) 
        gnnR_list = self.gnnR
        D_gR1_L[0] = np.array(gnnR_list[1])
        D_gR1_R[0] = np.array(gnnR_list[1])
        for m in range(2,N+1) :
            D_gR1_L[m-1]= D_gR1_L[m-2] @ self.ud_gnnR[m]
            D_gR1_R[m-1]= self.gnnR_u[m] @ D_gR1_R[m-2]
        self.D_gR1_L = D_gR1_L
        self.D_gR1_R = D_gR1_R
        
    def D_gnn_R(self,n,j):
        '''
        Calculate the derivative of G^R_nn with regard to the prefix of 
        control k at site j
        '''
        if j < n : 
            return np.zeros((self.NK,self.sol.M,self.sol.M)) 
        if n==1:
            self.init_DgR1()
            prod1= self.D_gR1_L[j-1] 
            prod2= self.D_gR1_R[j-1]
        else :
            gnnR_list = self.gnnR
            prod1 = np.matrix(gnnR_list[n])
            prod2 = np.matrix(gnnR_list[n])
            for m in range(n+1,j+1) :
                # Loop from n+1 to j included
                # If n==j, no loop since n+1>j
                prod1 = prod1 @ self.ud_gnnR[m]
                prod2 = self.gnnR_u[m] @ prod2
        return np.einsum('bc,acd,de',prod1, self.controls, prod2)
    
    def D_gnn_ret(self,n,j):
        '''
        Calculate the derivative of G^ret_nn with regard to the prefix of 
        control k at site j
        '''
        gnn = self.gnnRet[n]
        if n==j:
            self.center = self.controls
        else :
            unm1 =self.sol.hopRightFrom(n-1)
            unm1_dag =self.sol.hopLeftTo(n-1)
            self.DL = self.D_gnn_L(n-1,j)
            self.D_sigmaL = np.einsum('bc,acd,de', unm1, self.DL, unm1_dag) 

            un =self.sol.hopRightFrom(n)
            un_dag =self.sol.hopLeftTo(n)
            self.DR = self.D_gnn_R(n+1,j)
            self.D_sigmaR = np.einsum('bc,acd,de', un_dag, self.DR, un)
            self.center = self.D_sigmaL + self.D_sigmaR     
        return np.einsum('bc,acd,de',gnn,self.center,gnn)

    def D_ldos(self,E,n,k,j,BdeG =False):
        Dgret = self.D_gnn_ret(E,n,k,j)
        res = ldos_utils.ldos(Dgret, BdeG)
        return res
