"""
This file is part of the rgf_grape python package.
Copyright (C) 2017-2018 S. Boutin
For details of the rgf_grape algorithm and applications see:
S. Boutin, J. Camirand Lemyre, and I. Garate, Majorana bound state engineering 
via efficient real-space parameter optimization, ArXiv 1804.03170 (2018).

_functionsDefinitions.pyx:
These functions are use for what is called "rgf-grape_v1" in Figure S1 of the 
above reference.
This implementation is available to allow reproducibility of results. 
However, other than for benchmarking use, the more efficient recursive
implementation should be used instead.
"""

import numpy as np
cimport numpy as np
cimport cython

DTYPE = np.complex128
ctypedef np.complex128_t DTYPE_t

cdef extern from "fctDefs.h": 
	void generate_mL_SL(int N,  int M, int j, int start,int end, 
	double complex* gnn_ret, double complex* u_gnnL, double complex* mL)
	void generate_mR_SL(int N,  int M, int j, int start,int end, 
	double complex* gnn_ret, double complex* gnnL_ud, double complex* mR)
	void generate_mL_SR(int N,  int M, int j, int start,int end, 
	double complex* gnn_ret, double complex* ud_gnnR, double complex* mL)
	void generate_mR_SR(int N,  int M, int j, int start,int end, 
	double complex* gnn_ret, double complex* u_gnnR, double complex* mR)

	void reduce_sl(int M, int j, int start,int end,int nk,double* coeffs, 
	double complex* mL, double complex* mR, double complex* controls, double* res)

	void reduce_sr(int M, int j, int start,int end,int nk,double* coeffs, 
	double complex* mL, double complex* mR, double complex* controls, double* res)


def generateMatricesForLeftSum(int j,
	int M, 
	int start, int end,
	np.ndarray[DTYPE_t,ndim=3,mode='c']  gnn_ret, 
	np.ndarray[DTYPE_t,ndim=3,mode='c']  u_gnnL, 
	np.ndarray[DTYPE_t,ndim=3,mode='c']  gnnL_ud,
	np.ndarray[np.double_t,ndim=1,mode='c'] coeff, 
	np.ndarray[DTYPE_t,ndim=3,mode='c'] controls):
	cdef int nsl = end - start
	cdef np.ndarray[DTYPE_t,ndim=3,mode='c'] mL_SL = np.zeros((nsl,M,M), dtype=DTYPE)
	cdef np.ndarray[DTYPE_t,ndim=3,mode='c'] mR_SL = np.zeros((nsl,M,M), dtype=DTYPE)
	mL_SL[0] = np.eye(M)
	mR_SL[0] = np.eye(M)
	
	cdef int N = u_gnnL.shape[0]
	generate_mL_SL(N,M,j,start,end, &gnn_ret[start,0,0], <double complex*> u_gnnL.data, <double complex*> mL_SL.data)
	generate_mR_SL(N,M,j,start,end, &gnn_ret[start,0,0], <double complex*> gnnL_ud.data, <double complex*> mR_SL.data)

	cdef np.ndarray[np.double_t,ndim=1,mode='c'] res= np.zeros(controls.shape[0],dtype=np.double)
	cdef int nk = controls.shape[0]
	reduce_sl(M, j, start,end, nk,<double*> coeff.data,  <double complex*> mL_SL.data, <double complex*> mR_SL.data, <double complex*> controls.data, <double*> res.data)

	return res
	
def generateMatricesForRightSum(int j,
	int M, 
	int start, int end,
	np.ndarray[DTYPE_t,ndim=3,mode='c']  gnn_ret, 
	np.ndarray[DTYPE_t,ndim=3,mode='c']  ud_gnnR, 
	np.ndarray[DTYPE_t,ndim=3,mode='c']  gnnR_u,
	np.ndarray[np.double_t,ndim=1,mode='c'] coeff, 
	np.ndarray[DTYPE_t,ndim=3,mode='c'] controls ):
	cdef int nsr = start-end
	cdef np.ndarray[DTYPE_t,ndim=3,mode='c'] mL_SR = np.zeros((nsr,M,M), dtype=DTYPE)
	cdef np.ndarray[DTYPE_t,ndim=3,mode='c'] mR_SR = np.zeros((nsr,M,M), dtype=DTYPE)
	mL_SR[0] = np.eye(M)
	mR_SR[0] = np.eye(M)
	
	cdef int N = ud_gnnR.shape[0]
	generate_mL_SR(N,M,j,start,end, &gnn_ret[start,0,0], <double complex*> ud_gnnR.data, <double complex*> mL_SR.data)
	generate_mR_SR(N,M,j,start,end, &gnn_ret[start,0,0], <double complex*> gnnR_u.data, <double complex*> mR_SR.data)

	cdef np.ndarray[np.double_t,ndim=1,mode='c'] res= np.zeros(controls.shape[0],dtype=np.double)
	cdef int nk = controls.shape[0]
	reduce_sr(M, j, start,end, nk,<double*> coeff.data,  <double complex*> mL_SR.data, <double complex*> mR_SR.data, <double complex*> controls.data, <double*> res.data)

	return res