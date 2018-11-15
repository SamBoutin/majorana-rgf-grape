"""
This file is part of the rgf_grape python package.
Copyright (C) 2017-2018 S. Boutin
For details of the rgf_grape algorithm and applications see:
S. Boutin, J. Camirand Lemyre, and I. Garate, Majorana bound state engineering 
via efficient real-space parameter optimization, ArXiv 1804.03170 (2018).

_rgf_solver.pyx:
Efficient cython implementation of the main rgf routines.
Inversion is performed using direct call to LAPACK through the implementation
used by scipy.
Additional 
"""

import numpy as np
cimport numpy as np
cimport cython


DTYPE = np.complex128
ctypedef np.complex128_t DTYPE_t


# Calculate M = UHU^\dag
cdef extern from "rgf_utils.h": 
	void unitaryTransform(int N,double complex* U, double complex* H, double complex* M)


import ctypes
from scipy.linalg.cython_lapack cimport zgetrf, zgetri
@cython.boundscheck(False)
def lapack_inv(int M, np.ndarray[DTYPE_t,ndim=2,mode='c'] mat) :
	cdef int INFO
	cdef np.ndarray[int, ndim=1, mode='c'] IPIV = np.empty(M, dtype=ctypes.c_int)
	cdef np.ndarray[DTYPE_t,ndim=1, mode='c'] WORK = np.empty(M,dtype=np.complex128)
	zgetrf(&M, &M, &mat[0,0], &M, &IPIV[0],&INFO)
	zgetri(&M,&mat[0,0],&M,&IPIV[0],&WORK[0],&M,&INFO)

@cython.boundscheck(False)
def _gf(
	np.ndarray[DTYPE_t,ndim=2,mode='c'] E_id, 
	np.ndarray[DTYPE_t,ndim=2,mode='c'] h, 
	np.ndarray[DTYPE_t,ndim=2,mode='c'] sigma
	):
	cdef np.ndarray[DTYPE_t,ndim=2,mode='c'] res
	try : 
		res = E_id-h-sigma
		lapack_inv(res.shape[0],res)

	except np.linalg.linalg.LinAlgError as err:
		if 'Singular matrix' in err.args:
			E = E_id[0,0]
			M = E_id.shape[0]
			if (E.imag ==0):
				new_E_id = (E +1e-12j)*np.eye(M, dtype=DTYPE)
			else :
				new_E_id = (E.real + 10j*E.imag)*np.eye(M, dtype=DTYPE)
			print("The matrix inversion for the Green function calculation is "
				  "singular at E = " + str(E) + ". Adding small imaginary part "
				  + str(new_E_id[0,0].imag) + " for the calculation.")
			return _gf(new_E_id, h,sigma)
		else:
			raise
	return res

@cython.boundscheck(False)
@cython.wraparound(False)
def calculate_gnn_R(
	int N,
	int M,
	complex E_in,
	np.ndarray[DTYPE_t,ndim=3,mode='c'] coupl_dag,
	np.ndarray[DTYPE_t,ndim=3,mode='c'] ham,
	np.ndarray[DTYPE_t,ndim=3,mode='c'] Gnn_R):
	cdef int INFO
	cdef np.ndarray[int, ndim=1, mode='c'] IPIV = np.empty(M, dtype=ctypes.c_int)
	cdef np.ndarray[DTYPE_t,ndim=1, mode='c'] WORK = np.empty(M,dtype=np.complex128)
	cdef int i,j,k
	cdef complex E
	for i in range(0,N+1)[::-1] :
		E = E_in
		while True :
			unitaryTransform(M, &coupl_dag[i,0,0], &Gnn_R[i+1,0,0], &Gnn_R[i,0,0])
			for j in range(M):
				for k in range(M):
					Gnn_R[i,j,k] *=-1.
					Gnn_R[i,j,k] -= ham[i,j,k]
			for j in range(M):
				Gnn_R[i,j,j] += E
			zgetrf(&M, &M, &Gnn_R[i,0,0], &M, &IPIV[0],&INFO)
			zgetri(&M,&Gnn_R[i,0,0],&M,&IPIV[0],&WORK[0],&M,&INFO)
			E = E.real + 1.j*(E.imag*10+1e-12)
			if INFO!=0 :
				print("The matrix inversion for the Green function calculation is singular at E = ", E )
				print(" Adding small imaginary part {:e} for the calculation.")
			if INFO ==0:
				break

@cython.boundscheck(False)
@cython.wraparound(False)
def calculate_gnn_L(
	int N, 
	int M,
	double complex E_in,
	np.ndarray[DTYPE_t,ndim=3,mode='c'] coupl, 
	np.ndarray[DTYPE_t,ndim=3,mode='c'] ham, 
	np.ndarray[DTYPE_t,ndim=3,mode='c'] Gnn_L):
	cdef int INFO
	cdef np.ndarray[int, ndim=1, mode='c'] IPIV = np.empty(M, dtype=ctypes.c_int)
	cdef np.ndarray[DTYPE_t,ndim=1, mode='c'] WORK = np.empty(M,dtype=np.complex128)
	cdef int i,j,k
	cdef complex E

	for i in range(1,N+2): 
		E = E_in
		while True :
			unitaryTransform(M, &coupl[i-1,0,0], &Gnn_L[i-1,0,0], &Gnn_L[i,0,0])
			# Gnn_L[i] = _gf(E_id,ham[i], work)
			for j in range(M):
				for k in range(M):
					Gnn_L[i,j,k] *=-1.
					Gnn_L[i,j,k] -= ham[i,j,k]
			for j in range(M):
				Gnn_L[i,j,j] += E
			zgetrf(&M, &M, &Gnn_L[i,0,0], &M, &IPIV[0],&INFO)
			zgetri(&M,&Gnn_L[i,0,0],&M,&IPIV[0],&WORK[0],&M,&INFO)
			E = E.real + 1.j*(E.imag*10+1e-12)
			if INFO!=0 :
				print("The matrix inversion for the Green function calculation is singular at E = ", E )
				print(" Adding small imaginary part {:e} for the calculation.")
			if INFO ==0:
				break
