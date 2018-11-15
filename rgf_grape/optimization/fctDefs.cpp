/*
This file is part of the rgf_grape python package.
Copyright (C) 2017-2018 S. Boutin
For details of the rgf_grape algorithm and applications see:
S. Boutin, J. Camirand Lemyre, and I. Garate, Majorana bound state engineering 
via efficient real-space parameter optimization, ArXiv 1804.03170 (2018).
*/

#include "fctDefs.h"
#include <iostream>

typedef std::complex<double> cplx;

struct smartIndex2D{
	int N;
	smartIndex2D(int _n): N(_n) {}
	inline int ind(int i, int j) const {
		return i*N+j;
	} ;
};

struct smartIndex3D{
	int N;
	smartIndex2D sm2D;
	int M2;
	smartIndex3D(int _N, int _M): N(_N), sm2D(_M), M2(_M*_M) {}
	inline int ind(int i, int j,int k) const{
		return i*M2 +sm2D.ind(j,k);
	} ;
};

void matmult(const smartIndex2D& sm,const cplx a[], const cplx b[], cplx c[]){
	for (int i=0;i< sm.N; ++i ){
		for (int j=0;j<sm.N;++j){
			cplx val=0.;
			for (int k=0; k<sm.N; ++k){
				val += a[sm.ind(i,k)]*b[sm.ind(k,j)];
			}
			c[sm.ind(i,j)] = val;
		}
	}
}

void copyArray(int N, const cplx src[], cplx target[]){
	for (int i=0;i< N;++i){
		target[i] = src[i];
	}
}

void multiplyFromRight3D( int N, int M, cplx a[], const cplx b[],bool reverse){
	// Define a M^2 matrix array
	cplx *work =new cplx[M*M];
	smartIndex3D sm(N,M);
	int rv=1;
	if (reverse){
		rv =-1;
	}
	for (int i=0;i<N;++i){
		matmult(sm.sm2D,&a[sm.ind(i,0,0)],&b[sm.ind(rv*i,0,0)],work);
		copyArray(M*M, work, &a[sm.ind(i,0,0)]);
	}
	delete[] work;
}

void multiplyFromLeft3D( int N, int M, const cplx a[],  cplx b[], bool reverse){
	// Define a M^2 matrix array
	cplx *work =new cplx[M*M];
	smartIndex3D sm(N,M);
	int rv=1;
	if (reverse){
		rv =-1;
	}
	for (int i=0;i<N;++i){
		matmult(sm.sm2D,&a[sm.ind(rv*i,0,0)],&b[sm.ind(i,0,0)],work);
		copyArray(M*M, work, &b[sm.ind(i,0,0)]);
	}
	delete[] work;
}

void generate_mLR_SL( int N, int M, const std::complex<double> gnn_ret[], std::complex<double> mL[], std::complex<double> mR[]){
	multiplyFromRight3D(N,M,mR,gnn_ret);
	multiplyFromLeft3D(N,M,gnn_ret,mL);
}
void generate_mL_SL(int N,  int M, int j, int start,int end, 
	const std::complex<double> gnn_ret[], const std::complex<double> u_gnnL[], std::complex<double> mL[]){
	cplx *work =new cplx[M*M];
	smartIndex3D sm(N,M);
	const int nsl = end-start;
	for (int i=j; i< start;++i){
		matmult(sm.sm2D,&u_gnnL[sm.ind(i,0,0)],&mL[sm.ind(0,0,0)],work);
		copyArray(M*M, work, &mL[sm.ind(0,0,0)]);
	}
	for (int i=0;i<(nsl-1);++i){
		matmult(sm.sm2D, &u_gnnL[sm.ind(start+i,0,0)], &mL[sm.ind(i,0,0)], &mL[sm.ind(i+1,0,0)]);
	}
	multiplyFromLeft3D(nsl,M,gnn_ret,mL);
	delete[] work;
}

void generate_mR_SL(int N,  int M, int j, int start,int end, 
	const std::complex<double> gnn_ret[], const std::complex<double> gnnL_ud[], std::complex<double> mR[]){
	cplx *work =new cplx[M*M];
	smartIndex3D sm(N,M);
	const int nsl = end-start;
	for (int i=j; i< start;++i){
		matmult(sm.sm2D,&mR[sm.ind(0,0,0)],&gnnL_ud[sm.ind(i,0,0)],work);
		copyArray(M*M, work, &mR[sm.ind(0,0,0)]);
	}
	for (int i=0;i<(nsl-1);++i){
		matmult(sm.sm2D, &mR[sm.ind(i,0,0)],&gnnL_ud[sm.ind(start+i,0,0)],  &mR[sm.ind(i+1,0,0)]);
	}
	multiplyFromRight3D(nsl,M,mR,gnn_ret);
	delete[] work;
}

void generate_mL_SR(int N,  int M, int j, int start,int end, 
	const std::complex<double> gnn_ret[], const std::complex<double> ud_gnnR[], std::complex<double> mL[]){
	cplx *work =new cplx[M*M];
	smartIndex3D sm(N,M);
	const int nsl = start-end;
	for (int i=j; i> start;--i){
		matmult(sm.sm2D,&ud_gnnR[sm.ind(i,0,0)],&mL[sm.ind(0,0,0)],work);
		copyArray(M*M, work, &mL[sm.ind(0,0,0)]);
	}
	for (int i=0;i<(nsl-1);++i){
		matmult(sm.sm2D, &ud_gnnR[sm.ind(start-i,0,0)], &mL[sm.ind(i,0,0)], &mL[sm.ind(i+1,0,0)]);
	}
	multiplyFromLeft3D(nsl,M,gnn_ret,mL,true);
	delete[] work;
}

void generate_mR_SR(int N,  int M, int j, int start,int end, 
	const std::complex<double> gnn_ret[], const std::complex<double> gnnR_u[], std::complex<double> mR[]){
	cplx *work =new cplx[M*M];
	smartIndex3D sm(N,M);
	const int nsl = start-end;
	// if start< j :
	// 	r = slice(j,start,-1)
	// 	mR_SR[0] = reduce(np.dot, [g for g in gnnR_u[r]])
	for (int i=j; i> start;--i){
		matmult(sm.sm2D,&mR[sm.ind(0,0,0)],&gnnR_u[sm.ind(i,0,0)],work);
		copyArray(M*M, work, &mR[sm.ind(0,0,0)]);
	}
	for (int i=0;i<(nsl-1);++i){
		matmult(sm.sm2D, &mR[sm.ind(i,0,0)],&gnnR_u[sm.ind(start-i,0,0)],  &mR[sm.ind(i+1,0,0)]);
	}
	multiplyFromRight3D(nsl,M,mR,gnn_ret,true);
	delete[] work;
}

std::complex<double> traceOfProduct(int M, const std::complex<double> A[],
	const std::complex<double> B[],const std::complex<double> C[]){
	std::complex<double> res =0;
	for (int i=0;i<M;++i){
		for (int j=0;j<M;++j){
			for (int k=0;k<M;++k){
				res += A[i*M+j]*B[j*M+k]*C[k*M+i];
			}
		}
	}
	return res;
}	

void reduce_sl(int M, int j, int start,int end,int nk, const double coeffs[], 
	const std::complex<double> mL[], const std::complex<double> mR[], std::complex<double> controls[], double res[]){
	// for k,hk in enumerate(controls):
	// 	for n in range(start,end):
	// 		res[k]+= coeff[n]*np.einsum('ij,jk,ki', mL_SL[n-start],hk,mR_SL[n-start]).imag
	const int nsl = end-start;
	smartIndex3D smk(nk,M);
	smartIndex3D sm(0,M);
	for (int ctl=0; ctl<nk;++ctl){// Loop over controls :
		for (int n=0; n<nsl;++n){
			res[ctl]+= imag(traceOfProduct(M, &mL[sm.ind(n,0,0)], &controls[smk.ind(ctl,0,0)], &mR[sm.ind(n,0,0)]))*coeffs[n+start];
		}
	}
}

void reduce_sr(int M, int j, int start,int end,int nk, const double coeffs[], 
	const std::complex<double> mL[], const std::complex<double> mR[], std::complex<double> controls[], double res[]){
	// for k,hk in enumerate(controls):
	// 	for n in range(start,end):
	// 		res[k]+= coeff[n]*np.einsum('ij,jk,ki', mL_SL[n-start],hk,mR_SL[n-start]).imag
	const int nsr = start-end;
	smartIndex3D smk(nk,M);
	smartIndex3D sm(0,M);
	for (int ctl=0; ctl<nk;++ctl){// Loop over controls :
		for (int n=0; n<nsr;++n){
			res[ctl]+= imag(traceOfProduct(M, &mL[sm.ind(n,0,0)], &controls[smk.ind(ctl,0,0)], &mR[sm.ind(n,0,0)]))*coeffs[start-n];
		}
	}
}
