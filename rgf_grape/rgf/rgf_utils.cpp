/*
This file is part of the rgf_grape python package.
Copyright (C) 2017-2018 S. Boutin
For details of the rgf_grape algorithm and applications see:
S. Boutin, J. Camirand Lemyre, and I. Garate, Majorana bound state engineering 
via efficient real-space parameter optimization, ArXiv 1804.03170 (2018).
*/

#include "rgf_utils.h"

typedef std::complex<double> cplx;

struct smartIndex2D{
	// Helper to correctly convert 2D index to 1D index.
	int N;
	smartIndex2D(int _n): N(_n) {}
	inline int ind(int i, int j) const {
		return i*N+j;
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

void matmultAB_dag(const smartIndex2D& sm,const cplx a[], const cplx b[], cplx c[]){
	for (int i=0;i< sm.N; ++i ){
		for (int j=0;j<sm.N;++j){
			cplx val=0.;
			for (int k=0; k<sm.N; ++k){
				val += a[sm.ind(i,k)]*std::conj(b[sm.ind(j,k)]);
			}
			c[sm.ind(i,j)] = val;
		}
	}
}

void unitaryTransform(int N, const std::complex<double> U[], const std::complex<double> H[],std::complex<double> M[]){
	cplx *work =new cplx[N*N];
	smartIndex2D sm(N);
	matmult(sm, U,H, work);
	matmultAB_dag(sm, work,U, M);
	delete[] work;
}