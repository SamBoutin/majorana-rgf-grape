/*
This file is part of the rgf_grape python package.
Copyright (C) 2017-2018 S. Boutin
For details of the rgf_grape algorithm and applications see:
S. Boutin, J. Camirand Lemyre, and I. Garate, Majorana bound state engineering 
via efficient real-space parameter optimization, ArXiv 1804.03170 (2018).
*/

#include <complex>
#include <iostream>

void multiplyFromRight3D( int N, int M, std::complex<double> a[], const std::complex<double> b[],bool reverse=false);
void multiplyFromLeft3D( int N, int M, const std::complex<double> a[], std::complex<double> b[],bool reverse=false);

void generate_mL_SL(int N,  int M, int j, int start,int end, 
	const std::complex<double> gnn_ret[], const std::complex<double> u_gnnL[], std::complex<double> mL[]);

void generate_mR_SL(int N,  int M, int j, int start,int end, 
	const std::complex<double> gnn_ret[], const std::complex<double> gnnL_ud[], std::complex<double> mR[]);

void generate_mL_SR(int N,  int M, int j, int start,int end, 
	const std::complex<double> gnn_ret[], const std::complex<double> ud_gnnR[], std::complex<double> mL[]);

void generate_mR_SR(int N,  int M, int j, int start,int end, 
	const std::complex<double> gnn_ret[], const std::complex<double> gnnR_u[], std::complex<double> mR[]);

void reduce_sl(int M, int j, int start,int end, int nk,const double coeffs[], 
	const std::complex<double> mL[], const std::complex<double> mR[], std::complex<double> controls[], double res[]);
void reduce_sr(int M, int j, int start,int end, int nk,const double coeffs[], 
	const std::complex<double> mL[], const std::complex<double> mR[], std::complex<double> controls[], double res[]);