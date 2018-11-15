/*
This file is part of the rgf_grape python package.
Copyright (C) 2017-2018 S. Boutin
For details of the rgf_grape algorithm and applications see:
S. Boutin, J. Camirand Lemyre, and I. Garate, Majorana bound state engineering 
via efficient real-space parameter optimization, ArXiv 1804.03170 (2018).
*/

#include <complex>
#include <iostream>
#include <cassert>

// Calculate M = UHU^\dag
void unitaryTransform(int N, const std::complex<double> U[], const std::complex<double> H[],std::complex<double> M[]);
