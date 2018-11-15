"""
This file is part of the rgf_grape python package.
Copyright (C) 2017-2018 S. Boutin
For details of the rgf_grape algorithm and applications see:
S. Boutin, J. Camirand Lemyre, and I. Garate, Majorana bound state engineering 
via efficient real-space parameter optimization, ArXiv 1804.03170 (2018).

ldos_utilities.py :
Various functions useful for calculating, plotting and extraction information
from the local density of states (LDOS).
"""

import numpy as np


def ldos(g, BdeG=False):
    res = (-1./np.pi)*np.trace(g).imag
    if BdeG:
        return 0.5*res
    return res


def ldos_BdeG(g):
    # Factor of 2 due to the BdeG doubling of electron and holes.
    return ldos(g, BdeG=True)


def ldos_orbital(g, bnds=None, BdeG=False):
    if bnds is None:
        return ldos(g, BdeG)
    sl = slice(bnds[0], bnds[1])
    return ldos(g[sl, sl], BdeG)


def ldos_vs_X(rgf_sol, E0, bnds=None,  BdeG=False):
    data = []
    for i in range(rgf_sol.N):
        g = rgf_sol.calculate_gnn_ret(E0, i+1)
        data.append(ldos_orbital(g, bnds, BdeG))
    return np.array(data)


def findLocalMaxIndex(arr):
    """
    For an array  arr= [a,b,c], we have a local maxima at index 1 if b>a
    and b>c.
    In terms of array, we compare arr to arr shifted by 1 to left or right.
    For the boundary, the value is necessary true.
    """
    a = np.array(arr)
    c1 = np.r_[a[:-1] > a[1:], True]
    c2 = np.r_[True, a[1:] > a[:-1]]
    return np.where(c1 & c2)[0]


def findEnergyOfMaximums(energies, ldos):
    mp = findLocalMaxIndex(ldos)
    gaps = [energies[i] for i in mp]
    ldosVals = [ldos[i] for i in mp]
    return gaps, ldosVals


def extractGapFromLDOS(energies, ldos, threshold=1e-2):
    E_list, ldosVals = findEnergyOfMaximums(energies, ldos)
    if len(E_list) == 0:
        return 0, E_list
    indList = np.argsort(np.absolute(E_list))
    sorted_E = np.absolute(np.array(E_list)[indList])
    ind = 0
    if sorted_E[0] < 2*(energies[1] - energies[0]):
        ind = 1
    maxval = np.amax(ldos)
    while (ldosVals[indList[ind]] < threshold*maxval):
        ind = ind+1
        if ind >= len(indList):
            return np.nan, E_list
    return sorted_E[ind], E_list


def calculateGapFromLDOS(rgf_sol, energies, eta, sites, BdeG=False):
    data = np.zeros((len(energies), len(sites)), dtype=np.complex128)
    for c, E in enumerate(energies):
        # print ("E = ", E)
        for i, s in enumerate(sites):
            data[c, i] = rgf_sol.ldos(E+1.j*eta, s, BdeG)
    gapList = []
    gaps = []
    for i, s in enumerate(sites):
        gap, enList = extractGapFromLDOS(energies, data[:, i])
        gapList.append(enList)
        gaps.append(gap)
    return gaps, gapList, data
