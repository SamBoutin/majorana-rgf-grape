"""
This file is part of the rgf_grape python package.
Copyright (C) 2017-2018 S. Boutin
For details of the rgf_grape algorithm and applications see:
S. Boutin, J. Camirand Lemyre, and I. Garate, Majorana bound state engineering 
via efficient real-space parameter optimization, ArXiv 1804.03170 (2018).
"""

import numpy as np
from numpy.linalg import inv
from rgf_grape.optimization.grape_solver import Grape_optimizer
import rgf_grape.ldos_utilities as ldos_utilities

###############################################################################
# Reflection matrix determinant
###############################################################################
def getDetr(rMat, sol):
    if rMat.size > 0:
        detr_cpl = np.linalg.det(rMat)
        detr = np.linalg.det(rMat.real)
        if np.absolute(detr_cpl.imag) > 1e-1:
            print('Possible error : detr = ', detr_cpl)
            print('rMat =\n', rMat)
    else:
        print('Error : reflection matrix of size zero.')
        detr = np.nan
    return detr

def cf_detr(args):
    sol = args['sol']
    side = args.get('side', 'L')
    if side == 'LR':
        detrL, tmp = _cf_detr(sol, 'L')
        detrR, tmp = _cf_detr(sol, 'R')
        return detrL, detrR
    return _cf_detr(sol, side, args['E0'])


def _cf_detr(sol, side, E0=0):
    rMat = sol.rMatrix_majorana(side, E0)
    detr = getDetr(rMat, sol)
    return detr.real, detr


def grad_detr(args):
    E = args['E0']
    sol = args['sol']
    controls = args['controls']
    side = args.get('side', 'L')
    if side == 'LR':
        gL, dL = _grad_detr(E, sol, controls, 'L')
        gR, dR = _grad_detr(E, sol, controls, 'R')
        return gL+gR, dL+dR
    return _grad_detr(E, sol, controls, side)


def _grad_detr(E, sol, controls, side):
    grapeOpt = Grape_optimizer(sol, controls, E)
    if side == 'L':
        endIndex = 0
    else:  # R
        endIndex = sol.N+1
    rMat = sol.rMatrix_majorana(side, E)
    detr = getDetr(rMat, sol).real
    rMatInv = inv(rMat)

    grad = np.zeros((len(controls), sol.N))
    for j in range(1, sol.N+1):
        D_gnn_kj = grapeOpt.D_gnn_ret(endIndex, j)
        for k, hk in enumerate(controls):
            derivative = sol._rMat_majorana(side, D_gnn_kj[k], derivative=True)
            grad[k, j-1] = np.einsum('ij,ji', rMatInv, derivative).real
    grad *= detr
    return grad.ravel(), detr

###############################################################################
# Functions for heuristic gap gradient (based on zero-energy LDOS)
###############################################################################
def heuristic_gap(args):
    side = args['side']
    nN = args['nN']
    sitesToAvg = args['sitesToAvg_{}'.format(side)]
    h_gap, avg, avg_n, norm = _heuristic_gap(
        args['sol'], args['E0'], sitesToAvg, side, nN)
    avgF = avg_n/(avg*norm)
    return h_gap, avgF


def _heuristic_gap(sol, E, sitesToAvg, side, nN):
    avg = 0.
    avg_n = 0.
    for s in sitesToAvg:
        data_ldos = ldos_utilities.ldos(sol.calculate_gnn_ret(E, s), BdeG=True)
        avg += data_ldos
        avg_n += s*data_ldos
    norm = len(sitesToAvg)
    if side == 'L':
        h_gap = ((avg_n/avg - (sitesToAvg[0]+nN))/norm)**(-1.)
    else:
        h_gap = ((sitesToAvg[0] - nN - (avg_n/avg))/norm)**(-1.)
    return h_gap, avg, avg_n, norm


def performance_index(args):
    side = args.get('side', 'LR')
    if side == 'LR':
        args['side'] = 'L'
        phiL, detrL = performance_index(args)

        args['side'] = 'R'
        phiR, detrR = performance_index(args)
        phi = phiL+phiR
        detr = [detrL, detrR]
        args['side'] = 'LR'
    else:
        detr, tmp = cf_detr(args)
        args['results']['detr_{}'.format(side)] = detr

        h, x = heuristic_gap(args)
        args['results']['hgap_{}'.format(side)] = h

        # Add comment + set externally 1e-3 threshold.
        if detr < -0.999:
            phi = -1.*h
        elif detr > (1-1e-3):
            phi = h
        else:
            phi = h*detr
    return phi, detr


def _rgf_grad(args, hgap_gradient_function):
    side = args.get('side', 'LR')
    if side == 'LR':
        args['side'] = 'L'
        grad_L, phiL = _rgf_grad(args, hgap_gradient_function)
        args['side'] = 'R'
        grad_R, phiR = _rgf_grad(args, hgap_gradient_function)
        grad = grad_L + grad_R
        phi = phiL+phiR
        args['side'] = 'LR'
    else:
        g_det, detr = grad_detr(args)
        args['results']['detr_{}'.format(side)] = detr
        
        grad_h, h = hgap_gradient_function(args)
        args['results']['hgap_{}'.format(side)] = h

        # Add comment
        if np.absolute(np.absolute(detr)-1.) < 1e-3 :
          grad = grad_h *np.sign(detr)
          phi =h*np.sign(detr)
        else :
          grad = grad_h *detr + g_det*h
          phi =h*detr
        
    return grad, phi


def rgf_grad_naive(args, naive=False):
    return _rgf_grad(args, lambda x: heuristic_gap_gradient(x, naive=True))


def rgf_grad_v1(args, naive=False):
    return _rgf_grad(args, heuristic_gap_gradient)


###############################################################################
# Recursive gradient implementation
###############################################################################
def rgf_grape_grad(args):
    return _rgf_grad(args, heuristic_gap_gradient_rec)


def heuristic_gap_gradient_rec(args):
    E = args['E0']
    sol = args['sol']
    side = args['side']
    sitesToAvg = args['sitesToAvg_{}'.format(side)]
    controls = args['controls']
    nN = args['nN']
    return _heuristic_gap_grad_rec(sol, E, sitesToAvg, controls, side, nN)


def _heuristic_gap_grad_rec(sol, E, sitesToAvg, controls, side, nN):
    grapeOpt = Grape_optimizer(sol, controls, E)

    h_gap, avg, avg_n, norm = _heuristic_gap(sol, E, sitesToAvg, side, nN)
    cf = 0.5*h_gap*h_gap/(np.pi*norm*avg)
    if side == 'R':
        cf *= -1.
    full_list = np.array(
        [(s - avg_n/avg)*cf for s in range(sol.N+1)], dtype=np.double)
    coeff = np.zeros(sol.N+2, dtype=np.double)
    coeff[sitesToAvg] = full_list[sitesToAvg]

    grad = np.zeros((len(controls), sol.N))
    gnn_ret = grapeOpt.gnnRet
    gnnL = grapeOpt.gnnL 
    gnnR = grapeOpt.gnnR

    # Sn_0 sum
    for s in sitesToAvg:
        mat = (gnn_ret[s]@gnn_ret[s])
        for k, hk in enumerate(controls):
            grad[k, s-1] += coeff[s]*np.trace(mat@hk).imag

    # Sn_R
    mat = np.zeros((sol.M, sol.M), dtype=np.complex128)
    for s in range(min(sitesToAvg[0],sitesToAvg[-1])+1, sol.N+1):
        mat += coeff[s-1]*(gnn_ret[s-1]@gnn_ret[s-1])
        mat = grapeOpt.gnnR_u[s]@mat@grapeOpt.ud_gnnR[s]
        for k, hk in enumerate(controls):
            grad[k, s-1] += np.trace(mat@hk).imag

    # Sn_L
    mat = np.zeros((sol.M, sol.M), dtype=np.complex128)
    for s in range(1, max(sitesToAvg[0],sitesToAvg[-1]))[::-1]:
        mat += coeff[s+1]*(gnn_ret[s+1]@gnn_ret[s+1])
        mat = grapeOpt.gnnL_ud[s]@mat@grapeOpt.u_gnnL[s]
        for k, hk in enumerate(controls):
            grad[k, s-1] += np.trace(mat@hk).imag
    
    return grad.ravel(), h_gap


###############################################################################
# Exact gap performance index (using diagonalization)
###############################################################################
def exact_phi_OBC(args):
    detr, tmp = cf_detr(args)
    gap = args['wire'].calculate_OBC_gap()
    args['results']['detr_L'] = detr
    args['results']['hgap_L'] = gap
    if detr < -0.999:
        detr = -1.
    return 1+(1+gap)*detr, detr


def exact_phi_PBC(args):
    detr, tmp = cf_detr(args)
    gap = args['wire'].calculate_PBC_gap()
    args['results']['detr_L'] = detr
    args['results']['hgap_L'] = gap
    if detr < -0.999:
        detr =-1.
    return 1+(1+gap)*detr, detr

###############################################################################
# Heuristic performance index: first implementation correspoinding to 
# rgf_grape_v1 in Fig. S1 of the SM of the manuscript.
# This implementation is less efficient then the recursive approach above.
###############################################################################
from rgf_grape.optimization import _functionsDefinitions as _fctDef

def indicesForLeftSum(j, sitesList):
    if sitesList[-1] <= j:
        return 0, 0
    end = sitesList[-1]+1  # excluded i.e. start:end is the list we want
    if sitesList[0] > j:
        start = sitesList[0]
    else:
        start = j+1
    return start, end


def indicesForRightSum(j, sitesList):
    if sitesList[0] >= j:
        return 0, 0
    end = sitesList[0]-1  # excluded i.e. start:end:-1 is the list we want
    if sitesList[-1] < j:
        start = sitesList[-1]
    else:
        start = j-1
    return start, end


def generateMatricesForLeftSum(j, start, end, grapeOpt, coeffs):
    if end <= start:
        return np.zeros(grapeOpt.controls.shape[0])
    return _fctDef.generateMatricesForLeftSum(
        j, grapeOpt.sol.M, start, end,
        grapeOpt.gnnRet, grapeOpt.u_gnnL,
        grapeOpt.gnnL_ud, coeffs, grapeOpt.controls)


def generateMatricesForRightSum(j, sr, er, grapeOpt, coeffs):
    if sr <= er:
        return np.zeros(grapeOpt.controls.shape[0])
    return _fctDef.generateMatricesForRightSum(
        j, grapeOpt.sol.M, sr, er, grapeOpt.gnnRet,
        grapeOpt.ud_gnnR, grapeOpt.gnnR_u, coeffs, grapeOpt.controls)


def heuristic_gap_gradient(args, naive=False):
    E = args['E0']
    sol = args['sol']
    side = args.get('side', 'LR')
    controls = args['controls']
    nN = args['nN']
    if side == 'LR':
        gL, hL = _heuristic_gap_grad(
            sol, E, args['sitesToAvg_L'], controls, 'L', naive, nN)
        gR, hR = _heuristic_gap_grad(
            sol, E, args['sitesToAvg_R'], controls, 'R', naive, nN)
        return gL+gR, hL+hR
    sitesToAvg = args['sitesToAvg_{}'.format(side)]
    return _heuristic_gap_grad(sol, E, sitesToAvg, controls, side, naive, nN)


def _heuristic_gap_grad(sol, E, sitesToAvg, controls, side, naive, nN):
    grapeOpt = Grape_optimizer(sol, controls, E)
    h_gap, avg, avg_n, norm = _heuristic_gap(sol, E, sitesToAvg, side, nN)

    # Two -1. cancels here, one from the derivative and one from the ldos
    cf = 0.5*h_gap*h_gap/(np.pi*norm*avg)
    if side == 'R':
        cf *= -1.
        siteListIncr = sitesToAvg[::-1]
    else:
        siteListIncr = sitesToAvg
    coeff = np.array(
        [(s - avg_n/avg)*cf for s in range(sol.N+1)], dtype=np.double)

    grad = np.zeros((len(controls), sol.N))
    if naive:
        print('Naive heuristic gap gradient calculation (VERY SLOW)')
        coeff = coeff[sitesToAvg]
        for j in range(1, sol.N+1):
            res = np.zeros((len(controls)), dtype=np.double)
            for s, c in zip(sitesToAvg, coeff):
                res += c*np.imag(np.einsum('ijj',grapeOpt.D_gnn_ret(s, j)))
            grad[:, j-1] = res
    else:
        gnn_ret = grapeOpt.gnnRet
        if side == 'L':
            endIndex = 0
        else:
            endIndex = sol.N+1
        for j in range(1, sol.N+1):
            # j<n
            sl, el = indicesForLeftSum(j, siteListIncr)
            resL = generateMatricesForLeftSum(j, sl, el, grapeOpt, coeff)

            # j>n
            sr, er = indicesForRightSum(j, siteListIncr)
            resR = generateMatricesForRightSum(j, sr, er, grapeOpt, coeff)

            D_gnn_kj = grapeOpt.D_gnn_ret(endIndex, j)
            for k, hk in enumerate(controls):
                t2 = 0
                if siteListIncr[0] <= j <= siteListIncr[-1]:
                    t2 = coeff[j]*np.einsum(
                        'ij,jk,ki', gnn_ret[j], hk, gnn_ret[j]).imag
                grad[k, j-1] = (t2 + resL[k] + resR[k]).real

    return grad.ravel(), h_gap