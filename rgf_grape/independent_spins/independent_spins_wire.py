"""
This file is part of the rgf_grape python package.
Copyright (C) 2017-2018 S. Boutin
For details of the rgf_grape algorithm and applications see:
S. Boutin, J. Camirand Lemyre, and I. Garate, Majorana bound state engineering 
via efficient real-space parameter optimization, ArXiv 1804.03170 (2018).
"""

import numpy as np

from rgf_grape.majorana.majorana_wire import MajoranaWire
import rgf_grape.majorana.majorana_wire as mp


class WireIndependentSpins(MajoranaWire):
    def __init__(self, config):
        MajoranaWire.__init__(self, config)
        
        self.textureEnergies = config.get_list('WIRE', 'texture energies')
        self.nb_textures = len(self.textureEnergies)
        self.textures = np.zeros((self.nb_textures, self.N+2, 3))

    def local_zeeman_field(self, site):
        field = 1.*self.uniformField

        for i, amp in enumerate(self.textureEnergies):
            field += amp*self.textures[i, site, :]
        return field

    def smoothing_penalty(self, scaling, index):
        prod = np.einsum('ij,ij->i', 
                         self.textures[index, 1:-2], 
                         self.textures[index, 2:-1])
        return np.sum(1-prod)*scaling


def cartesianFromPolar(theta, phi, amplitude=1, extra=False):
    ct = np.cos(theta)
    st = np.sin(theta)
    cf = np.cos(phi)
    sf = np.sin(phi)
    vec = np.array([st * cf, st * sf, ct])*amplitude
    if extra:
        return vec.T, ct, st, cf, sf
    else:
        return vec.T


def grad_angles_conversions(theta, phi, amplitude):
    zeeman, ct, st, cf, sf = cartesianFromPolar(theta, phi, amplitude, True)
    dbdt = np.array([ct * cf, ct * sf, -st])*amplitude
    dbdf = np.array([-st * sf, st * cf, 0*st])*amplitude
    return dbdt.T, dbdf.T


def polarGradFromCartesianGrad(theta, phi, gradCart, amplitude):
    dbdt, dbdf = grad_angles_conversions(theta, phi, amplitude)
    gradTheta = np.sum(gradCart*dbdt, axis=-1)
    gradPhi = np.sum(gradCart*dbdf, axis=-1)
    return gradTheta, gradPhi


###############################################################################
# Update functions
###############################################################################
def complete_list_of_textures(xIn, wire, args):
    '''
    Get a list of textures, where each element is an array of the type
    [theta_1, theta_2, ..., theta_N, phi_1, ..., phi_N]
    '''
    x_list = []
    nb_textures = wire.nb_textures
    opt_phis = args['optPhis']
    opt_thetas = args['optThetas']
    fixed_phases = args['fixed_phases']
    cpt_fixed = 0
    pos_x = 0
    for i, (t, f) in enumerate(zip(opt_thetas, opt_phis)):
        pts_per_phase = args['periods'][i]
        if t and f:
            x_list.append(xIn[pos_x:(pos_x+2*pts_per_phase)])
            pos_x = pos_x+2*pts_per_phase
        elif t:
            thetas = xIn[pos_x:(pos_x+pts_per_phase)]
            phis = fixed_phases[cpt_fixed]
        elif f:
            thetas = fixed_phases[cpt_fixed]
            phis = xIn[pos_x:(pos_x+pts_per_phase)]
        else:
            x_list.append(None)
        if t ^ f: # (exclusive or)
            x_list.append(np.concatenate([thetas, phis]))
            pos_x = pos_x+pts_per_phase
            cpt_fixed = cpt_fixed+1
    return x_list


def reduce_list_of_textures(x_list, args):
    '''
    From a list over all textures, reduce it to keep only the elements that
    are optimized. 
    Useful for gradient calculation.
    '''
    xOut = []
    opt_phis = args['optPhis']
    opt_thetas = args['optThetas']
    for i, (t, f) in enumerate(zip(opt_thetas, opt_phis)):
        if t and f:
            xOut.append(x_list[i])
        elif t:
            pts_per_phase = x_list[i].size//2
            xOut.append(x_list[i][:pts_per_phase])
        elif f:
            pts_per_phase = x_list[i].size//2
            xOut.append(x_list[i][pts_per_phase:])
    return xOut


def updateTextures(xIn, args):
    x_list = complete_list_of_textures(xIn, args['wire'], args['texture_opts'])
    for i, x in enumerate(x_list):
        if x is not None:
            updateField(x, i, args)


def updateField(x, index, args):
    wire = args['wire']
    N = wire.N
    nbP = args['texture_opts']['periods'][index]
    vecs = cartesianFromPolar(x[:nbP], x[nbP:])
    wire.textures[index, 1:-1] = np.tile(vecs, (N//nbP, 1))
    
    if args['calculatePenalty']:
        penalty_weight = args['penalty_weight']
        args['penalty'] += wire.smoothing_penalty(penalty_weight, index)


def updateAll(x, args):
    args['penalty'] = 0
    # We assume the order : field, mu, delta
    nb_mu = args['nb_mu']
    nb_delta = args['nb_delta']
    nb_field = args['nb_field']
    if nb_field > 0:
        updateTextures(x[:nb_field], args)
    if nb_mu > 0:
        mp.updateMu(x[nb_field:(nb_field+nb_mu)], args)
    if nb_delta > 0:
        mp.updateDelta(x[-nb_delta:], args['wire'], args, args['delta_opts'])


###############################################################################
# Filter gradient
###############################################################################
def filter_textures(xIn, grad, args):
    x_list = complete_list_of_textures(xIn, args['wire'], args['texture_opts'])
    polar_grad = []
    for i, x in enumerate(x_list):
        if x is not None:
            polar_grad.append(filterPolarIndep(x, grad, i, args))
        else :
            polar_grad.append(None)
    l = reduce_list_of_textures(polar_grad, args['texture_opts'])
    return np.concatenate(l)


def gradient_penalty_texture(x, nbP, nbTrans, weight):
    vecs = cartesianFromPolar(x[:nbP], x[nbP:])
    gtheta = np.zeros(nbP)
    gphi = np.zeros(nbP)
    dbdt, dbdf = grad_angles_conversions(x[:nbP], x[nbP:], 1.)
    
    # einsum here does an element-wise dot-product.
    gtheta[1:] = np.einsum('ij,ij->i', dbdt[1:], vecs[:-1])
    gphi[1:] = np.einsum('ij,ij->i', dbdf[1:], vecs[:-1])
    gtheta[:-1] += np.einsum('ij,ij->i', dbdt[:-1], vecs[1:])
    gphi[:-1] += np.einsum('ij,ij->i', dbdf[:-1], vecs[1:])
    if nbTrans > 1:
        c = (nbTrans-1)/nbTrans
        gtheta[-1] += c*(dbdt[-1] @ vecs[0])
        gphi[-1] += c*(dbdf[-1] @ vecs[0])
        gtheta[0] += c*(dbdt[0] @ vecs[-1])
        gphi[0] += c*(dbdf[0] @ vecs[-1])
    return -1.*weight*nbTrans*np.concatenate((gtheta, gphi))


def filterPolarIndep(x, grad, index, args):
    wire = args['wire']
    N = wire.N
    nbP = args['texture_opts']['periods'][index]
    nbTrans = N//nbP
    assert grad.size == 3 * N
    gCart = grad.reshape((N, 3), order='F')

    t = np.tile(x[:nbP], nbTrans)
    f = np.tile(x[nbP:], nbTrans)
    texture_E = wire.textureEnergies[index]
    theta, phi = polarGradFromCartesianGrad(t, f, gCart, texture_E)
    grad = np.concatenate([theta, phi])

    # Going back to coarse lattice :
    finalGrad = np.zeros(2 * nbP)
    finalGrad[:nbP] = [np.sum((grad[:N])[i::nbP]) for i in range(nbP)]
    finalGrad[nbP:] = [np.sum((grad[N:])[i::nbP]) for i in range(nbP)]
    
    if args['calculatePenalty']:
        weight = args['penalty_weight']
        finalGrad += gradient_penalty_texture(x, nbP, nbTrans, weight)

    return finalGrad


def filterGradAll(x, grad, args):
    # We assume the order : field, mu, delta
    N = args['wire'].N
    nb_mu = args['mu_opts']['nb_mu']
    nb_delta = args['nb_delta']
    nb_field = x.size - nb_mu - nb_delta
    st = 0
    gradOut = []

    if nb_field > 0:
        gradOut.append(filter_textures(x[:nb_field], grad[:3 * N], args))
        st = 3*N
    if nb_mu > 0:
        sl_mu = slice(nb_field, nb_field+nb_mu)
        gradOut.append(mp.filterGradMu(x[sl_mu], grad[st:(st+N)], args))
        st = st+N
    if nb_delta > 0:
        x_delta = x[-nb_delta:]
        grad_delta = grad[st:(st+N)]
        wire = args['wire']
        gradOut.append(mp.filterGradDelta(x_delta, grad_delta, wire, args))

    return np.concatenate(gradOut)