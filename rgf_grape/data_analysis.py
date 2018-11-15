"""
This file is part of the rgf_grape python package.
Copyright (C) 2017-2018 S. Boutin

For details of the rgf_grape algorithm and applications see:
S. Boutin, J. Camirand Lemyre, and I. Garate, Majorana bound state engineering 
via efficient real-space parameter optimization, ArXiv 1804.03170 (2018).
"""

import os
import numpy as np
import pandas as pd
import holoviews as hv
import rgf_grape

def load_data_pkl(params, filename, basinhopping=False):
    data_in = rgf_grape.utilities.load_file(filename)
    titles = data_in[0]
    data_in = data_in[1:]
    # We only keep the first trace 
    #(could be many if the basin hopping routine was used).
    try:
        index = data_in.index([None])
        data_in = data_in[:index]
    except ValueError:
        pass
    data_in = np.array(data_in)
    df = pd.DataFrame(data_in, columns=titles)
    return analyze_data(params, df, basinhopping)

def load_xvals_pkl(filename):
    data_in = rgf_grape.utilities.load_file(filename)
    titles = data_in[0]
    data_in = data_in[1:]

    # We only keep the first trace 
    #(could be many if the basin hopping routine was used).
    try:
        index = data_in.index([None])
        data_in = data_in[:index]
    except ValueError:
        pass
    data_in = np.array(data_in)
    
    # We only keep the x values:
    x_index = titles.index('x0')
    return data_in[:, x_index:]

def analyze_fields(params, data, basinhopping=False):
    nb_field = params.optimization_params['nb_field']
    field_period = params.optimization_params['texture_opts']['periods'][0]
    nb_phases = nb_field//field_period

    # Group the magnetic texture angles by site index along the wire.
    def group_by_site(row):
        return tuple((row['x{}'.format(t*field_period+i)] for t in range(nb_phases)))
    
    for i in range(field_period):
        data[i+1] = data.apply(group_by_site, axis=1)
    
    dims = params.callback.args['dims']
    xvals = ('x{}'.format(d) for d in range(dims))
    data.drop(xvals, axis=1, inplace=True)
    data = pd.melt(data, id_vars=['iter'], var_name='site', value_name='x')
    data['site'] = data['site'].apply(lambda x: int(x))
    data.sort_values(['iter', 'site'], inplace=True)
    data.reset_index(drop=True, inplace=True)
    

    nb_textures = params.wire.nb_textures
    texture_opts = params.optimization_params['texture_opts']
    opt_phi = texture_opts['optPhis']
    opt_theta = texture_opts['optThetas']

    idx = 0
    if nb_field > 0:
        fixed_phases = texture_opts['fixed_phases']
        for i, (t, f) in enumerate(zip(opt_theta, opt_phi)):
            ti = 'theta{}'.format(i)
            fi = 'phi{}'.format(i)
            if t and f:
                data[ti] = data['x'].apply(lambda x: x[idx])
                data[fi] = data['x'].apply(lambda x: x[idx+1])
                idx = idx+2
            elif t:
                data[ti] = data['x'].apply(lambda x: x[idx])
                data[fi] = data['site'].apply(lambda x: fixed_phases[i][x-1])
                idx = idx+1
            elif f:
                data[ti] = data['site'].apply(lambda x: fixed_phases[i][x-1])
                data[fi] = data['x'].apply(lambda x: x[idx])
                idx = idx+1
            else :
                data[ti] = data['site'].apply(lambda x: fixed_phases[i][0][x-1])
                data[fi] = data['site'].apply(lambda x: fixed_phases[i][1][x-1])
            data['bx{}'.format(i)] = np.sin(data[ti])*np.cos(data[fi])
            data['by{}'.format(i)] = np.sin(data[ti])*np.sin(data[fi])
            data['bz{}'.format(i)] = np.cos(data[ti])

    if nb_textures > 1:
        data['bxt'] = params.wire.textureEnergies[0]*data['bx0']
        data['byt'] = params.wire.textureEnergies[0]*data['by0']
        data['bzt'] = params.wire.textureEnergies[0]*data['bz0']
        textureEnergies = params.wire.textureEnergies
        for i in range(1, nb_textures):
            data['bxt'] += textureEnergies[i]*data['bx{}'.format(i)]
            data['byt'] += textureEnergies[i]*data['by{}'.format(i)]
            data['bzt'] += textureEnergies[i]*data['bz{}'.format(i)]
        data['bnorm'] = np.sqrt(data['bxt']**2+data['byt']**2+data['bzt']**2)
    data.drop('x', axis=1, inplace=True)

    return data

def attributes_list(data_iter, basinhopping):
    id_vars = ['Phi', 'iter_time', 'det_r']
    if basinhopping:
        id_vars = ['Phi', 'iter_time', 'accept', 'penalty_weight', 'iter_opt']
    if 'hgap' in data_iter.columns:
        id_vars += ['hgap']
    if 'gap_cond' in data_iter.columns:
        id_vars += ['gap_cond']
    return id_vars

def analyze_data(params, data_iter, basinhopping=False):
    '''
    Convert the optimization results in a panda dataframe. 
    If there is a magnetic texture optimization, convert the angles to the 
    magnetic field components.
    '''
    
    dims = params.callback.args['dims']
    nb_mu = params.optimization_params['nb_mu']
    nb_delta = params.optimization_params['nb_delta']
    nb_field = params.optimization_params['nb_field']
    field_period = params.optimization_params['texture_opts']['periods'][0]
    nb_phases = nb_field//field_period
    assert nb_mu + nb_delta + nb_field == dims
    
    field_ids = ['x{}'.format(i) for i in range(nb_field)]
    mu_ids = ['x{}'.format(d+nb_field) for d in range(nb_mu)]
    delta_ids = ['x{}'.format(d+nb_field+nb_mu) for d in range(nb_delta)]

    # We separate the real-space profiles from the scalar optimization data.
    # data_iter will only contain 'scalar' information about the iterations.
    data_iter = data_iter.loc[:, ~data_iter.columns.str.contains('^Unnamed')]
    id_vars = attributes_list(data_iter, basinhopping)
    data = pd.DataFrame.copy(data_iter)
    data.drop(id_vars, inplace=True, axis=1)
    data_iter.drop(field_ids, inplace=True, axis=1)

    # if we optimize a uniform chemical potential, it goes with data_iter,
    # otherwise it goes as a seperate real-space profile dataframe.
    if nb_mu >0:
        mu_scaling = params.optimization_params['mu_opts']['mu_scaling']
        data_iter[mu_ids] *= mu_scaling

        if params.optimization_params['mu_opts']['optMuProfile']:
            data_iter = data_iter.drop(mu_ids, axis=1)
            data_mu = pd.DataFrame.copy(data)
            data_mu.drop(field_ids, inplace=True, axis=1)
            data_mu.drop(delta_ids, inplace=True, axis=1)
            data_mu[mu_ids] *= mu_scaling

            # Rewrite columns as rows and adding a site index.
            data_mu = pd.melt(data_mu, id_vars=['iter'], var_name='site', value_name='mu')
            data_mu['site'] = data_mu['site'].apply(lambda x: int(x[1:])-nb_field +1)
            data_mu.sort_values(['iter', 'site'], inplace=True)
            data_mu.reset_index(drop=True, inplace=True)
        else:
            new_cols = {mu: 'mu{}'.format(i) for i,mu in enumerate(mu_ids)}
            data_iter.rename(columns=new_cols, inplace=True)
            data_mu = None

    if nb_delta>0:    
        data_iter = data_iter.drop(delta_indices, axis=1)
        data_delta = pd.DataFrame.copy(data)
        data_delta.drop(field_ids, inplace=True, axis=1)
        data_delta.drop(mu_ids, inplace=True, axis=1)
        
        # Rewrite the multiple columns (delta_0, delta_1, ...)
        # As a multiple rows and 2 columns (index and delta)
        data_delta = pd.melt(data_delta, id_vars=['iter'], var_name='site', value_name='delta')
        data_delta['site'] = data_delta['site'].apply(lambda x: int(x[1:])-nb_field-nb_mu +1 + params.wire.nL)
        data_delta['delta'] *= params.optimization_params['delta_scaling']
        data_delta.sort_values(['iter', 'site'], inplace=True)
        data_delta.reset_index(drop=True, inplace=True)
    else:
        data_delta = None

    # Separate the field information.
    data_field = analyze_fields(params, data, basinhopping)
    
    return data_field, data_iter, data_delta, data_mu

def generate_plots_all_fields(dataIn, params, nb_iter=None):
    data = reduce_data(dataIn, nb_iter)
    ds = hv.Dataset(data, ['iter', 'site'])
    components = ['bx', 'by', 'bz']
    plots = []
    nb_textures = params.wire.nb_textures
    texture_opts = params.optimization_params['texture_opts']

    opt_phi = texture_opts['optPhis']
    opt_theta = texture_opts['optThetas']
    for i in range(params.wire.nb_textures):
        if opt_phi[i] or opt_theta[i]:
            ci = [c+str(i) for c in components]
            fields = [ds.to(hv.Curve, 'site', b, label=b) for b in ci]
            plots.append(fields[0]*fields[1]*fields[2])
    if params.wire.nb_textures > 1:
        ci = [c+'t' for c in components]
        fields = [ds.to(hv.Curve, 'site', b, label=b) for b in ci]
        plots.append(fields[0]*fields[1]*fields[2])
        bnorm = ds.to(hv.Curve, 'site', 'bnorm', label='|B|')
        plots.append(bnorm)
    return plots

def generate_plots_delta(dataIn, params, nb_iter=None, key='delta'):
    data = reduce_data(dataIn, nb_iter)
    ds = hv.Dataset(data, ['iter', 'site'])
    return ds.to(hv.Curve, 'site', key, label=key)

def reduce_data(data, nb_iter=None):
    if nb_iter is not None:
        min_iter = data['iter'].min()
        max_iter = data['iter'].max()
        ls = np.linspace(min_iter, max_iter, min(max_iter-min_iter+1, nb_iter), dtype=int)
        data = data[data['iter'].isin(ls)]
    return data

def generate_iter_plots(data, params):
    data_iter = data.dropna(axis=1, how='all')
    ds = hv.Dataset(data_iter, ['iter'])
    cols = [d for d in data_iter.columns if d != 'iter']
    plots = [ds.to(hv.Curve, 'iter', c) for c in cols]
    return plots

def compare_x_v2(params, fold, scale_energies=True):
    N = params.wire.N
    nb = N - params.wire.nL - params.wire.nR
    args = params._generate_cb_args()
    if scale_energies:
        delta = params.wire.delta.mean()
        eigname = 'Eigenenergy [Delta]'
    else:
        delta=1.
        eigname = 'Eigenenergy [t]'
    labels = ['Initial', 'Optimized']

    conds = rgf_grape.utilities.load_file(fold+'final_0_cond.pkl')[1:3]
    ldoss = rgf_grape.utilities.load_file(fold+'final_0_ldos.pkl')[1:3]
    spectra = rgf_grape.utilities.load_file(fold+'final_0_diag_eig.pkl')[1:3]
    eigenvectors = rgf_grape.utilities.load_file(fold+'final_0_diag_vec.pkl')
    eigenvectors = [eig[1] for eig in eigenvectors[1:3]]

    plots = []
    props = []
    # 1. Conductance
    curves = []
    for cond, l in zip(conds, labels):
        tuples = [(e/delta, c) for (e, c) in zip(args['energies'], cond)]
        curves.append(hv.Curve(tuples, kdims='Energy [Delta]', vdims='G(E)',
                               label=l))
        gap, e = rgf_grape.ldos_utilities.extractGapFromLDOS(
            args['energies'], cond, threshold=0.1)
        props.append({'cond_gap': gap/delta})

    plots.append(hv.Overlay(curves))

    # 2. LDOS
    curves = []
    for c, l in zip(ldoss, labels):
        tuples = [(s, ldos) for (s, ldos) in zip(range(1, c.size+1), c)]
        curves.append(hv.Curve(tuples, kdims='Site index', vdims='LDOS(E=0)',
                               label=l))
    plots.append(hv.Overlay(curves))

    # 3. Eigenvectors
    curves = []
    nb_eigvecs = eigenvectors[0].shape[0]
    n0 = nb_eigvecs//2-1
    print(n0)
    for c, l in zip(eigenvectors, labels):
        c = c.T
        print(c.shape)
        dens = majorana_densities(c[:,[n0, n0+1]])
        tuples0 = [(s, ldos) for (s, ldos) in zip(range(1, c.size+1), dens[0])]
        tuples1 = [(s, ldos) for (s, ldos) in zip(range(1, c.size+1), dens[1])]
        curves.append(hv.Curve(tuples0, kdims='Site index',
                      vdims='Wavefunction amplitude', label=l+' L'))
        curves.append(hv.Curve(tuples1, kdims='Site index',
                      vdims='Wavefunction amplitude', label=l+' R'))
    plots.append(hv.Overlay(curves))

    # 4. Spectrums
    curves = []
    for i, (c, l) in enumerate(zip(spectra, labels)):
        pt = 100
        sl = slice(2*nb-pt, 2*nb+pt)
        ran = range(sl.start, sl.stop+1)
        tuples = [(s, ldos/delta) for (s, ldos) in zip(ran, c[sl])]
        curves.append(hv.Curve(tuples, kdims='Eigenvalue index',
                      vdims='Zoomed '+eigname,
                      label=l, group='eigen'))
        props[i]['gap_obc'] = (c[2*nb+1]-c[2*nb-2])*0.5
        props[i]['e_split_obc'] = c[2*nb]-c[2*nb-1]
    plots.append(hv.Overlay(curves))
    # 4. Spectrums
    curves = []
    for i, (c, l) in enumerate(zip(spectra, labels)):
        pt = 20
        sl = slice(2*nb-pt, 2*nb+pt)
        ran = range(sl.start, sl.stop+1)
        tuples = [(s, ldos/delta) for (s, ldos) in zip(ran, c[sl])]
        curves.append(hv.Curve(tuples, kdims='Index',
                      vdims=eigname, group='eigen',
                      label=l)
                    *hv.Scatter(tuples)
                    )
    plots.append(hv.Overlay(curves))

    return plots, props

def compare_x(xs, params, output_file=None, labels=['Spiral', 'Optimized']):
    N = params.wire.N
    nb = N - params.wire.nL- params.wire.nR
    args = params.cost_function_args
    energies = params._generate_cb_args()['energies']
    if output_file is not None and os.path.isfile(output_file):
        data = rgf_grape.utilities.load_file(output_file)
        x_file = data[0]
        conds = data[1]
        ldoss = data[2]
        spectra = data[3]
        eigenvectors = data[4]
        assert np.any(np.equal(xs,x_file))
    else:
        conds = []
        ldoss = []
        spectra = []
        eigenvectors = []
        for x in xs:
            params.x0 = x
            params.update_function(x, args)
            sol = params.wire.make_RGF_solver()
            cond = []
            for en in energies:
                cond.append(sol.conductance_BdeG('L', en + 1e-8).real)
            conds.append(cond)

            ldoss.append(rgf_grape.ldos_utilities.ldos_vs_X(sol, 0, BdeG=True))

            h = params.wire.make_OBC_hamiltonian()
            eigvals, eigvecs = np.linalg.eigh(h)
            spectra.append(eigvals)
            eigenvectors.append(eigvecs[:, 2*nb-2:2*nb+2])
        if output_file is not None:
            data = [xs, conds, ldoss, spectra, eigenvectors]
            rgf_grape.utilities.save_to_file(output_file, data)

    plots = []
    props = []
    
    # 1. Conductance
    curves = []
    for cond, l in zip(conds, labels):
        tuples = [(e, c) for (e, c) in zip(energies, cond)]
        curves.append(hv.Curve(tuples, kdims='Energy [Delta]', vdims='G(E)',
                               label=l))        
        gap, e = rgf_grape.ldos_utilities.extractGapFromLDOS(
            energies, cond, threshold=0.1)
        props.append({'cond_gap': gap})

    plots.append(hv.Overlay(curves))

    # 2. LDOS
    curves = []
    for c, l in zip(ldoss, labels):
        tuples = [(s, ldos) for (s, ldos) in zip(range(1, c.size+1), c)]
        curves.append(hv.Curve(tuples, kdims='Site index', vdims='LDOS(E=0)',
                               label=l))
    plots.append(hv.Overlay(curves))

    # 3. Eigenvectors
    curves = []
    for c, l in zip(eigenvectors, labels):
        dens = majorana_densities(c[:, [1, 2]])
        tuples0 = [(s, ldos) for (s, ldos) in zip(range(1, c.size+1), dens[0])]
        tuples1 = [(s, ldos) for (s, ldos) in zip(range(1, c.size+1), dens[1])]
        curves.append(hv.Curve(tuples0, kdims='Site index',
                      vdims='Wavefunction amplitude', label=l+' L'))
        curves.append(hv.Curve(tuples1, kdims='Site index',
                      vdims='Wavefunction amplitude', label=l+' R'))
    plots.append(hv.Overlay(curves))

    # 4. Spectrums
    curves = []
    for i, (c, l) in enumerate(zip(spectra, labels)):
        sl = slice(2*nb-20, 2*nb+20)
        ran = range(sl.start, sl.stop+1)
        tuples = [(s, ldos) for (s, ldos) in zip(ran, c[sl])]
        curves.append(hv.Curve(tuples, kdims='Eigenvalue index',
                      vdims='Eigenenergy [Delta]',
                      label=l, group='eigen')*hv.Scatter(tuples))
        props[i]['gap_obc'] = c[2*nb+1]-c[2*nb-2]
        props[i]['e_split_obc'] = c[2*nb]-c[2*nb-1]
    plots.append(hv.Overlay(curves))

    return conds, ldoss, plots, props

def majorana_densities(eig_vecs, n_orbs=4, symmetrize=True):
    if symmetrize:
        majPlus = (eig_vecs[:, 0]+eig_vecs[:, 1])/np.sqrt(2.)
        majMinus = (eig_vecs[:, 0]-eig_vecs[:, 1])/np.sqrt(2.)
    else:
        majPlus = eig_vecs[:, 0]
        majMinus = eig_vecs[:, 1]
    nb = eig_vecs.shape[0]//n_orbs
    densPlus = np.sum(np.abs(majPlus).reshape((nb, n_orbs)), axis=1)**2/2
    densMinus = np.sum(np.abs(majMinus).reshape((nb, n_orbs)), axis=1)**2/2
    return [densPlus, densMinus]