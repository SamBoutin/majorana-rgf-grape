"""
This file is part of the rgf_grape python package.
Copyright (C) 2017-2018 S. Boutin
For details of the rgf_grape algorithm and applications see:
S. Boutin, J. Camirand Lemyre, and I. Garate, Majorana bound state engineering 
via efficient real-space parameter optimization, ArXiv 1804.03170 (2018).
"""

import numpy as np

id2 = np.matrix([[1., 0.], [0., 1.]])
sx = np.matrix([[0., 1.], [1., 0]])
sy = np.matrix([[0., -1.j], [1.j, 0]])
sz = np.matrix([[1., 0.], [0., -1.]])

id4 = np.kron(id2, id2)
s0tx = np.kron(sx, id2)
s0ty = np.kron(sy, id2)
s0tz = np.kron(sz, id2)

sxt0 = np.kron(id2, sx)
sxtx = np.kron(sx, sx)
sxty = np.kron(sy, sx)
sxtz = np.kron(sz, sx)

syt0 = np.kron(id2, sy)
sytx = np.kron(sx, sy)
syty = np.real(np.kron(sy, sy))
sytz = np.kron(sz, sy)

szt0 = np.kron(id2, sz)
sztx = np.kron(sx, sz)
szty = np.kron(sy, sz)
sztz = np.kron(sz, sz)
