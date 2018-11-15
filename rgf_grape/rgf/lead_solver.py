"""
This file is part of the rgf_grape python package.
Copyright (C) 2017-2018 S. Boutin
For details of the rgf_grape algorithm and applications see:
S. Boutin, J. Camirand Lemyre, and I. Garate, Majorana bound state engineering 
via efficient real-space parameter optimization, ArXiv 1804.03170 (2018).

lead_solver.py:
Wrapper around kwant for calculating the lead self-energies
"""

import numpy as np
from numpy.linalg import inv, norm
import kwant

class lead_solver:
	def __init__(self, h_onSite, hopTowardsScattering):
		"""
		Input : 
			h_onSite : M x M matrix
			hop : M x M matrix
			hop is the hopping matrix from the lead towards the scattering region
			It corresponds to the upper diagonal in the case of right lead (hopping hop to the left)
		"""
		assert h_onSite.shape == hopTowardsScattering.shape
		assert h_onSite.shape[0] == h_onSite.shape[1]
		self.thresh =1e-12
		if norm(hopTowardsScattering) < self.thresh:
			self.isHardWall = True
		else :
			self.isHardWall = False

		self.lastParams = {'sigma':None, 'gf': None, 'modes':None}
		
		self.M= h_onSite.shape[0]
		self.h = np.matrix(h_onSite, dtype=np.complex128)
		# Left and right hoppings are related by dagger
		self.hopTowardsScattering = np.matrix(hopTowardsScattering, dtype=np.complex128)
		if not self.isHardWall:
			self.hopTowards_inv = inv(self.hopTowardsScattering)
		self.hopAwayFromScattering = self.hopTowardsScattering.H 
		self.id_M  = np.matlib.identity(self.M)
		self.zero_M  = np.matlib.zeros((self.M,self.M))

	def isCalculated(self, params,key, up=True):
		if (self.lastParams[key] !=params) :
			if up :
				self.lastParams[key] = params
			return False
		return True
	
	def selfenergy(self,E):
		if self.isHardWall :
			return self.zero_M
		if self.isCalculated(E, 'sigma'):
			return self.sigma
		self.sigma = kwant.physics.selfenergy(self.h-self.id_M*E, self.hopAwayFromScattering)
		return self.sigma
		
	def surface_GF(self,E):
		if self.isHardWall :
			return self.zero_M
		if self.isCalculated(E, 'gf'):
			return self.gf
		sigma = self.selfenergy(E)
		hInv = self.hopTowards_inv
		self.gf = hInv @ sigma @ hInv.H
		return self.gf

	def calculateModes(self,E):
		if self.isCalculated(E, 'modes'):
			return self.propModes
		self.propModes, allModes = kwant.physics.modes(self.h-self.id_M*E,  self.hopAwayFromScattering)
		n = self.propModes.velocities.size//2
		assert 2*n  == self.propModes.velocities.size, 'Odd number of modes'
		self.nProp = n
		return self.propModes

	def propModesOut(self,E):
		if self.isHardWall :
			return [],[]
		modes = self.calculateModes(E)
		# phis are normalized with root velocities such that 
		# np.linalg.norm(phi[:,i]*np.sqrt(vel[i])) == 1
		phi = modes.wave_functions[:,self.nProp:self.nProp*2]
		vel = modes.velocities[self.nProp:self.nProp*2]
		return phi, vel	

	def propModesIn(self,E):
		if self.isHardWall :
			return [],[]
		modes = self.calculateModes(E)
		phi = modes.wave_functions[:,0:self.nProp]
		vel = modes.velocities[0:self.nProp]
		return phi, vel
