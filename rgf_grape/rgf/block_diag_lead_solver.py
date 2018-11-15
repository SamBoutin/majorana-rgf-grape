"""
This file is part of the rgf_grape python package.
Copyright (C) 2017-2018 S. Boutin
For details of the rgf_grape algorithm and applications see:
S. Boutin, J. Camirand Lemyre, and I. Garate, Majorana bound state engineering 
via efficient real-space parameter optimization, ArXiv 1804.03170 (2018).
"""

import numpy as np
from rgf_grape.rgf.lead_solver import lead_solver
import rgf_grape.pauliMatrices as pauli
from numpy.linalg import norm

class lead_solver_phSym:
	'''
	wrapper around the lead solver to enforce block diagonal symmetries.
	In the case of a system with particle-hole symmetry such as mettalic leads.
	This wrapper can be used to enforce the basis choice as either an electron-
	hole or a Majorana basis.
	Could be made more general to allow for more basis or symmetry options.
	'''
	def __init__(self,h_onSite,hopTowScat,projList=[],phOperator=-pauli.syty):
		'''
		hopTowScat : hoppingMatrixToward Scattering region
		'''
		self.M = h_onSite.shape[0]
		if projList is None:
			projList = [ [1]*self.M ]
		if len(np.array(projList).shape)==1 : # Only one projector given
			projList=[projList]
			if np.any(projList[0] != [1]*self.M) :
				projList += [[ 1-i for i in projList[0]]] # adding the complement projector
		assert len(projList) ==2 , 'Assumes projectors for e-h parts of lead Hamiltonian'
		self.nbProj = len(projList)
		self.projList = [ np.array(p) for p in projList]
		self.checkProjListSumsToEye()
		self.buildProjectionMatrix()
		self.h_onSite = h_onSite
		self.h_blocks  = self.createBlocksFromMat(h_onSite)
		self.hopTow = hopTowScat
		self.hopTow_blocks=self.createBlocksFromMat(hopTowScat)
		self.leadSolvers = [ lead_solver(h,u) for (h,u) in zip(self.h_blocks, self.hopTow_blocks) ]
		if phOperator.shape[0] != self.M:
			self.phOperator = np.kron(np.eye(self.M//4),phOperator)
		else:
			self.phOperator = phOperator

		diffH = norm( h_onSite - self.createMatFromBlocks(self.h_blocks) )
		diffHop= norm(hopTowScat -self.createMatFromBlocks(self.hopTow_blocks))

		if diffH > 1e-6 :
			print('Possible error in leadSolver, diffH = ',diffH)
			print(h_onSite)
			print(self.createMatFromBlocks(self.h_blocks))
		if diffHop> 1e-6 :
			print('Possible error in leadSolver, diffHop = ',diffHop)
			print(hopTowScat)
			print(self.createMatFromBlocks(self.hopTow_blocks))
		self.nbModes= [0,0]

	def checkProjListSumsToEye(self):
		s = np.array(self.projList[0])
		for p in self.projList[1:] :
			s += p
		assert np.all(s == np.array([1]*s.size))

	def buildProjectionMatrix(self):
		pMatList = []
		for p in self.projList :
			n = np.sum(p)
			pMatList+= [np.zeros((self.M, n))]
			ind = np.nonzero(p)[0]
			for j,i in enumerate(ind):
				pMatList[-1][i,j] = 1
		self.pMatList = pMatList

	def createBlocksFromMat(self, h):
		assert self.M== self.projList[0].size
		hArr= []
		for pMat in self.pMatList :
			hArr += [pMat.T @ h @ pMat]
		return hArr

	def createMatFromBlocks(self, hArr):
		assert self.M== self.projList[0].size
		h = np.zeros((self.M, self.M), dtype=np.complex128)
		for (pMat, hb) in  zip (self.pMatList, hArr) :
			h += pMat @ hb @ pMat.T
		return h

	def surface_GF(self, E):
		gfs = []
		for ls in self.leadSolvers :
			gfs += [ls.surface_GF(E)]
		return self.createMatFromBlocks(gfs)

	def selfenergy(self, E):
		sigma = []
		for ls in self.leadSolvers :
			sigma += [ls.selfenergy(E)]
		return self.createMatFromBlocks(sigma)

	def correctPhases(self, phi):
		if phi.size>0:
			for i in range(phi.shape[1]):
				a = np.argmax(np.round(np.absolute(phi[:,i]),decimals=20))
				phi[:,i]*= np.conj(phi[a,i])/np.absolute(phi[a,i])
		return phi

	def propModesOut_eh(self,E):	
		phis = []; vels = []
		for i, (p,ls) in enumerate(zip(self.pMatList, self.leadSolvers)) :
			ls.surface_GF(E)
			phi,v = ls.propModesOut(E)
			vels += v.tolist()
			phi = self.correctPhases(phi)
			if phi.size is 0 :
				self.nbModes[i] = 0
			else:
				phis+= [p @ phi]
				self.nbModes[i] = phi.shape[1]

		if len(phis) is 0 :
			return np.array([]), np.array([])
		res = np.column_stack(phis)

		return res, np.array(vels)

	def propModesIn_eh(self,E):
		phis = []; vels = []
		for i,(p,ls) in enumerate(zip(self.pMatList, self.leadSolvers)) :
			ls.surface_GF(E)
			phi,v = ls.propModesIn(E)
			vels += v.tolist()
			phi = self.correctPhases(phi)
			if phi.size is  0 :
				self.nbModes[i] = 0
			else:
				phis+= [p @ phi]
				self.nbModes[i] = phi.shape[1]
		
		if len(phis) is 0 :
			return np.array([]), np.array([])
		res = np.column_stack(phis)
		
		return res, np.array(vels)

	def propModesOut_majorana(self,E):	
		E=0
		ls= self.leadSolvers[0]
		p = self.pMatList[0]
		phi,v = ls.propModesOut(E)
		vels = v.tolist()
		if phi.size is 0 :
			return np.array([]), np.array([])
		phi = self.correctPhases(phi)
		phis = p @ phi
		res = np.column_stack([phis+ self.phOperator @ phis.conj(), -1j*(phis-self.phOperator @ phis.conj())])/(np.sqrt(2.))
		return res, np.array(vels*2)

	def propModesIn_majorana(self,E):
		E=0
		ls= self.leadSolvers[0]
		p = self.pMatList[0]
		phi,v = ls.propModesIn(E)
		vels = v.tolist()
		if phi.size is 0 :
			return np.array([]), np.array([])
		phi = self.correctPhases(phi)
		phis= p @ phi
		res = np.column_stack([phis+self.phOperator @ phis.conj(), -1j*(phis-self.phOperator @ phis.conj())])/(np.sqrt(2.))
		return res, np.array(vels*2)

