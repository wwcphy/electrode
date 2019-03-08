# -*- coding: utf8 -*-
# 
# Adjust the ion's position/frequency
# 
# By Wance 09/20/2018
# 

from __future__ import (absolute_import, print_function,
        unicode_literals, division)

import warnings
import copy
import numpy as np
from scipy import constants as ct
from scipy.linalg import solve as scipy_solve


class Adjust():
	system = None
	basis = None    # np.eye(3)

	# self.system, self.basis, self.E_config, self.xms/volts/curves, self._propose

	def __init__(self, system):    # must input a System() instance
		self.system = system
		volts = {}
		for ele in system:
			volts[ele.name] = []    # initial .dc
		self.volts = volts    # self.volts is a dict, {ele1:[v0,v1,...],ele2:[v0,v1...],...}

	def set_move_init(self, E_config=None, xm=None, axis_order=None):
		# 1. Based on the minimum point, use three eigen-axes as bases. You can manually set axis_order.
		# 2. Deploy dc change/unchange configuration. E_config=[[ele, ele...],[ele, ele,..]...], [] 
		# encloses the electrodes with the same voltage changes.

		# minimum xm
		# system.minimum() still has some problem, if it appears, you can manually input xm.
		if xm == None:
			xi = (0.,0.,0.)
			xm = self.system.minimum(xi)
		self.xms = [np.array(xm)]

		# eigenvector -> bases
		evalue, evector = self.system.modes(xm)
		if axis_order == None:
			# sort according to the x coordinate of eigenvectors. [::-1] in acsending order
			axis_order = evector[0].argsort()[::-1]
		evalue, evector = evalue[axis_order], evector[:,axis_order].T
		self.curves, self.basis = [evalue], evector
		ax_check = [(ax, ev) for ax, ev in zip("axial radial1 radial2".split(), evector)]

		# DC E_configuration
		self.E_config = E_config

		return xm, ax_check

	def construct_config(self,deltaV):
		# Internal method.

		config = {}
		for eles, dV in zip(self.E_config,deltaV):
			for ele in eles:
				config[ele] = self.volts[ele][-1]+dV
		return config

	def set_volts(self, config, change_rf=False, config_rf=None, save=False):
		# Set voltages for system. Dict config = {dc_name:dc_volt}, config_rf = {rf_name:rf_volt}.
		# RF electrodes could also have DC bais.

		for ele in config:
			self.system[ele].dc = config[ele]
		# if change_rf:
		# 	for ele in config_rf:
		# 		self.system[ele].rf = config_rf[ele]
		if save:
			for ele in config:
				self.volts[ele].append(config[ele])
			for ele in self.volts:    # self.volts.keys()
				if ele not in config:
					self.volts[ele].append(self.volts[ele][-1])    # keep the last value. 
		return None

	def volt_change(self, axis=1, l=200e-6, dis=10e-9, same_scale=False, method="force"):
		# 1. Change of voltages of electrodes in self.E_config(), save to self.volts
		# 
		# axis is the direction ion will move, 0,1,2-->axial, radial1, radial2. Default is along radial1. 
		# l is the length scale (=200um), dis is the distance that you intend to move. 

		# scale force E
		if method == "force":
			icurve = -1
			if same_scale: icurve = 0
			E_scale = self.curves[icurve][axis]*dis/l
			print("Force F at minimum with {:.2f}nm displacement: {} (eV/m)".format(dis/1e-9, E_scale/l))

		# solve voltage change deltaV
		Nele = len(self.E_config)    # If you only confine force, Nele = 3
		Id,Zero = np.eye(Nele),np.zeros(Nele)
		coef = np.zeros((3,Nele))	# coef is the coeffient matrix of equations. Nele electrodes as variables, 3d vector, so 3*Nele.
		xm_old = self.xms[-1]	# minimum of last move
		if method == "force":
			xm_evaluate = xm_old
			nonhomo = E_scale*self.basis[axis]
		elif method == "position":
			xm_evaluate = xm_old+dis/l*self.basis[axis]
			nonhomo = -self.system.potential(x=xm_evaluate,derivative=1)[0]    # system.potential
		for ith in range(Nele):    # ith is the column index of coef.
			for ele in self.E_config[ith]:
				coef[:,ith] += self.system[ele].potential(xm_evaluate,derivative=1)[0].T    # GridElectrode.potential
		deltaV = scipy_solve(coef,nonhomo.T)    # Corresponds to the order of E_config.
		# deltaV = [scipy_solve(coef,Ev*ev) for Ev,ev in zip(E_scale,self.basis)]
		return deltaV

	def fix_position():
		pass

	def xm_freq_change(self,deltaV):
		# solve next minimum and frequencies.

		xm_old = self.xms[-1]
		config = self.construct_config(deltaV)
		self.set_volts(config, save=False)
		xm_new = self.system.minimum(xm_old+0.01, method="Nelder-Mead")    # add +0.01 to aviod precision loss
		xm_offset = xm_new-xm_old
		evalue, evector = self.system.modes(xm_new)
		axis_order = evector[0].argsort()[::-1]		# sort evector x
		evalue, evector = evalue[axis_order], evector[:,axis_order].T
		freq_change = np.sqrt(evalue)-np.sqrt(self.curves[0])
		vec_change = evector-self.basis
		return xm_new, xm_offset, evalue, freq_change, vec_change	# They haven't scaled by l or m/q.

	def propose_move(self, axis=1, l=200e-6, dis=10e-9, same_scale=False, method="force"):
		# Use method to propose a configuration for moving, and save it in self._propose.

		deltaV = self.volt_change(axis=axis, l=l, dis=dis, same_scale=same_scale, method=method)
		xm_new, xm_offset, evalue, freq_change, vec_change = self.xm_freq_change(deltaV)
		self._propose = [deltaV,xm_new,evalue]	# save solved solution temporarily as a _propose.
		return deltaV, [xm_new,xm_offset], freq_change, vec_change

	def next_step(self,deltaV=None,accept=True):
		# accept the move or not. You can specify deltaV by youself, or it will use self._propose as deltaV
		# Pay attention, if you have used propose_move() or xm_freq_change(), the deltaV have been applied
		# to system.

		if deltaV == None:
			try:
				deltaV, xm_new, evalue = self._propose
				userown = False    # user own deltaV
			except AttributeError:
				warnings.warn("Propose a volt_change() first.")

		if accept:
			config = self.construct_config(deltaV)
			self.set_volts(config, save=True)
			if userown:
				xm_new, evalue = self.xm_freq_change(deltaV)[[0,2]]
			self.xms.append(xm_new)
			self.curves.append(evalue)
			# print("Move")
		else:
			# reset voltages back to erase the change due to xm_freq_change()
			self.set_volts(self.Volts[-1])		
			# print("Stay")

		del self._propose
		return 0

