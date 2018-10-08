# Created by wwc, 07/24/2018


import logging

import numpy as np
import matplotlib.pyplot as plt
import matplotlib


class DataVisual():

	@property
	def figax(self):
		return self._figax

	@figax.setter
	def figax(self,img):
		if len(img)<2:
			raise ValueError("Two arguments")
		cond = isinstance(img[0],matplotlib.figure.Figure) and isinstance(img[1],matplotlib.axes.Axes)
		if cond:
			self._figax = (img[0], img[1])    # (fig,ax)
		else:
			raise TypeError("Need one Figure object and one Axes object.")

	def to_Result():
		pass

	def section_contour(self,coord,data,normal='x',color=plt.cm.Blues):
		try:
			fig,ax = getattr(self,'figax',None)    # If fig, ax aren't set, raise AttributeError.
		except AttributeError:
			raise AttributeError("Please set figure and axes objects first.")

		

