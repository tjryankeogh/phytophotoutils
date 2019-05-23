#!/usr/bin/env python
"""
@package phyto_photo_utils.saturation
@file phyto_photo_utils/saturation.py
@author Thomas Ryan-Keogh
@brief module containing the functions for plotting.
"""

def plot_saturation_data(fyield, pfd, fo=None, fm=None, sigma=None, ro=None, rsq=None):

	"""
	Parameters
	----------

	fyield : np.array, dtype=float, shape=[n,]
		The raw fluorescence yield data.
	pfd : np.array, dtype=float, shape=[n,]
		The photon flux density.
	fo : float, default=None
		The minimum fluorescence value.
	fm : float, default=None
		The maximum fluorescence value.
	sigma: float, default=None
		The effective absorption cross-section value.
	ro: float, default=None
		The connectivity coefficient.
	rsq: float, default=None
		The r-squared value of the fit.

	Returns
	-------

	ax : object
		a matplotlib figure object
	"""


	from ._equations import __fit_kolber__
	from matplotlib.pyplot import subplots
	from numpy import arange, array

	if ro is None:
		params = [fo, fm, sigma]
	else:
		params = [fo, fm, sigma, ro]

	fyield = array(fyield)
	pfd = array(pfd)
	fvfm = (params[1] - params[0])/params[1]
	x = arange(0,len(fyield),1)

	fig, ax = subplots(1, 1, figsize=[5,4], dpi=90)

	ax.plot(x, fyield, marker='o', lw=0, label='Raw Data', color='0.5')
	formula = r"F$_v$/F$_m$ = {:.2f}""\n""$\u03C3$$_{{PSII}}$ = {:.2f}; r$^2$ = {:.2f}".format(fvfm, sigma, rsq)
	ax.plot(x, __fit_kolber__(pfd, *params), color='k', label='{}'.format(formula))
	ax.legend()
	ax.set_ylabel('Fluorescence Yield')
	ax.set_xlabel('Flashlet Number')

	return ax

def plot_relaxation_data(fyield, seq_time, fo_relax=None, fm_relax=None, tau=None, alpha=None, rsq=None):
	"""
	Parameters
	----------

	fyield : np.array, dtype=float, shape=[n,]
		The raw fluorescence yield data.
	seq_time : np.array, dtype=float, shape=[n,]
		The time of the flashlet measurements.
	fo_relax : float, default=None
		The minimum fluorescence value in the relaxation phase.
	fm_relax : float, default=None
		The maximum fluorescence value in the relaxation phase.
	tau: float, default=None
		The 
	alpha: float, default=None
		The
	rsq: float, default=None
		The r-squared value of the fit.

	Returns
	-------

	ax : object
		a matplotlib figure object
	"""
	from ._equations import __fit_single_relaxation__, __fit_triple_relaxation__
	from matplotlib.pyplot import subplots
	from numpy import arange, array

	fyield = array(fyield)
	seq_time = array(seq_time)

	x = arange(0,len(fyield),1)+100

	fig, ax = subplots(1, 1, figsize=[5,4], dpi=90)

	ax.plot(x, fyield, marker='o', lw=0, label='Raw Data', color='0.5')
	ax.set_ylabel('Fluorescence Yield')
	ax.set_xlabel('Flashlet Number')

	if alpha is None:
		params = [fo_relax, fm_relax, tau]
	
		formula = r"F$_o$$_{{Relax}}$ = {:.2f}; F$_m$$_{{Relax}}$ = {:.2f}""\n""$\U0001D70F$ = {:.2f}; r$^2$ = {:.2f}".format(fo_relax, fm_relax, tau, rsq)
		ax.plot(x, __fit_single_relaxation__(seq_time, *params), color='k', label='{}'.format(formula))

	else:
		params = [fo_relax, fm_relax, alpha[0], tau[0], alpha[1], tau[1], alpha[2], tau[2]]

		formula = r"F$_o$$_{{Relax}}$ = {:.2f}; F$_m$$_{{Relax}}$ = {:.2f}""\n""$\U0001D70F$$_1$ = {:.2f}; $\U0001D70F$$_2$ = {:.2f}""\n""$\U0001D70F$$_3$ = {:.2f}; r$^2$ = {:.2f}".format(fo_relax, fm_relax, tau[0], tau[1], tau[2], rsq)
		ax.plot(x, __fit_triple_relaxation__(seq_time, *params), color='k', label='{}'.format(formula))

	ax.legend()


	return ax


def plot_fluorescence_light_curve(par, etr, etrmax=None, alpha=None, rsq=None, sigma=None, phi=False):
	
	"""
	Parameters
	----------

	par : np.array, dtype=float
		the actinic light data from the fluorescence light curve
	etr : np.array, dtype=float
		the electron transport rate data
	etrmax : float, default=None
		the maximum electron transport rate
	alpha : float, default=None
		the light limited slope of electron transport
	rsq: float, default=None
		the r-squared value of the fit
	sigma: float, default=None
		the effective absorption-cross section
	phi: bool, default=False
		if True, etr data is phi and the modified Webb et al. (1974) fit is used

	Returns
	-------

	ax : object
		a matplotlib figure object
	"""

	from ._equations import __calculate_Webb_model__, __calculate_modified_Webb_model__
	from matplotlib.pyplot import subplots
	from numpy import array

	x = array(par)
	y = array(etr)
	

	fig, ax = subplots(1, 1, figsize=[5,4], dpi=90)

	ax.plot(x, y, marker='o', lw=0, label='Raw Data', color='0.5')
	formula = r"ETR$_{{max}}$ = {:.2f}""\n""$\u03B1$$^{{ETR}}$ = {:.2f}""\n"" r$^2$ = {:.2f}".format(etrmax, alpha, rsq)
	ax.set_xlabel('Actinic Light ($\u03BC$mol photons m$^{-2}$ s${-1}$)')
	
	if phi == False:
		params = [etrmax, alpha]
		ax.plot(x, __calculate_Webb_model__(x, *params), color='k', label='{}'.format(formula))
		ax.set_ylabel('ETR (mol e$^{-1}$ mol RCII$^{-1}$ s$^{-1}$)')
		ax.legend()
	
	else:
		sig = sigma*6.022e-3
		params = [etrmax/sig, alpha/sig]
		ax.plot(x, __calculate_modified_Webb_model__(x, *params), color='k', label='{}'.format(formula))
		ax.set_xscale('log')
		ax.set_ylabel('\u03D5')
		ax.legend()
	
	return ax
