#!/usr/bin/env python

from ._equations import __fit_kolber_p__, __fit_kolber_nop__, __fit_single_relaxation__, __fit_triple_relaxation__, __calculate_Webb_model__, __calculate_modified_Webb_model__
from matplotlib.pyplot import subplots, close
from numpy import arange, array

def plot_saturation_data(fyield, pfd, fo=None, fm=None, sigma=None, ro=None, rmse=None):

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
		The effective absorption cross-section value in Å\ :sup:`2`.
	ro: float, default=None
		The connectivity coefficient.
	rmse: float, default=None
		The RMSE value of the fit.

	Returns
	-------

	ax : object
		A matplotlib figure object.

	Example
	-------
	>>> plot_saturation_data(fyield, pfd, fo=fo, fm=fm, sigma=sigma, ro=None, rmse=rmse)
	"""

	fyield = array(fyield)
	pfd = array(pfd)
	fvfm = (fm - fo)/fm
	x = arange(0,len(fyield),1)

	close()

	fig, ax = subplots(1, 1, figsize=[5,4], dpi=90)

	ax.plot(x, fyield, marker='o', lw=0, label='Raw Data', color='0.5')
	ax.set_ylabel('Fluorescence Yield')
	ax.set_xlabel('Flashlet Number')

	if ro is None:
		params = [fo, fm, sigma]
		formula = r"F$_v$/F$_m$ = {:.2f}""\n""$\u03C3$$_{{PSII}}$ = {:.2f}; RMSE = {:.2f}".format(fvfm, sigma, rmse)
		ax.plot(x, __fit_kolber_nop__(pfd, *params), color='k', label='{}'.format(formula))
	else:
		params = [fo, fm, sigma, ro]
		formula = r"F$_v$/F$_m$ = {:.2f}""\n""$\u03C3$$_{{PSII}}$ = {:.2f}; RMSE = {:.2f}".format(fvfm, sigma, rmse)
		ax.plot(x, __fit_kolber_p__(pfd, *params), color='k', label='{}'.format(formula))

	ax.legend()
	

	return ax

def plot_relaxation_data(fyield, seq_time, fo_relax=None, fm_relax=None, tau=None, alpha=None, rmse=None):
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
		The rate of reoxidation in μs.
	alpha: float, default=None
		The ratio of reoxidisation components.
	rmse: float, default=None
		The RMSE value of the fit.

	Returns
	-------

	ax : object
		A matplotlib figure object.
	
	Example
	-------
	>>> ppu.plot_relaxation_data(fyield, seq_time, fo_relax=fo_r, fm_relax=fm_r, tau=(tau1, tau2, tau3), alpha=(alpha1, alpha2, alpha3), rsq=rsq)
	"""

	fyield = array(fyield)
	seq_time = array(seq_time)

	x = arange(0,len(fyield),1)+100

	close()

	fig, ax = subplots(1, 1, figsize=[5,4], dpi=90)

	ax.plot(x, fyield, marker='o', lw=0, label='Raw Data', color='0.5')
	ax.set_ylabel('Fluorescence Yield')
	ax.set_xlabel('Flashlet Number')

	if alpha is None:
		params = [fo_relax, fm_relax, tau]
	
		formula = r"F$_o$$_{{Relax}}$ = {:.2f}; F$_m$$_{{Relax}}$ = {:.2f}""\n""$\U0001D70F$ = {:.2f}; RMSE = {:.2f}".format(fo_relax, fm_relax, tau, rmse)
		ax.plot(x, __fit_single_relaxation__(seq_time, *params), color='k', label='{}'.format(formula))

	else:
		params = [fo_relax, fm_relax, alpha[0], tau[0], alpha[1], tau[1], alpha[2], tau[2]]

		formula = r"F$_o$$_{{Relax}}$ = {:.2f}; F$_m$$_{{Relax}}$ = {:.2f}""\n""$\U0001D70F$$_1$ = {:.2f}; $\U0001D70F$$_2$ = {:.2f}""\n""$\U0001D70F$$_3$ = {:.2f}; RMSE = {:.2f}".format(fo_relax, fm_relax, tau[0], tau[1], tau[2], rmse)
		ax.plot(x, __fit_triple_relaxation__(seq_time, *params), color='k', label='{}'.format(formula))

	ax.legend()


	return ax


def plot_fluorescence_light_curve(par, etr, etrmax=None, alpha=None, rmse=None, sigma=None, phi=False):
	
	"""
	Parameters
	----------

	par : np.array, dtype=float
		The actinic light data from the fluorescence light curve.
	etr : np.array, dtype=float
		The electron transport rate data.
	etrmax : float, default=None
		The maximum electron transport rate.
	alpha : float, default=None
		The light limited slope of electron transport.
	rmse: float, default=None
		The RMSE value of the fit.
	sigma: float, default=None
		The effective absorption-cross section.
	phi: bool, default=False
		If True, etr data is phi and the modified Webb et al. (1974) fit is used.

	Returns
	-------

	ax : object
		A matplotlib figure object.

	Example
	-------
	>>> ppu.plot_fluorescence_light_curve(par, etr, etrmax=etr_max, alpha=alpha, rsq=rsq, sigma=sigma, phi=True)
	"""

	x = array(par)
	y = array(etr)
	
	close()
	fig, ax = subplots(1, 1, figsize=[5,4], dpi=90)

	ax.plot(x, y, marker='o', lw=0, label='Raw Data', color='0.5')
	formula = r"ETR$_{{max}}$ = {:.2f}""\n""$\u03B1$$^{{ETR}}$ = {:.2f}""\n"" RMSE = {:.2f}".format(etrmax, alpha, rmse)
	ax.set_xlabel('Actinic Light ($\u03BC$mol photons m$^{-2}$ s${-1}$)')
	
	if phi == False:
		params = [etrmax, alpha]
		ax.plot(x, __calculate_Webb_model__(x, *params), color='k', label='{}'.format(formula))
		ax.set_ylabel('ETR (mol e$^{-1}$ mol RCII$^{-1}$ s$^{-1}$)')
	
	else:
		if sigma is None:
			print('UserError - no sigma data provided.')
		sig = sigma*6.022e-3
		params = [etrmax/sig, alpha/sig]
		ax.plot(x, __calculate_modified_Webb_model__(x, *params), color='k', label='{}'.format(formula))
		ax.set_xscale('log')
		ax.set_ylabel('\u03D5')
	
	ax.legend()

	
	return ax
