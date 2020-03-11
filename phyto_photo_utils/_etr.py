#!/usr/bin/env python

from ._equations import __calculate_residual_etr__, __calculate_residual_phi__, __calculate_Webb_model__, __calculate_modified_Webb_model__, __calculate_bias__, __calculate_fit_errors__, __calculate_rmse__
from numpy import mean, array, isnan, inf, repeat, nan, concatenate
from pandas import DataFrame
from scipy.optimize import least_squares
import warnings

def calculate_etr(fo, fm, sigma, par, light_independent=True, dark_sigma=False, light_step_size=None, last_steps_average=False, outlier_multiplier=3, return_data=False, bounds=True, alpha_lims=[0,4], etrmax_lims=[0,2000], method='trf', loss='soft_l1', f_scale=0.1, max_nfev=None, xtol=1e-9):
      
	"""
	
	Convert the processed transient data into an electron transport rate and perform a fit using the Webb Model.

	Parameters
	----------
	fo : np.array, dtype=float, shape=[n,]
		The minimum fluorescence level.
	fm : np.array, dtype=float, shape=[n,] 
		The maximum fluorescence level.
	sigma : np.array, dtype=float, shape=[n,] 
		The effective absorption cross-section of PSII in Å\ :sup:`2`.
	par : np.array, dtype=float, shape=[n,]
		The actinic light levels in μE m\ :sup:`2` s\ :sup:`-1`.
	light_independent : bool, default=True
		If True, will use the method outlined in Silsbe & Kromkamp 2012. 
	dark_sigma : bool
		If True, will use mean of σ\ :sub:`PSII` under 0 actinic light for calculation. If False, will use σ\ :sub:`PSII` and σ\ :sub:`PSII`' for calculation.
	light_step_size : int
		The number of measurements for initial light step.
	last_steps_average : bool, default=False,
		If True, means will be created from the last 3 measurements per light step. Else, mean will be created from entire light step excluding outliers.
	outlier_multiplier : int, default=3
		The multiplier to apply to the standard deviation for determining the upper and lower limits.
	return_data : bool, default=False
		If True, will return the final data used for the fit.
	bounds : bool, default=True
		If True, will set lower and upper limit bounds for the estimation, not suitable for methods 'lm'.
	alpha_lims : [int, int], default=[0,4]
		The lower and upper limit bounds for fitting α\ :sup:`ETR`.
	etrmax_lims : [int, int], default=[0,2000]
	 	The lower and upper limit bounds for fitting ETR\ :sub:`max`.
	fit_method : str, default='trf'
		The algorithm to perform minimization. See ``scipy.optimize.least_squares`` documentation for more information on non-linear least squares fitting options.
	loss_method : str, default='soft_l1'
		The loss function to be used. Note: Method ‘lm’ supports only ‘linear’ loss. See ``scipy.optimize.least_squares`` documentation for more information on non-linear least squares fitting options.
	fscale : float, default=0.1
	 	The soft margin value between inlier and outlier residuals. See ``scipy.optimize.least_squares`` documentation for more information on non-linear least squares fitting options.
	max_nfev : int, default=None		
		The number of iterations to perform fitting routine. If None, the value is chosen automatically. See ``scipy.optimize.least_squares`` documentation for more information on non-linear least squares fitting options.
	xtol : float, default=1e-9			
		The tolerance for termination by the change of the independent variables. See ``scipy.optimize.least_squares`` documentation for more information on non-linear least squares fitting options.

	Returns
	-------
	
	etr_max : float
		The maximum electron transport rate.
	alpha : float
		The light limited slope of electron transport.
	ek : float
		The photoacclimation of ETR.
	rsq : np.array, dtype=float, shape=[n,]
		The r\ :sup:`2` value.
	bias : np.array, dtype=float, shape=[n,]
		The bias of fit.
	chi : np.array, dtype=float, shape=[n,]
		The chi-squared goodness of fit.
	rchi : np.array, dtype=float, shape=[n,]
		The reduced chi-squared goodness of fit.
	rmse : np.array, dtype=float, shape=[n,]
		The root mean squared error of the fit.
	etrmax_err : float
		The fit error of ETR\ :sup:`max`.
	alpha_err : float
		The fit error of α\ :sub:`ETR`.
	data : [np.array, np.array]
		Optional, the final data used for the fitting procedure.


	Example
	-------
	>>> etr_max, alpha, ek, rsq, bias, chi, etr_max_err, alpha_err = ppu.calculate_e_dependent_etr(fo, fm, fvfm, sigma, par, return_data=False)
	"""

	warnings.simplefilter(action = "ignore", category = RuntimeWarning)

	fo = array(fo)
	fm = array(fm)
	fvfm = (fm - fo) / fm
	sigma = array(sigma)
	par = array(par)

	lss = light_step_size - 1 # Python starts at 0
	
	if light_independent:
		etr = fvfm / mean(fvfm[0:lss])
	else:
		if dark_sigma:
			etr = (par * mean(sigma[0:lss]) * (fvfm / mean(fvfm[0:lss]))) * 6.022e-3
	
		else:
			f_o = mean(fo[0:lss]) / (mean(fvfm[0:lss]) + (mean(fo[0:lss])/fm))
			fqfv = (fm - fo) / (fm - f_o)
			etr = par * sigma * fqfv * 6.022e-3

	df = DataFrame([par, etr])
	df = df.T
	df.columns = ['par', 'etr']
	# create means of each light step using last n measurements
	if last_steps_average:
		df = df.groupby('par').apply(lambda x: x.iloc[-3:].mean()).reset_index()
	else:
		# exclude outliers if more than mean ± (stdev * multiplier)
		grp = df.groupby(by='par')
		mn = grp.mean()
		std = grp.std()
		c = grp.count()
		ulim = repeat((mn.etr.values + std.etr.values * outlier_multiplier), c.etr.values)
		llim = repeat((mn.etr.values - std.etr.values * outlier_multiplier), c.etr.values)
		idx = []
		for i, items in enumerate(grp.indices.items()):
			idx.append(items[-1])

		idx = concatenate(idx, axis=0)

		# Create pandas DataFrame of upper and lower using original indexes of data
		mask = DataFrame([ulim, llim, idx]).T
		mask.columns = ['ulim','llim','index']
		mask = mask.set_index('index').sort_index()

		m = (df.etr.values > mask.ulim) | (df.etr.values < mask.llim)

		# Where condition is True, set values of value to NaN
		df.loc[m.values,'etr'] = nan

		# Create means per light step
		df = df.groupby('par').mean().reset_index()

	# Define data for fitting and estimates of ETRmax and alpha
	P = array(df.etr)
	E = array(df.par)

	p0 = [1000, 1.5]

	# Mask missing data
	if light_independent:
		mask = isnan(P) | isnan(E) | (P < 0) | (E == 0)
	else:
		mask = isnan(P) | isnan(E)
	
	E = E[~mask]
	P = P[~mask]
	
	if bounds:
		bds = [etrmax_lims[0], alpha_lims[0]],[etrmax_lims[1], alpha_lims[1]]
		if (bds[0][0] > bds[1][0]) | (bds[0][1] > bds[1][1]):
			print('Lower bounds greater than upper bounds - fitting with no bounds.')
			bds = [-inf, inf]
	else:
		bds = [-inf, inf]

	if max_nfev is None:
		opts = {'method':method, 'loss':loss, 'f_scale':f_scale, 'xtol':xtol} 
	else:
		opts = {'method':method, 'loss':loss, 'f_scale':f_scale, 'max_nfev':max_nfev, 'xtol':xtol} 

	try:
		if light_independent:
			popt = least_squares(__calculate_residual_phi__, p0, args=(E, P), bounds=(bds), **opts)
		else:
			popt = least_squares(__calculate_residual_etr__, p0, args=(E, P), bounds=(bds), **opts)

		if light_independent:
			etr_max = popt.x[0]
			alpha = popt.x[1]
			sigma = mean(sigma[0:lss])
			etr_max *= sigma * 6.022e-3
			alpha *= sigma * 6.022e-3
		else:
			etr_max = popt.x[0]
			alpha = popt.x[1]

		ek = etr_max / alpha
		
		if light_independent:
			sol = __calculate_modified_Webb_model__(E, *popt.x)
		else:
			sol = __calculate_Webb_model__(E, *popt.x)

		bias = __calculate_bias__(sol, P)
		rmse = __calculate_rmse__(popt.fun, P)				
		perr = __calculate_fit_errors__(popt.jac, popt.fun)
		etr_max_err = perr[0]
		alpha_err = perr[1]
	
	except Exception:
		print(('Unable to calculate fit, skipping sequence'))
		etr_max, alpha, ek, bias, rmse, etr_max_err, alpha_err = repeat(nan, 7)
	
	if return_data:
		return etr_max, alpha, ek, bias, rmse, etr_max_err, alpha_err, [E,P]
	else:
		return etr_max, alpha, ek, bias, rmse, etr_max_err, alpha_err
