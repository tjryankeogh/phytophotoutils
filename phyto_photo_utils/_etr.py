#!/usr/bin/env python

from ._equations import __calculate_residual_etr__, __calculate_residual_phi__, __calculate_residual_beta__, __calculate_residual_mbeta__, __calculate_alpha_model__, __calculate_beta_model__, __calculate_modified_alpha_model__, __calculate_modified_beta_model__, __calculate_bias__, __calculate_fit_errors__, __calculate_rmse__, __calculate_nrmse__
from numpy import mean, array, isnan, inf, repeat, nan, concatenate
from pandas import DataFrame, Series
from scipy.optimize import least_squares
import warnings

def calculate_amplitude_etr(fo, fm, sigma, par, alpha_phase=True, light_independent=True, dark_sigma=False, etrmax_fitting=True, serodio_sigma=False, light_step_size=None, last_steps_average=False, outlier_multiplier=3, return_data=False, bounds=True, alpha_lims=[0,4], etrmax_lims=[0,2000], method='trf', loss='soft_l1', f_scale=0.1, max_nfev=None, xtol=1e-9):
      
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
	alpha_phase : bool, default=True
		If True, will fit the data without photoinhibition. If False, will fit the data with the photoinhibition paramater β.
	light_independent : bool, default=True
		If True, will use the method outlined in Silsbe & Kromkamp 2012. 
	dark_sigma : bool
		If True, will use mean of σ\ :sub:`PSII` under 0 actinic light for calculation. If False, will use σ\ :sub:`PSII` and σ\ :sub:`PSII`' for calculation.
	etrmax_fitting : bool
		If True, will fit α\ :sup:`ETR` and ETR\ :sub:`max` and manually calculate E\ :sub:'k'. If False, will fit α\ :sup:`ETR` and E\ :sub:'k' and manually calculate ETR\ :sub:`max`.
	serodio_sigma : bool
		If True, will apply a Serodio correction for samples that have dark relaxation.
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
	
	Results are returned as pd.Series with the following parameters.

	etr_max : float
		The maximum electron transport rate.
	alpha : float
		The light limited slope of electron transport.
	ek : float
		The photoacclimation of ETR.
	alpha_bias : float
		The bias of the alpha fit. If alpha_phase is False, value is not returned.
	alpha_rmse : float
		The root mean squared error of the alpha fit. If alpha_phase is False, value is not returned.
	alpha_nrmse : float
		The normalised root mean squared error of the alpha fit. If alpha_phase is False, value is not returned.
	beta_bias : float
		The bias of the alpha fit. If alpha_phase is True, value is not returned.
	beta_rmse : float
		The root mean squared error of the alpha fit. If alpha_phase is True, value is not returned.
	beta_nrmse : float
		The normalised root mean squared error of the alpha fit. If alpha_phase is True, value is not returned.
	etrmax_err : float
		The fit error of ETR\ :sup:`max`. If etrmax_fitting is False, value returned is NaN.
	alpha_err : float
		The fit error of α\ :sub:`ETR`.
	ek_err : float
		The fit error of E\ :sub:`k`. If etrmax_fitting is True, value returned is NaN.
	alpha_nfev : np.array, dype=int, shape=[n,]
		The number of functional evaluations done on the alpha phase fitting routine.
	alpha_flag : np.array, dtype=int, shape=[n,]
		The code associated with the fitting routine success, positive values = SUCCESS, negative values = FAILURE.
		-1 : the ETR data is empty.
		0 : the maximum number of function evaluations is exceeded.
		1 : gtol termination condition is satisfied.
		2 : ftol termination condition is satisfied.
		3 : xtol termination condition is satisfied.
		4 : Both ftol and xtol termination conditions are satisfied.
	alpha_success : np.array, dtype=bool, shape=[n,]
		A boolean array reporting whether fit was successful (TRUE) or if not successful (FALSE)
	beta_nfev : np.array, dype=int, shape=[n,]
		The number of functional evaluations done on the beta phase fitting routine. If alpha_phase is True, value returned is NaN.
	beta_flag : np.array, dtype=int, shape=[n,]
		The code associated with the fitting routine success, positive values = SUCCESS, negative values = FAILURE. If alpha_phase is True, value returned is NaN.
		-1 : the ETR data is empty.
		0 : the maximum number of function evaluations is exceeded.
		1 : gtol termination condition is satisfied.
		2 : ftol termination condition is satisfied.
		3 : xtol termination condition is satisfied.
		4 : Both ftol and xtol termination conditions are satisfied.
	beta_success : np.array, dtype=bool, shape=[n,]
		A boolean array reporting whether fit was successful (TRUE) or if not successful (FALSE). If alpha_phase is True, value returned is NaN.
	data : [np.array, np.array]
		Optional, the final data used for the fitting procedure.


	Example
	-------
	>>> res = ppu.calculate_etr(fo, fm, sigma, par, return_data=False)
	"""

	warnings.simplefilter(action = "ignore", category = RuntimeWarning)

	fo = array(fo)
	fm = array(fm)
	fvfm = (fm - fo) / fm
	sigma = array(sigma)
	par = array(par)
	
	if serodio_sigma:
		df = DataFrame([par, fo, fm, sigma])
		df = df.T
		df.columns = ['par', 'fo', 'fm', 'sigma']
		# create means of each light step using last n measurements
		if last_steps_average:
			df = df.groupby('par').apply(lambda x: x.iloc[-3:].mean()).reset_index()
		else:
			# exclude outliers if more than mean ± (stdev * multiplier)
			grp = df.groupby(by='par')
			mn = grp.mean()
			std = grp.std()
			c = grp.count()
			ulim = repeat((mn.fm.values + std.fm.values * outlier_multiplier), c.fm.values)
			llim = repeat((mn.fm.values - std.fm.values * outlier_multiplier), c.fm.values)
			idx = []
			for i, items in enumerate(grp.indices.items()):
				idx.append(items[-1])

			idx = concatenate(idx, axis=0)

			# Create pandas DataFrame of upper and lower using original indexes of data
			mask = DataFrame([ulim, llim, idx]).T
			mask.columns = ['ulim','llim','index']
			mask = mask.set_index('index').sort_index()

			m = (df.fm.values > mask.ulim) | (df.fm.values < mask.llim)

			# Where condition is True, set values of value to NaN
			df.loc[m.values,'etr'] = nan

			# Create means per light step
			df = df.groupby('par').mean().reset_index()

	if light_step_size == 1:
		if light_independent:
			etr = fvfm / fvfm[0]
		else:
			if dark_sigma:
				etr = (par * mean(sigma[0]) * (fvfm / mean(fvfm[0]))) * 6.022e-3
		
			else:
				f_o = mean(fo[0]) / (mean(fvfm[0]) + (mean(fo[0])/fm))
				fqfv = (fm - fo) / (fm - f_o)
				etr = par * sigma * fqfv * 6.022e-3
		
		df = DataFrame([par, etr])
		df = df.T
		df.columns = ['par', 'etr']
	
	else:	
		lss = light_step_size - 1 # Python starts at 0
		
		if light_independent:
			if serodio_sigma:
				dff = DataFrame([par, fo, fm, sigma])
				dff = dff.T
				dff.columns = ['par', 'fo', 'fm', 'sigma']
				if last_steps_average:
					dff = dff.groupby('par').apply(lambda x: x.iloc[-3:].mean()).reset_index(drop=True)
				else:
					dff = dff.groupby('par').mean().reset_index()
				
				idx = dff.fm.idxmax() + 1
				sigma_max = dff.sigma.iloc[:idx].max()
				fo[:dff.fo.idxmax()] = dff.fo.max()
				fm[:dff.fm.idxmax()] = dff.fm.max()
				fvfm = (fm - fo) / fm
				etr = fvfm / mean(fvfm[0:lss])

			else:
				fvfm = (fm - fo) / fm 
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
			if alpha_phase:
				if etrmax_fitting:				
					if serodio_sigma:
						etr_max = popt.x[0] * sigma_max * 6.022e-3
						alpha = popt.x[1] * sigma_max * 6.022e-3
						ek = etr_max / alpha

					else:
						sigma_etr = mean(sigma[0:lss])
						etr_max = popt.x[0] * sigma_etr * 6.022e-3
						alpha = popt.x[1] * sigma_etr * 6.022e-3
						ek = etr_max / alpha

				else:
					
					if serodio_sigma:
						sigma = sigma_max
						ek = popt.x[0] 
						alpha = popt.x[1] * sigma_max * 6.022e-3
						etr_max = ek * alpha
					else:
						sigma_etr = mean(sigma[0:lss])
						ek = popt.x[0] 
						alpha = popt.x[1] * sigma_etr * 6.022e-3
						etr_max = ek * alpha

			else:
				eB = popt.x[0]
				a = popt.x[1]
				#m = E > ekb
				E2 = E#[m]
				P2 = P#[m]
				popt_beta = least_squares(__calculate_residual_mbeta__, p0, args=(E2, P2, a, eB), **opts)
				
				if serodio_sigma:
					sigma = sigma_max
					ek = popt.x[0]
					alpha = popt.x[1] * sigma * 6.022e-3
					etr_max = popt_beta.x[0] * sigma * 6.022e-3

				else:
					sigma = mean(sigma[0:lss])
					ek = popt.x[0]
					alpha = popt.x[1] * sigma * 6.022e-3
					etr_max = popt_beta.x[0] * sigma * 6.022e-3

		else:
			if alpha_phase:
				popt = least_squares(__calculate_residual_etr__, p0, args=(E, P), bounds=(bds), **opts)

				if etrmax_fitting:
					etr_max = popt.x[0]
					alpha = popt.x[1]
					ek = etr_max / alpha
				else:
					ek = popt.x[0]
					alpha = popt.x[1]
					etr_max = ek * alpha

			else:
				popt = least_squares(__calculate_residual_etr__, p0, args=(E, P), bounds=(bds), **opts)
				eB = popt.x[0]
				a = popt.x[1]

				E2 = E
				P2 = P
				popt_beta = least_squares(__calculate_residual_beta__, p0, args=(E2, P2, a, eB), **opts)
				
				ek = popt.x[0]
				alpha = popt.x[1]
				etr_max = popt_beta.x[0]
		

		if alpha_phase:
			if light_independent:
				sol = __calculate_modified_alpha_model__(E, *popt.x)
			else:
				sol = __calculate_alpha_model__(E, *popt.x)
			
			alpha_bias = __calculate_bias__(sol, P)
			alpha_rmse = __calculate_rmse__(popt.fun, P)
			alpha_nrmse = __calculate_nrmse__(popt.fun, P)				
			alpha_perr = __calculate_fit_errors__(popt.jac, popt.fun)

			beta_bias = nan
			beta_rmse = nan
			beta_nrmse = nan
			beta_nfev = nan
			beta_flag = nan
			beta_success = nan

			if max_nfev is None:
				alpha_nfev = popt.nfev
			else:
				alpha_nfev = max_nfev
			
			alpha_flag = popt.status
			alpha_success = popt.success
		
		else:
			if light_independent:
				sol = __calculate_modified_alpha_model__(E, *popt.x)
				solb = __calculate_modified_beta_model__(E2, *popt_beta.x, a, eB)
			else:
				sol = __calculate_alpha_model__(E, *popt.x)
				solb = __calculate_beta_model__(E2, *popt_beta.x, a, eB)
			
			alpha_bias = __calculate_bias__(sol, P)
			alpha_rmse = __calculate_rmse__(popt.fun, P)
			alpha_nrmse = __calculate_nrmse__(popt.fun, P)					
			alpha_perr = __calculate_fit_errors__(popt.jac, popt.fun)

			beta_bias = __calculate_bias__(solb, P2)
			beta_rmse = __calculate_rmse__(popt_beta.fun, P2)	
			beta_nrmse = __calculate_nrmse__(popt_beta.fun, P2)			
			beta_perr = __calculate_fit_errors__(popt_beta.jac, popt_beta.fun)

			if max_nfev is None:
				alpha_nfev = popt.nfev
				beta_nfev = popt_beta.nfev
			else:
				alpha_nfev = max_nfev
				beta_nfev = max_nfev
			
			alpha_flag = popt.status
			alpha_success = popt.success


			beta_flag = popt_beta.status
			beta_success = popt_beta.success
		
		if etrmax_fitting:
			etr_max_err = alpha_perr[0]
			alpha_err = alpha_perr[1]
			ek_err = nan
		else:
			if alpha_phase:
				ek_err = alpha_perr[0]
				alpha_err = alpha_perr[1]
				etr_max_err = nan
			else:
				ek_err = alpha_perr[0]
				alpha_err = alpha_perr[1]
				etr_max_err = beta_perr[0]
	
	except Exception:
		print(('Unable to calculate fit, skipping sequence'))
		if alpha_phase:
			alpha_flag = -1
			alpha_success = 'False'
			
			if etrmax_fitting:
				etr_max, alpha, ek, alpha_bias, alpha_rmse, alpha_nrmse, etr_max_err, alpha_err, alpha_nfev = repeat(nan, 9)
			else:
				etr_max, alpha, ek, alpha_bias, alpha_rmse, alpha_nrmse, alpha_err, ek_err, alpha_nfev = repeat(nan, 9)
		else:
			beta_flag = -1
			beta_success = 'False'

			if etrmax_fitting:
				etr_max, alpha, ek, beta_bias, beta_rmse, beta_nrmse, etr_max_err, alpha_err, beta_nfev = repeat(nan, 9)
			else:
				etr_max, alpha, ek, beta_bias, beta_rmse, beta_nrmse, alpha_err, ek_err, beta_nfev = repeat(nan, 9)
	
	if return_data:
		if alpha_phase:	
			if etrmax_fitting:
				results = Series([etr_max, alpha, ek, alpha_bias, alpha_rmse, alpha_nrmse, etr_max_err, alpha_err, alpha_nfev, alpha_flag, alpha_success])
			else:
				results = Series([etr_max, alpha, ek, alpha_bias, alpha_rmse, alpha_nrmse, alpha_err, ek_err, alpha_nfev, alpha_flag, alpha_success])
			data = [E,P]
			return results, [E,P]
		else:
			if etrmax_fitting:
				results = Series([etr_max, alpha, ek, beta_bias, beta_rmse, beta_nrmse, etr_max_err, alpha_err, beta_nfev, beta_flag, beta_success])
			else:
				results = Series([etr_max, alpha, ek, beta_bias, beta_rmse, beta_nrmse, alpha_err, ek_err, beta_nfev, beta_flag, beta_success])
			data = [E,P]
			return results, [E,P]
			
	else:
		if alpha_phase:	
			if etrmax_fitting:
				results = Series([etr_max, alpha, ek, alpha_bias, alpha_rmse, alpha_nrmse, etr_max_err, alpha_err, alpha_nfev, alpha_flag, alpha_success])
			else:
				results = Series([etr_max, alpha, ek, alpha_bias, alpha_rmse, alpha_nrmse, alpha_err, ek_err, alpha_nfev, alpha_flag, alpha_success])
			return results

		else:
			if etrmax_fitting:
				results = Series([etr_max, alpha, ek, beta_bias, beta_rmse, beta_nrmse, etr_max_err, alpha_err, beta_nfev, beta_flag, beta_success])
			else:
				results = Series([etr_max, alpha, ek, beta_bias, beta_rmse, beta_nrmse, alpha_err, ek_err, beta_nfev, beta_flag, beta_success])
			return results


