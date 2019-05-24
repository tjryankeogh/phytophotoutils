#!/usr/bin/env python

def calculate_e_dependent_etr(fo, fm, fvfm, sigma, par, dark_sigma=True, light_step_size=None, outlier_multiplier=3, return_data=False, bounds=True, alpha_lims=[0,4], etrmax_lims=[0,2000], method='trf', loss='soft_l1', f_scale=0.1, max_nfev=1000, xtol=1e-9):
      
	"""
	
	Process the raw transient data and perform the no connectivity saturation model.

	Parameters
	----------
	fo : np.array, dtype=float, shape=[n,]
		The minimum fluorescence yield
	fm : np.array, dtype=float, shape=[n,] 
		The maximum fluorescence yield
	fvfm : np.array, dtype=float, shape=[n,]
		The photosynthetic efficiency
	sigma : np.array, dtype=float, shape=[n,] 
		The effective absorption cross-section of PSII in Å\ :sup:`2`
	par : np.array, dtype=float, shape=[n,]
		The actinic light levels in μE m\ :sup:`2` s\ :sup:`-1`
	dark_sigma : bool
		If True, will use mean of σ\ :sub:`PSII` under 0 actinic light for calculation. If False, will use σ\ :sub:`PSII` and σ\ :sub:`PSII`' for calculation
	light_step_size : int
		The number of measurements for initial light step
	outlier_multiplier : int, default=3
		The multiplier to apply to the standard deviation for determining the upper and lower limits
	return_data : bool, default=False
		If True, will return the final data used for the fit
	bounds : bool, default=True
		If True, will set lower and upper limit bounds for the estimation, not suitable for methods 'lm'
	alpha_lims : [int, int], default=[0,4]
		The lower and upper limit bounds for fitting α\ :sup:`ETR`
	etrmax_lims : [int, int], default=[0,2000]
	 	The lower and upper limit bounds for fitting ETR\ :sub:`max`
	fit_method : str, default='trf'
		The algorithm to perform minimization. See ``scipy.optimize.least_squares`` documentation for more information on non-linear least squares fitting options.
	loss_method : str, default='soft_l1'
		The loss function to be used. Note: Method ‘lm’ supports only ‘linear’ loss. See ``scipy.optimize.least_squares`` documentation for more information on non-linear least squares fitting options.
	fscale : float, default=0.1
	 	The soft margin value between inlier and outlier residuals. See ``scipy.optimize.least_squares`` documentation for more information on non-linear least squares fitting options.
	max_nfev : int, default=100			
		The number of iterations to perform fitting routine.
	xtol : float, default=1e-9			
		The tolerance for termination by the change of the independent variables. See ``scipy.optimize.least_squares`` documentation for more information on non-linear least squares fitting options.

	Returns
	-------
	
	etr_max : float
		The maximum electron transport rate
	alpha : float
		The light limited slope of electron transport
	ek : float
		The photoacclimation of ETR
	rsq : float
		The r\ :sup:`2` value
	bias : float
		The bias of the fit
	chi : float
		The chi-squared goodness of the fit
	etrmax_err : float
		The fit error of ETR\ :sup:`max`
	alpha_err : float
		The fit error of α\ :sub:`ETR`
	data : [np.array, np.array]
		Optional, the final data used for the fitting procedure


	Example
	-------
	>>> etr_max, alpha, ek, rsq, bias, chi, etr_max_err, alpha_err = ppu.calculate_e_dependent_etr(fo, fm, fvfm, sigma, par, return_data=False)
	"""
	from ._equations import __calculate_Webb_model__, __calculate_rsquared__, __calculate_bias__, __calculate_chisquared__, __calculate_fit_errors__
	from numpy import mean, array, isnan, inf, repeat, nan, concatenate
	from pandas import DataFrame
	from scipy.optimize import least_squares 

	fvfm = array(fvfm)
	sigma = array(sigma)
	par = array(par)

	lss = light_step_size - 1 # Python starts at 0
	
	if dark_sigma:
		etr = (par * mean(sigma[0:lss]) * (fvfm / mean(fvfm[0:lss]))) * 6.022e-3
	
	else:
		f_o = mean(fo[0:lss]) / (mean(fvfm[0:lss]) + (mean(fo[0:lss])/fm))
		fqfv = (fm - fo) / (fm - f_o)
		etr = par * sigma * fqfv * 6.022e-3

	df = DataFrame([par, etr])
	df = df.T
	df.columns = ['par', 'etr']

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
	#TO DO apply function of excluding outliers from means

	# Define data for fitting and estimates of ETRmax and alpha
	P = array(df.etr)
	E = array(df.par)

	p0 = [1000, 2]

	# Mask missing data
	mask = isnan(P)
	E = E[~mask]
	P = P[~mask]
		
	def residual(p, E, P):
		return P - __calculate_Webb_model__(E, *p)
	
	bds = [-inf, inf]
	if bounds:
			bds = [etrmax_lims[0], alpha_lims[0]],[etrmax_lims[1], alpha_lims[1]]

	# See scipy.optimize.least_squares documentation for more information on non-linear least squares fitting options
	opts = {'method':method,'loss':loss, 'f_scale':f_scale, 'max_nfev':max_nfev, 'xtol':xtol} 

	popt = least_squares(residual, p0, args=(E, P), bounds=(bds), **opts)
    
	etr_max = popt.x[0]
	alpha = popt.x[1]
	ek = etr_max / alpha

	res = P - __calculate_Webb_model__(E, *popt.x)
	rsq = __calculate_rsquared__(res, P)
	bias = __calculate_bias__(res[1:], P[1:])
	chi = __calculate_chisquared__(res[1:], P[1:])		
	perr = __calculate_fit_errors__(popt, res)
	etr_max_err = perr[0]
	alpha_err = perr[1]

	return etr_max, alpha, ek, rsq, bias, chi, etr_max_err, alpha_err, [E,P]


def calculate_e_independent_etr(fvfm, sigma, par, light_step_size=None, outlier_multiplier=3, return_data=False, bounds=True, alpha_lims=[0,4], etrmax_lims=[0,2000], method='trf', loss='soft_l1', f_scale=0.1, max_nfev=1000, xtol=1e-9):
      
	"""
	
	Process the raw transient data and perform the no connectivity saturation model.

	Parameters
	----------
	fvfm : np.array, dtype=float, shape=[n,] 
		The photosynthetic efficiency
	sigma : np.array, dtype=float, shape=[n,] 
		The effective absorption cross-section of PSII in Å\ :sup:`2`
	par : np.array, dtype=float, shape=[n,]
		The actinic light levels in μE m\ :sup:`2` s\ :sup:`-1`
	dark_sigma : bool
		If True, will use mean of σ\ :sub:`PSII` under 0 actinic light for calculation. If False, will use σ\ :sub:`PSII` and σ\ :sub:`PSII`' for calculation
	light_step_size : int
		The number of measurements for initial light step
	outlier_multiplier : int, default=3
		The multiplier to apply to the standard deviation for determining the upper and lower limits
	return_data: bool, default=False
		If True, will return the final data used for the fit
	bounds: bool, default=True
		If True, will set lower and upper limit bounds for the estimation, not suitable for methods 'lm'
	alpha_lims: [int, int], default=[0,4]
		The lower and upper limit bounds for fitting α\ :sup:`ETR`
	etrmax_lims: [int, int], default=[0,2000]
	 	The lower and upper limit bounds for fitting ETR\ :sub:`max`
	fit_method : str, default='trf'
		The algorithm to perform minimization. See ``scipy.optimize.least_squares`` documentation for more information on non-linear least squares fitting options.
	loss_method : str, default='soft_l1'
		The loss function to be used. Note: Method ‘lm’ supports only ‘linear’ loss. See ``scipy.optimize.least_squares`` documentation for more information on non-linear least squares fitting options.
	fscale : float, default=0.1
	 	The soft margin value between inlier and outlier residuals. See ``scipy.optimize.least_squares`` documentation for more information on non-linear least squares fitting options.
	max_nfev : int, default=100			
		The number of iterations to perform fitting routine.
	xtol : float, default=1e-9			
		The tolerance for termination by the change of the independent variables. See ``scipy.optimize.least_squares`` documentation for more information on non-linear least squares fitting options.

	Returns
	-------
	
	etr_max : float
		The maximum electron transport rate
	alpha : float
		The light limited slope of electron transport
	ek : float
		The photoacclimation of ETR
	rsq : float
		The r\ :sup:`2` value
	bias : float
		The bias of the fit
	chi : float
		The chi-squared goodness of the fit
	etrmax_err : float
		The fit error of ETR\ :sup:`max`
	alpha_err : float
		The fit error of α\ :sub:`ETR`
	data : [np.array, np.array]
		Optional, the final data used for the fitting procedure


	Example
	-------
	>>> etr_max, alpha, ek, rsq, bias, chi, etr_max_err, alpha_err = ppu.calculate_e_independent_etr(fvfm, sigma, par, return_data=False)
	"""
	from ._equations import __calculate_modified_Webb_model__, __calculate_rsquared__, __calculate_bias__, __calculate_chisquared__, __calculate_fit_errors__
	from numpy import mean, array, isnan, inf, repeat, nan, concatenate
	from pandas import DataFrame
	from scipy.optimize import least_squares 

	lss = light_step_size - 1
	sigma = mean(sigma[0:lss])

	phi = fvfm / mean(fvfm[0:lss])

	# Create pandas DataFrame for using groupby
	df = DataFrame([phi, par])
	df = df.T
	df.columns = ['phi', 'par']
	df[df.phi < 0] = nan
	df = df.dropna()

	# exclude outliers if more than mean ± (stdev * multiplier)
	grp = df.groupby(by='par')
	mn = grp.mean()
	std = grp.std()
	c = grp.count()
	ulim = repeat((mn.phi.values + std.phi.values * outlier_multiplier), c.phi.values)
	llim = repeat((mn.phi.values - std.phi.values * outlier_multiplier), c.phi.values)
	idx = []
	for i, items in enumerate(grp.indices.items()):
		idx.append(items[-1])

	idx = concatenate(idx, axis=0)

	# Create pandas DataFrame of upper and lower using original indexes of data
	mask = DataFrame([ulim, llim, idx]).T
	mask.columns = ['ulim','llim','index']
	mask = mask.set_index('index').sort_index()

	m = (df.phi.values > mask.ulim) | (df.phi.values < mask.llim)

	# Where condition is True, set values of value to NaN
	df.loc[m.values,'phi'] = nan

	# Create means per light step on QC data
	df = df.groupby('par').mean().reset_index()

	# Define data for fitting and estimates of ETRmax and alpha
	P = array(df.phi)
	E = array(df.par)

	p0 = [1000, 2]

	# Mask missing data
	mask = isnan(P) | (P < 0) | (E == 0)
	E = E[~mask]
	P = P[~mask]
		
	def residual(p, E, P):
		return P - __calculate_modified_Webb_model__(E, *p)
	
	bds = [-inf, inf]
	if bounds:
		bds = [etrmax_lims[0], alpha_lims[0]],[etrmax_lims[1], alpha_lims[1]]

	opts = {'method':method, 'loss':loss, 'f_scale':f_scale, 'max_nfev':max_nfev, 'xtol':xtol} 
	
	popt = least_squares(residual, p0, args=(E, P), bounds=(bds), **opts)
    
	etr_max = popt.x[0]
	alpha = popt.x[1]
	etr_max *= sigma * 6.022e-3
	alpha *= sigma * 6.022e-3
	ek = etr_max / alpha

	res = P - __calculate_modified_Webb_model__(E, *popt.x)
	rsq = __calculate_rsquared__(res, P)
	bias = __calculate_bias__(res, P)
	chi = __calculate_chisquared__(res, P)		
	perr = __calculate_fit_errors__(popt, res)
	etr_max_err = perr[0]
	alpha_err = perr[1]
	

	return etr_max, alpha, ek, rsq, bias, chi, etr_max_err, alpha_err, [E,P]

