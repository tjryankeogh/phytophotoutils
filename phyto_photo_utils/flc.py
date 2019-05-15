"""
@package phyto_photo_utils.flc
@file phyto_photo_utils/flc.py
@author Thomas Ryan-Keogh
@brief module containing the processing functions for fluorescence light curves

Note: See example csv file for the correct dataframe formatting
"""

def e_dependent_etr(fo, fm, fvfm, sigma, par, dark_sigma=True, light_step_size=12, bounds=True, alpha_lims=[0,4], etrmax_lims=[0,2000], fit_method='trf', loss_method='soft_l1', fscale=0.1, n_iter=1000, xtol=1e-9):
	"""

	Process the fluorescence light curve data under the e dependent model.

	The function uses the E-dependent model to calculate rETR and then fit to Webb et al. (1974) model for the determination of PE parameters, alpha and ETRmax.

	Parameters
	----------
	
	fo: numpy.ndarray
		The minimum fluorescence data.
	fm: numpy.ndarray
		The maximum fluorescence data.
	fvfm:: numpy.ndarray
		The fvfm data.
	sigma: numpy.ndarray
		The sigmaPSII data in A^2.
	par: np.ndarray 
		The actinic light levels of the fluorescence light curve.
	dark_sigma: bool
		If True, will use mean of sigmaPSII under 0 actinic light for calculation. If False, will use sigmaPSII and sigmaPSII' for calculation.
	light_step_size: int
		The number of measurements for initial light step.
	bounds: bool
		if True, will set lower and upper limit bounds for the estimation, not suitable for methods 'lm'
	alpha_lims: [int, int]
	 	the lower and upper limit bounds for fitting alpha
	etrmax_lims: [int, int]
	 	the lower and upper limit bounds for fitting ETRmax	 
	fit_method: str
		The algorithm to perform minimization. 
		See scipy.optimize.least_squares documentation for more information on non-linear least squares fitting options.
	loss_method: str
		The loss function to be used. Note: Method ‘lm’ supports only ‘linear’ loss.
		See scipy.optimize.least_squares documentation for more information on non-linear least squares fitting options.
	fscale: float
	 	The soft margin value between inlier and outlier residuals.
	 	See scipy.optimize.least_squares documentation for more information on non-linear least squares fitting options.
	n_iter: int			
		The number of iterations to perform fitting routine.
	xtol: float			
		The tolerance for termination by the change of the independent variables.
		See scipy.optimize.least_squares documentation for more information on non-linear least squares fitting options.

	Returns
	-------

	etr_max: float
		The maximum electron transport rate.
	alpha: float
		The light limited slope.
	rsq: float
		The correlation coefficient, the r^^2 value.
	bias: float
		The bias of the final fit.
	etrmax_err: float
		The fit error of ETRmax
	alpha_err: float
		The fit error of alpha
	   
	"""
	from numpy import array, isnan, exp, sum, mean, sqrt, diag, linalg, inf
	from scipy.optimize import least_squares 
	from sklearn.metrics import mean_squared_error
	from pandas import DataFrame

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

	# Create means per light step
	df = df.groupby('par').mean().reset_index()

	# Define data for fitting and estimates of ETRmax and alpha
	P = array(df.etr)
	E = array(df.par)

	p0 = [1000, 2]

	# Mask missing data
	mask = isnan(P)
	E = E[~mask]
	P = P[~mask]
	
	# Webb et al. (1974) model
	def PI_fit(E, P, a):
			model = P * (1 - exp(-a * E/P))
			return model
	
	def residual(p, E, P):
		return P - PI_fit(E, *p)
	
	bds = [-inf, inf]
	if bounds:
			bds = [etrmax_lims[0], alpha_lims[0]],[etrmax_lims[1], alpha_lims[1]]

	# See scipy.optimize.least_squares documentation for more information on non-linear least squares fitting options
	opts = {'method':fit_method,'loss':loss_method, 'f_scale':fscale, 'max_nfev':n_iter, 'xtol':xtol} 

	try:
		popt = least_squares(residual, p0, args=(E, P), bounds=(bds), **opts)
	    
		etr_max = popt.x[0]
		alpha = popt.x[1]
		ek = etr_max / alpha

		# Calculate curve fitting statistical metrics
		res = P - PI_fit(E, popt.x[0], popt.x[1])
		rsq = 1 - (sum(res**2)/sum((P - mean(P))**2))
		bias = sum((1-res)/P) / (len(P)*100)
		rms = sqrt(mean_squared_error(P, PI_fit(E, etr_max, alpha)))		
		J = popt.jac
		pcov = linalg.inv(J.T.dot(J)) * mean(res**2)
		perr = sqrt(diag(pcov))
		etr_max_err = perr[0]
		alpha_err = perr[1]

		return etr_max, alpha, ek, rsq, bias, etr_max_err, alpha_err

	except Exception:
		print('Optimization was unable to be performed, parameters set to NaN.')
		pass    


def e_independent_etr(fvfm, sigma, par, light_step_size=12, PE_model='Webb', bounds=True, alpha_lims=[0,4], etrmax_lims=[0,2000], fit_method='trf', loss_method='soft_l1', fscale=0.1, n_iter=1000, xtol=1e-9):
	"""
	   
	INFORMATION
	-----------

	Process the fluorescence light curve data using the e independent model (see Silsbe & Kromkamp 2012)

	The function uses the E-independent model to calculate phi and then fit to Webb et al. (1974) model for the determination of PE parameters, alpha and ETRmax.

	Parameters
	----------

	fvfm:: numpy.ndarray
		The fvfm data.
	sigma: numpy.ndarray
		The sigmaPSII data in A^2.
	par: np.ndarray 
		The actinic light levels of the fluorescence light curve.
	dark_sigma: bool
		If True, will use mean of sigmaPSII under 0 actinic light for calculation. If False, will use sigmaPSII and sigmaPSII' for calculation.
	light_step_size: int
		The number of measurements for initial light step.
	PE_model: str
		The photosynthesis-irradiance model to use for the fitting routine. Options inlude Webb (see Webb et al., 1974) or Platt (see Platt et al., 1980).
	bounds: bool
		if True, will set lower and upper limit bounds for the estimation, not suitable for methods 'lm'
	alpha_lims: [int, int]
	 	the lower and upper limit bounds for fitting alpha
	etrmax_lims: [int, int]
	 	the lower and upper limit bounds for fitting ETRmax	 
	fit_method: str
		The algorithm to perform minimization. 
		See scipy.optimize.least_squares documentation for more information on non-linear least squares fitting options.
	loss_method: str
		The loss function to be used. Note: Method ‘lm’ supports only ‘linear’ loss.
		See scipy.optimize.least_squares documentation for more information on non-linear least squares fitting options.
	fscale: float
	 	The soft margin value between inlier and outlier residuals.
	 	See scipy.optimize.least_squares documentation for more information on non-linear least squares fitting options.
	n_iter: int			
		The number of iterations to perform fitting routine.
	xtol: float			
		The tolerance for termination by the change of the independent variables.
		See scipy.optimize.least_squares documentation for more information on non-linear least squares fitting options.

	Returns
	-------

	etr_max: float
		The maximum electron transport rate.
	alpha: float
		The light limited slope.
	rsq: float
		The correlation coefficient, the r^^2 value.
	bias: float
		The bias of the final fit.
	etrmax_err: float
		The fit error of ETRmax
	alpha_err: float
		The fit error of alpha
	   
	"""
	from numpy import array, isnan, exp, sum, mean, sqrt, diag, inf, linalg
	from pandas import DataFrame
	from scipy.optimize import least_squares 
	from sklearn.metrics import mean_squared_error

	lss = light_step_size - 1 # Python starts at 0
	sigma = mean(sigma[0:lss])

	phi = fvfm / mean(fvfm[0:lss])

	# Create pandas DataFrame for using groupby
	df = DataFrame([phi, par])
	df = df.T
	df.columns = ['phi', 'par']

	# Create means per light step
	df = df.groupby('par').mean().reset_index()

	# Define data for fitting and estimates of ETRmax and alpha
	P = array(df.phi)
	E = array(df.par)

	p0 = [1000, 2]

	# Mask missing data
	mask = isnan(P) | (P < 0) | (E == 0)
	E = E[~mask]
	P = P[~mask]
	
	# Modified Webb et al. (1974) model - see Silsbe & Kromkamp (2012)
	def PI_fit(E, P, a):
		return P * (1 - exp(-a * E/P)) * (E**-1)
	
	def residual(p, E, P):
		return P - PI_fit(E, *p)
	
	bds = [-inf, inf]
	if bounds:
		#if PE_model == 'Platt':
		#	bds = [etrmax_lims[0], alpha_lims[0], 0],[etrmax_lims[1], alpha_lims[1], 1]
		#else:
			bds = [etrmax_lims[0], alpha_lims[0]],[etrmax_lims[1], alpha_lims[1]]

	# See scipy.optimize.least_squares documentation for more information on non-linear least squares fitting options
	opts = {'method':fit_method,'loss':loss_method, 'f_scale':fscale, 'max_nfev':n_iter, 'xtol':xtol} 
	
	try:
		popt = least_squares(residual, p0, args=(E, P), bounds=(bds), **opts)
	    
		etr_max = popt.x[0]
		alpha = popt.x[1]
		etr_max *= sigma * 6.022e-3
		alpha *= sigma * 6.022e-3
		ek = etr_max / alpha
		# Calculate curve fitting statistical metrics
		res = P - PI_fit(E, popt.x[0], popt.x[1])
		rsq = (1 - (sum(res**2)/sum((P - mean(P))**2)))
		bias = sum((1-res)/P) / (len(P)*100)
		rms = sqrt(mean_squared_error(P, PI_fit(E, etr_max, alpha)))		
		J = popt.jac
		pcov = linalg.inv(J.T.dot(J)) * mean(res**2)
		perr = sqrt(diag(pcov))
		etr_max_err = perr[0]
		alpha_err = perr[1]
		

		return etr_max, alpha, ek, rsq, bias, etr_max_err, alpha_err

	except Exception:
		#print('Optimization was unable to be performed, parameters set to NaN.')
		pass
