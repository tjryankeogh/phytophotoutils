#!/usr/bin/env python
"""
@package phyto_photo_utils.saturation
@file phyto_photo_utils/saturation.py
@author Thomas Ryan-Keogh
@brief module containing the processing functions for calculating the saturation of fluorescence transients
"""

def calc_fixedpmodel(pfd, fyield, seq, datetime, blank=10, sat_len=100, skip=1, ro=0.3, bounds=True, sig_lims =[100,2200], fit_method='trf', loss_method='soft_l1', fscale=0.1, n_iter=1000, xtol=1e-9):
    
	"""

	Process the raw transient data and perform the Kolber et al. 1998 saturation model.
	
	Note: This sets a user defined fixed value for ro (the connectivity factor).

	Parameters
	----------
	pfd: numpy.ndarray 
		the photon flux density of the instrument
	fyield: numpy.ndarray 
		the fluorescence yield of the instrument
	seq: numpy.ndarray 
		the measurement number
	datetime: numpy.ndarray (datetime64)
		the date & time of each measurement in the numpy datetime64 format
	blank: float, int or numpy.ndarray
		the blank value, if np.ndarray must be the same length as fyield
	sat_len: int
		the number of flashlets in saturation sequence
	skip: int
		the number of flashlets to skip at start
	bounds: bool
		if True, will set lower and upper limit bounds for the estimation, not suitable for methods 'lm'
	sig_lims: [int, int]
	 	the lower and upper limit bounds for fitting sigmaPSII 
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

	res: pandas.DataFrame 
		The results of the fitting routine with columns: fo, fm, sigma, rsq, bias, rms, fo_err, fm_err, sigma_err, nfl

		fo: minimum fluorescence yield
		fm: maximum fluorescence yield
		fvfm: maximum photochemical efficiency
		sigma: effective absorption cross-section of PSII in A^^2
		rsq: r^^2 value
		bias: bias of fit
		chi: chi-squared goodness of fit
		fo_err: fit error of Fo
		fm_err: fit error of Fm
		sigma_err: fit error of SigmaPSII
		nfl: number of flashlets used for fitting

	"""
	from numpy import array, mean, arange, unique, isnan, count_nonzero, sum, NaN, sqrt, diag, linalg, inf
	from scipy.optimize import least_squares
	from tqdm import tqdm
	from pandas import Series, concat
	#from sklearn.metrics import mean_squared_error
    
	# This is from Kolber et al. 1998 equation 1
	def p_model(pfd, fyield, ro):
		def sat_func(pfd, fo, fm, sig):
			c = pfd[:] * 0.
			c[0] = pfd[0] * sig
			for i in arange(1, len(pfd)):
				c[i] = c[i-1] + pfd[i] * sig * (1 - c[i-1])/(1 - ro * c[i-1])
			return fo + (fm - fo) * c * (1-ro) / (1-c*ro)
	    
		# Count number of flashlets excluding NaNs
		fyield = fyield[~isnan(fyield)]
		nfl = count_nonzero(fyield)

		# Estimates of saturation parameters
		fo = fyield[:3].mean()
		fm = fyield[-3:].mean()

		# Rule checking to see if both are positive and that fm is greater than fo
		#if fm < fo:
		#	print('Fm is not greater than Fo - unable to perform optimization.')
		#if fm < 0:
		#	print('Fm is not positive - unable to perform optimization.')
		#if fo < 0:
		#	print('Fo is not positive - unable to perform optimization.')

		fo10 = fo * 0.1
		fm10 = fm * 0.1
		sig = 1500                   
		x0 = [fo, fm, sig]

		bds = [-inf, inf]
		if bounds:
			bds = [fo-fo10, fm-fm10, sig_lims[0]],[fo+fo10, fm+fm10, sig_lims[1]]

		# See scipy.optimize.least_squares documentation for more information on non-linear least squares fitting options
		opts = {'method':fit_method,'loss':loss_method, 'f_scale':fscale, 'max_nfev':n_iter, 'xtol':xtol} 

		def residual(p, pfd, fyield):
			return fyield - sat_func(pfd, *p)

		try:
			popt = least_squares(residual, x0, bounds=(bds), args=(pfd, fyield), **opts)

			fo = popt.x[0]
			fm = popt.x[1]
			sigma = popt.x[2]

			# Calculate curve fitting statistical metrics
			res = fyield - sat_func(pfd, fo, fm, sigma)
			rsq = 1 - (sum(res**2)/sum((fyield - mean(fyield))**2))
			bias = sum((1-res)/fyield) / (len(fyield)*100)
			chi = sum(res**2 / fyield)		
			J = popt.jac
			pcov = linalg.inv(J.T.dot(J)) * mean(res**2)
			perr = sqrt(diag(pcov))
			fo_err = perr[0]
			fm_err = perr[1]
			sigma_err = perr[2]
			
			return fo, fm, sigma, rsq, bias, chi, fo_err, fm_err, sigma_err, nfl

		except Exception:
			print('Optimization was unable to be performed, parameters set to NaN.')
			pass
	                
	pfd = array(pfd)
	fyield = array(fyield)
	seq = array(seq)
	dt = array(datetime)
	
	res = []

	for s in tqdm(unique(seq)):
	    
		i = seq == s
		x = pfd[i]
		y = fyield[i]
		sat = p_model(x[skip:sat_len], y[skip:sat_len], ro)
		res.append(Series(sat))

	res = concat(res, axis=1)
	res = res.T
	
	if res.empty:
		pass
	
	else: 

		res.columns = ['fo','fm','sigma','rsq','bias','chi','fo_err','fm_err','sigma_err','nfl']
	
		# Subtract blank from Fo and Fm - blank can be single value or same length as dataframe
		res['fo'] -= blank
		res['fm'] -= blank

		# Calculate Fv/Fm after blank subtraction
		fvfm = (res.fm - res.fo)/res.fm
		res.insert(3, "fvfm", fvfm)
		
		res['datetime'] = unique(dt)
	
	return res

def calc_pmodel(pfd, fyield, seq, datetime, blank=10, sat_len=100, skip=1, bounds=True, ro_lims=[0.0,1.0], sig_lims =[100,2200], fit_method='trf', loss_method='soft_l1', fscale=0.1, n_iter=1000, xtol=1e-9):
    
	"""

	Process the raw transient data and perform the Kolber et al. 1998 saturation model, including modelling the connectivity coefficient.


	Parameters
	----------
	pfd: numpy.ndarray 
		the photon flux density of the instrument
	fyield: numpy.ndarray 
		the fluorescence yield of the instrument
	seq: numpy.ndarray 
		the measurement number
	datetime: numpy.ndarray (datetime64)
		the date & time of each measurement in the numpy datetime64 format
	blank: float, int or numpy.ndarray
		the blank value, if np.ndarray must be the same length as fyield
	sat_len: int
		the number of flashlets in saturation sequence
	skip: int
		the number of flashlets to skip at start
	bounds: bool
		if True, will set lower and upper limit bounds for the estimation, not suitable for methods 'lm'
	ro_lims: [float, float]
		the lower and upper limit bounds for fitting the connectivity coefficient
	sig_lims: [int, int]
	 	the lower and upper limit bounds for fitting sigmaPSII 
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

	res: pandas.DataFrame 
		The results of the fitting routine with columns: fo, fm, sigma, rsq, bias, rms, fo_err, fm_err, sigma_err, nfl

		fo: minimum fluorescence yield
		fm: maximum fluorescence yield
		fvfm: maximum photochemical efficiency
		sigma: effective absorption cross-section of PSII in A^^2
		rsq: r^^2 value
		bias: bias of fit
		chi: chi-squared goodness of fit
		fo_err: fit error of Fo
		fm_err: fit error of Fm
		sigma_err: fit error of SigmaPSII
		nfl: number of flashlets used for fitting

	"""

	from numpy import array, mean, arange, unique, isnan, count_nonzero, sum, NaN, sqrt, diag, linalg, inf
	from scipy.optimize import least_squares
	from tqdm import tqdm
	from pandas import Series, concat
	#from sklearn.metrics import mean_squared_error

	# This is from Kolber et al. 1998 equation 1
	def p_model(pfd, fyield):
		def sat_func(pfd, fo, fm, sig, ro):
			c = pfd[:] * 0.
			c[0] = pfd[0] * sig
			for i in arange(1, len(pfd)):
				c[i] = c[i-1] + pfd[i] * sig * (1 - c[i-1])/(1 - ro * c[i-1])
			return fo + (fm - fo) * c * (1-ro) / (1-c*ro)
	    
	    # Count number of flashlets excluding NaNs
		fyield = fyield[~isnan(fyield)]
		nfl = count_nonzero(fyield)
	    
	    # Estimates of saturation parameters
		fo = fyield[:3].mean()
		fm = fyield[-3:].mean()

		# Rule checking to see if both are positive and that fm is greater than fo
		#if fm < fo:
		#	raise UserWarning('Fm is not greater than Fo - unable to perform optimization.')
		#if fm < 0:
		#	raise UserWarning('Fm is not positive - unable to perform optimization.')
		#if fo < 0:
		#	raise UserWarning('Fo is not positive - unable to perform optimization.')

		fo10 = fo * 0.1
		fm10 = fm * 0.1
		sig = 1500
		ro = 0.3
		x0 = [fo, fm, sig, ro]

		bds = [-inf, inf]
		if bounds:
			bds = [fo-fo10, fm-fm10, sig_lims[0], ro_lims[0]],[fo+fo10, fm+fm10, sig_lims[1], ro_lims[1]]
	    
		# See scipy.optimize.least_squares documentation for more information on non-linear least squares fitting options
		opts = {'method':fit_method,'loss':loss_method, 'f_scale':fscale, 'max_nfev':n_iter, 'xtol':xtol} 
	    
		def residual(p, pfd, fyield):
			return fyield - sat_func(pfd, *p)
	    
		try:
			popt = least_squares(residual, x0, bounds=(bds), args=(pfd, fyield), **opts)

			fo = popt.x[0]
			fm = popt.x[1]
			sigma = popt.x[2]
			ro = popt.x[3]

			# Calculate curve fitting statistical metrics
			res = fyield - sat_func(pfd, fo, fm, sigma, ro)
			rsq = 1 - (sum(res**2)/sum((fyield - mean(fyield))**2))
			bias = sum((1-res)/fyield) / (len(fyield)*100)
			chi = sum(res**2 / fyield)	
			J = popt.jac
			pcov = linalg.inv(J.T.dot(J)) * mean(res**2)
			perr = sqrt(diag(pcov))
			fo_err = perr[0]
			fm_err = perr[1]
			sigma_err = perr[2]
			ro_err = perr[3]

			return fo, fm, sigma, ro, rsq, bias, chi, fo_err, fm_err, sigma_err, ro_err, nfl
		
		except Exception:
			print('Optimization was unable to be performed, parameters set to NaN.')
			pass
      
	pfd = array(pfd)
	fyield = array(fyield)
	seq = array(seq)
	dt = array(datetime)
	
	res = []

	for s in tqdm(unique(seq)):
	    
	    i = seq == s
	    x = pfd[i]
	    y = fyield[i]
	    sat = p_model(x[skip:sat_len], y[skip:sat_len])
	    res.append(Series(sat))

	res = concat(res, axis=1)
	res = res.T
	
	if res.empty:
		pass
	
	else: 

		res.columns = ['fo','fm','sigma','ro','rsq','bias','chi','fo_err','fm_err','sigma_err','ro_err','nfl']
	
		# Subtract blank from Fo and Fm - blank can be single value or same length as dataframe
		res['fo'] -= blank
		res['fm'] -= blank

		# Calculate Fv/Fm after blank subtraction
		fvfm = (res.fm - res.fo)/res.fm
		res.insert(3, "fvfm", fvfm)
		
		res['datetime'] = unique(dt)
	
	return res

def calc_nopmodel(pfd, fyield, seq, datetime, blank=10, sat_len=100, skip=1, bounds=True, sig_lims =[100,2200], fit_method='trf', loss_method='soft_l1', fscale=0.1, n_iter=1000, xtol=1e-9):
      
	"""
	
	Process the raw transient data and perform the no connectivity saturation model.

	Parameters
	----------
	pfd: numpy.ndarray 
		the photon flux density of the instrument
	fyield: numpy.ndarray 
		the fluorescence yield of the instrument
	seq: numpy.ndarray 
		the measurement number
	datetime: numpy.ndarray (datetime64)
		the date & time of each measurement in the numpy datetime64 format
	blank: float, int or numpy.ndarray
		the blank value, if np.ndarray must be the same length as fyield
	sat_len: int
		the number of flashlets in saturation sequence
	skip: int
		the number of flashlets to skip at start
	bounds: bool
		if True, will set lower and upper limit bounds for the estimation, not suitable for methods 'lm'
	sig_lims: [int, int]
	 	the lower and upper limit bounds for fitting sigmaPSII 
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

	res: pandas.DataFrame 
		The results of the fitting routine with columns: fo, fm, sigma, rsq, bias, rms, fo_err, fm_err, sigma_err, nfl

		fo: minimum fluorescence yield
		fm: maximum fluorescence yield
		fvfm: maximum photochemical efficiency
		sigma: effective absorption cross-section of PSII in A^^2
		rsq: r^^2 value
		bias: bias of fit
		chi: chi-squared goodness of fit
		fo_err: fit error of Fo
		fm_err: fit error of Fm
		sigma_err: fit error of SigmaPSII
		nfl: number of flashlets used for fitting

	"""

	from numpy import array, mean, arange, unique, isnan, count_nonzero, sum, NaN, sqrt, diag, linalg, exp, cumsum, inf
	from scipy.optimize import least_squares
	from tqdm import tqdm
	from pandas import Series, concat
	#from sklearn.metrics import mean_squared_error

	def nop_model(pfd, fyield):
		def sat_func(pfd, fo, fm, sig):
			return fo + (fm - fo) * (1 - exp(-sig * cumsum(pfd)))
        
		# Count number of flashlets excluding NaNs
		fyield = fyield[~isnan(fyield)]
		nfl = count_nonzero(fyield)

		# Estimates of saturation parameters
		fo = fyield[:3].mean()
		fm = fyield[-3:].mean()

		# Rule checking to see if both are positive and that fm is greater than fo
		#if fm < fo:
		#	raise UserWarning('Fm is not greater than Fo - unable to perform optimization.')
		#if fm < 0:
		#	raise UserWarning('Fm is not positive - unable to perform optimization.')
		#if fo < 0:
		#	raise UserWarning('Fo is not positive - unable to perform optimization.')

		fo10 = fo * 0.1
		fm10 = fm * 0.1
		sig = 1500                   
		x0 = [fo, fm, sig]

		# See scipy.optimize.least_squares documentation for more information on non-linear least squares fitting options
		opts = {'method':fit_method,'loss':loss_method, 'f_scale':fscale, 'max_nfev':n_iter, 'xtol':xtol} 

		def residual(p, pfd, fyield):
			return fyield - sat_func(pfd, *p)
		
		bds = [-inf, inf]
		if bounds:
			bds = [fo-fo10, fm-fm10, sig_lims[0]],[fo+fo10, fm+fm10, sig_lims[1]]

		try:
			popt = least_squares(residual, x0, bounds=(bds), args=(pfd, fyield), **opts)

			fo = popt.x[0] 
			fm = popt.x[1]
			sigma = popt.x[2]

			# Calculate curve fitting statistical metrics
			res = fyield - sat_func(pfd, fo, fm, sigma)
			rsq = 1 - (sum(res**2)/sum((fyield - mean(fyield))**2))
			bias = sum((1-res)/fyield) / (len(fyield)*100)
			chi = sum(res**2 / fyield)		
			J = popt.jac
			pcov = linalg.inv(J.T.dot(J)) * mean(res**2)
			perr = sqrt(diag(pcov))
			fo_err = perr[0]
			fm_err = perr[1]
			sigma_err = perr[2]

			return fo, fm, sigma, rsq, bias, chi, fo_err, fm_err, sigma_err, nfl
		except Exception:
			print('Optimization was unable to be performed, parameters set to NaN.')
			pass
        
	pfd = array(pfd)
	fyield = array(fyield)
	seq = array(seq)
	dt = array(datetime)
    
	res = []
	for s in tqdm(unique(seq)):
        
		i = seq == s
		x = pfd[i]
		y = fyield[i]
		sat = nop_model(x[skip:sat_len], y[skip:sat_len])
		res.append(Series(sat))
    
	res = concat(res, axis=1)
	res = res.T

	if res.empty:
		pass
	
	else: 

		res.columns = ['fo','fm','sigma','rsq','bias','chi','fo_err','fm_err','sigma_err','nfl']
	
		# Subtract blank from Fo and Fm - blank can be single value or same length as dataframe
		res['fo'] -= blank
		res['fm'] -= blank

		# Calculate Fv/Fm after blank subtraction
		fvfm = (res.fm - res.fo)/res.fm
		res.insert(3, "fvfm", fvfm)
		
		res['datetime'] = unique(dt)
	
	return res

