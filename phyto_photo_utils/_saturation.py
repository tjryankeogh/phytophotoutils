#!/usr/bin/env python
"""
@package phyto_photo_utils.saturation
@file phyto_photo_utils/saturation.py
@author Thomas Ryan-Keogh
@brief module containing the processing functions for calculating the saturation of fluorescence transients
"""

def calculate_fixedpmodel(pfd, fyield, seq, datetime, blank=0, sat_len=100, skip=0, ro=0.3, bounds=True, sig_lims =[100,2200], method='trf', loss='soft_l1', f_scale=0.1, max_nfev=1000, xtol=1e-9):
    
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
	ro: float
		the fixed value of the connectivity coefficient
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
	
	from ._fitting import __fit_fixed_p_model__
	from numpy import array, unique
	from tqdm import tqdm
	from pandas import Series, concat

	pfd = array(pfd)
	fyield = array(fyield)
	seq = array(seq)
	dt = array(datetime)
	
	res = []

	opts = {'bounds':bounds, 'sig_lims':sig_lims, 'method':method,'loss':loss, 'f_scale':f_scale, 'max_nfev':max_nfev, 'xtol':xtol} 

	for s in tqdm(unique(seq)):
	    
		i = seq == s
		x = pfd[i]
		y = fyield[i]
		sat = __fit_fixed_p_model__(x[skip:sat_len], y[skip:sat_len], ro, **opts)
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

def calculate_pmodel(pfd, fyield, seq, datetime, blank=0, sat_len=100, skip=0, bounds=True, ro_lims=[0.0,1.0], sig_lims =[100,2200], method='trf', loss='soft_l1', f_scale=0.1, max_nfev=1000, xtol=1e-9):
    
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
	from ._fitting import __fit_calc_p_model__
	from numpy import array, unique
	from tqdm import tqdm
	from pandas import Series, concat
      
	pfd = array(pfd)
	fyield = array(fyield)
	seq = array(seq)
	dt = array(datetime)
	
	opts = {'bounds':bounds, 'sig_lims':sig_lims, 'ro_lims':ro_lims, 'method':method,'loss':loss, 'f_scale':f_scale, 'max_nfev':max_nfev, 'xtol':xtol} 

	res = []

	for s in tqdm(unique(seq)):
	    
	    i = seq == s
	    x = pfd[i]
	    y = fyield[i]
	    sat = __fit_calc_p_model__(x[skip:sat_len], y[skip:sat_len], **opts)
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

def calculate_nopmodel(pfd, fyield, seq, datetime, ro=None, blank=0, sat_len=100, skip=0, bounds=True, sig_lims =[100,2200], method='trf', loss='soft_l1', f_scale=0.1, max_nfev=1000, xtol=1e-9):
      
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

	from ._fitting import __fit_no_p_model__
	from numpy import array, unique 
	from tqdm import tqdm
	from pandas import Series, concat
        
	pfd = array(pfd)
	fyield = array(fyield)
	seq = array(seq)
	dt = array(datetime)
    
	opts = {'bounds':bounds, 'sig_lims':sig_lims, 'method':method,'loss':loss, 'f_scale':f_scale, 'max_nfev':max_nfev, 'xtol':xtol} 

	res = []
	for s in tqdm(unique(seq)):
        
		i = seq == s
		x = pfd[i]
		y = fyield[i]
		sat = __fit_no_p_model__(x[skip:sat_len], y[skip:sat_len], ro=ro, **opts)
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

