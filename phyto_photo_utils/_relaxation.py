#!/usr/bin/env python
"""
@package phyto_photo_utils.relaxation
@file phyto_photo_utils/relaxation.py
@author Thomas Ryan-Keogh
@brief module containing the processing functions for calculating the relaxation of fluorescence transients
"""

def calculate_single_relaxation(fyield, seq_time, seq, datetime, blank=0, sat_len=100, rel_len=60, sat_flashlets=0, bounds=True, tau_lims=[100,50000], method='trf', loss='soft_l1', f_scale=0.1, max_nfev=1000, xtol=1e-9):
	"""

	Process the raw transient data and perform the Kolber et al. 1998 relaxation model.

	Parameters
	----------
	seq_time: numpy.ndarray 
		the sequence time of the flashlets
	fyield: numpy.ndarray 
		the fluorescence yield of the instrument
	seq: numpy.ndarray 
		the measurement number
	datetime: numpy.ndarray (datetime64)
		the date & time of each measurement in the numpy datetime64 format
	blank: float, int or numpy.ndarray
		the blank value, if np.ndarray must be the same length as fyield
	sat_len: int
		the number of flashlets in the saturation sequence
	rel_len: int
		the number of flashlets in the relaxation sequence
	sat_flashlets: int
		the number of saturation flashlets to include at the start
	bounds: bool
		if True, will set lower and upper limit bounds for the estimation, not suitable for methods 'lm'
	tau_lims: [int, int]
	 	the lower and upper limit bounds for fitting tau 
	fit_method: str
		The algorithm to perform minimization. Default: 'trf'.
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
		The results of the fitting routine with columns: fo_r, fm_r, tau, rsq, bias, rms, fo_err, fm_err, tau_err, nfl

		fo_r: minimum fluorescence of relaxation phase
		fm_r: maximum fluorescence of relaxation phase
		tau: rate of QA- reoxidation (microseconds)
		rsq: r^^2 value
		bias: bias of fit
		chi: chi-squared goodness of fit
		fo_err: fit error of Fo_relax
		fm_err: fit error of Fm_relax
		tau_err: fit error of tau
		nfl: number of flashlets used for fitting

	"""
	from ._fitting import __fit_single_decay__
	from numpy import array, unique
	from pandas import Series, concat
	from tqdm import tqdm
		
	seq_time = array(seq_time)
	fyield = array(fyield)
	seq = array(seq)
	dt = array(datetime)

	opts = {'bounds':bounds, 'tau_lims':tau_lims, 'method':method,'loss':loss, 'f_scale':f_scale, 'max_nfev':max_nfev, 'xtol':xtol} 

	res = []
    
	for s in tqdm(unique(seq)):

		i = seq == s
		x = seq_time[i]
		y = fyield[i]
		x_min = min(x[sat_len:])
		x = x[sat_len-sat_flashlets:sat_len+rel_len] - x_min
		y = y[sat_len-sat_flashlets:sat_len+rel_len]

		rel = __fit_single_decay__(x, y)
		res.append(Series(rel))

	res = concat(res, axis=1)
	res = res.T

	if res.empty:
		pass
	
	else: 
		res.columns = ['fo_r','fm_r','tau','rsq','bias','chi','fo_err','fm_err','tau_err','nfl']
		res['datetime'] = unique(dt)

	return res

def calculate_triple_relaxation(fyield, seq_time, seq, datetime, blank=0, sat_len=100, rel_len=60, sat_flashlets=0, bounds=True, tau1_lims=[100, 800], tau2_lims=[800, 2000], tau3_lims=[2000, 50000], method='trf', loss='soft_l1', f_scale=0.1, max_nfev=1000, xtol=1e-9):
    
	"""
	
	Process the raw transient data and perform the Kolber et al. 1998 relaxation model.

	Parameters
	----------
	seq_time: numpy.ndarray 
		the sequence time of the flashlets
	fyield: numpy.ndarray 
		the fluorescence yield of the instrument
	seq: numpy.ndarray 
		the measurement number
	datetime: numpy.ndarray (datetime64)
		the date & time of each measurement in the numpy datetime64 format
	blank: float, int or numpy.ndarray
		the blank value, if np.ndarray must be the same length as fyield
	sat_len: int
		the number of flashlets in the saturation sequence
	rel_len: int
		the number of flashlets in the relaxation sequence
	sat_flashlets: int
		the number of saturation flashlets to include at the start
	bounds: bool
		if True, will set lower and upper limit bounds for the estimation, not suitable for methods 'lm'
	tau1_lims: [int, int]
	 	the lower and upper limit bounds for fitting tau_1
	tau2_lims: [int, int]
	 	the lower and upper limit bounds for fitting tau_2
	tau3_lims: [int, int]
	 	the lower and upper limit bounds for fitting tau_3
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
		The results of the fitting routine with columns: 
		fo_r, fm_r, alpha_1 tau_1, alpha_2 tau_2, alpha_3 tau_3, rsq, bias, rms, fo_err, fm_err, alpha1_err, tau1_err, alpha2_err, tau2_err, alpha3_err, tau3_err, nfl

		fo_r: minimum fluorescence of relaxation phase
		fm_r: maximum fluorescence of relaxation phase
		alpha_1
		tau_1:  (microseconds)
		alpha_2:
		tau_2:
		alpha_3:
		tau_3:
		rsq: r^^2 value
		bias: bias of fit
		chi: chi-squared goodness of fit
		fo_err: fit error of Fo_relax
		fm_err: fit error of Fm_relax
		alpha1_err: fit error of alpha_1
		tau1_err: fit error of tau_1
		alpha2_err: fit error of alpha_2
		tau2_err: fit error of tau_2
		alpha3_err: fit error of alpha_3
		tau3_err: fit error of tau_3
		nfl: number of flashlets used for fitting
	   
	"""

	from ._fitting import __fit_triple_decay__
	from numpy import array, unique
	from pandas import Series, concat
	from tqdm import tqdm

	seq_time = array(seq_time)
	fyield = array(fyield)
	seq = array(seq)
	dt = array(datetime)

	opts = {'bounds':bounds, 'tau1_lims':tau1_lims, 'tau2_lims':tau2_lims, 'tau3_lims':tau3_lims, 'method':method,'loss':loss, 'f_scale':f_scale, 'max_nfev':max_nfev, 'xtol':xtol} 

	res = []

	for s in tqdm(unique(seq)):

		i = seq == s
		x = seq_time[i]
		y = fyield[i]
		x_min = min(x[sat_len:])
		x = x[sat_len-sat_flashlets:sat_len+rel_len] - x_min
		y = y[sat_len-sat_flashlets:sat_len+rel_len]
	    
		rel = __fit_triple_decay__(x, y, **opts)
		
		res.append(Series(rel))

	res = concat(res, axis=1)
	res = res.T
	
	if res.empty:
		pass
	
	else:
		res.columns = ['fo_r', 'fm_r', 'alpha1', 'tau1', 'alpha2', 'tau2', 'alpha3', 'tau3', 'rsq', 'bias', 'chi','for_err', 'fmr_err', 'alpha1_err', 'tau1_err', 'alpha2_err', 'tau2_err', 'alpha3_err', 'tau3_err', 'nfl']
		res['datetime'] = unique(dt)

	return res

