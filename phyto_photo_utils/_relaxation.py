#!/usr/bin/env python

from ._fitting import __fit_single_decay__, __fit_triple_decay__
from numpy import array, unique
from pandas import Series, concat
from tqdm import tqdm

def calculate_single_relaxation(fyield, seq_time, seq, datetime, blank=0, sat_len=100, rel_len=60, sat_flashlets=0, bounds=True, tau_lims=[100,50000], method='trf', loss='soft_l1', f_scale=0.1, max_nfev=1000, xtol=1e-9):
	"""

	Process the raw transient data and perform the Kolber et al. 1998 relaxation model.

	Parameters
	----------
	seq_time : np.array, dtype=float, shape=[n,] 
		The sequence time of the flashlets in μs.
	fyield : np.array, dtype=float, shape=[n,] 
		The fluorescence yield of the instrument.
	seq : np.array, dtype=int, shape=[n,] 
		The measurement number.
	datetime : np.array, dtype=datetime64, shape=[n,]
		The date & time of each measurement in the numpy datetime64 format.
	blank : np.array, dtype=float, shape=[n,]
		The blank value, must be the same length as fyield.
	sat_len : int, default=100
		The number of flashlets in the saturation sequence.
	rel_len : int, default=60
		The number of flashlets in the relaxation sequence.
	sat_flashlets : int, default=0
		The number of saturation flashlets to include at the start.
	bounds : bool, default=True
		If True, will set lower and upper limit bounds for the estimation, not suitable for methods 'lm'.
	tau_lims : [int, int], default=[100, 50000]
	 	The lower and upper limit bounds for fitting τ. 
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
	res: pandas.DataFrame, shape=[n,10]
		The results of the fitting routine with columns as below:
	fo_r : np.array, dtype=float, shape=[n,]
		The minimum fluorescence level of relaxation phase.
	fm_r : np.array, dtype=float, shape=[n,]
		The maximum fluorescence level of relaxation phase
	tau : np.array, dtype=float, shape=[n,]
		The rate of QA\ :sup:`-` reoxidation in μs.
	rsq: np.array, dtype=float, shape=[n,]
		The r-squared value of the fit.
	bias : np.array, dtype=float, shape=[n,]
		The bias of fit.
	chi : np.array, dtype=float, shape=[n,]
		The chi-squared goodness of fit.
	fo_err : np.array, dtype=float, shape=[n,]
		The fit error of Fo_relax.
	fm_err : np.array, dtype=float, shape=[n,]
		The fit error of Fm_relax.
	tau_err : np.array, dtype=float, shape=[n,]
		The fit error of τ.
	nfl : np.array, dtype=float, shape=[n,]
		The number of flashlets used for fitting.

	Example
	-------
	>>> rel = ppu.calculate_single_relaxation(fyield, seq_time, seq, datetime, blank=0, sat_len=100, rel_len=40, bounds=True, tau_lims=[100, 50000])
	"""

		
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

		rel = __fit_single_decay__(x, y, **opts)
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
	seq_time : np.array, dtype=float, shape=[n,] 
		The sequence time of the flashlets in μs.
	fyield : np.array, dtype=float, shape=[n,] 
		The fluorescence yield of the instrument.
	seq : np.array, dtype=int, shape=[n,] 
		The measurement number.
	datetime : np.array, dtype=datetime64, shape=[n,]
		The date & time of each measurement in the numpy datetime64 format.
	blank : np.array, dtype=float, shape=[n,]
		The blank value, must be the same length as fyield.
	sat_len : int, default=100
		The number of flashlets in the saturation sequence.
	rel_len : int, default=60
		The number of flashlets in the relaxation sequence.
	sat_flashlets : int, default=0
		The number of saturation flashlets to include at the start.
	bounds : bool, default=True
		If True, will set lower and upper limit bounds for the estimation, not suitable for methods 'lm'.
	tau1_lims: [int, int], default=[100, 800]
	 	The lower and upper limit bounds for fitting τ1.
	tau2_lims: [int, int], default=[800, 2000]
	 	The lower and upper limit bounds for fitting τ2.
	tau3_lims: [int, int], default=[2000, 50000]
	 	The lower and upper limit bounds for fitting τ3.
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
	res: pandas.DataFrame, shape=[n,20]
		The results of the fitting routine with columns as below:
	fo_r : np.array, dtype=float, shape=[n,]
		The minimum fluorescence level of relaxation phase.
	fm_r : np.array, dtype=float, shape=[n,]
		The maximum fluorescence level of relaxation phase
	alpha1 : np.array, dtype=float, shape=[n,]
	tau1 : np.array, dtype=float, shape=[n,]
		The rate of QA\ :sup:`-` reoxidation in μs.
	alpha2 : np.array, dtype=float, shape=[n,]
	tau2 : np.array, dtype=float, shape=[n,]
		The rate of QB\ :sup:`-` reoxidation in μs.
	alpha3 : np.array, dtype=float, shape=[n,]
	tau3 : np.array, dtype=float, shape=[n,]
		The rate of PQ reoxidation in μs.
	rsq: np.array, dtype=float, shape=[n,]
		The r-squared value of the fit.
	bias : np.array, dtype=float, shape=[n,]
		The bias of fit.
	chi : np.array, dtype=float, shape=[n,]
		The chi-squared goodness of fit.
	fo_err : np.array, dtype=float, shape=[n,]
		The fit error of Fo_relax.
	fm_err : np.array, dtype=float, shape=[n,]
		The fit error of Fm_relax.
	alpha1_err : np.array, dtype=float, shape=[n,]
		The fit error of α1.
	tau1_err : np.array, dtype=float, shape=[n,]
		The fit error of τ1.
	alpha2_err : np.array, dtype=float, shape=[n,]
		The fit error of α2.
	tau2_err : np.array, dtype=float, shape=[n,]
		The fit error of τ2.
	alpha3_err : np.array, dtype=float, shape=[n,]
		The fit error of α3.
	tau3_err : np.array, dtype=float, shape=[n,]
		The fit error of τ3.
	nfl : np.array, dtype=float, shape=[n,]
		The number of flashlets used for fitting.

	Example
	-------
	>>> rel = ppu.calculate_triple_relaxation(fyield, seq_time, seq, datetime, blank=0, sat_len=100, rel_len=60, sat_flashlets=1, bounds=True, tau1_lims=[100, 800], tau2_lims=[800, 2000], tau3_lims=[2000, 50000])
	"""

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
		res.columns = ['fo_r', 'fm_r', 'alpha1', 'tau1', 'alpha2', 'tau2', 'alpha3', 'tau3', 'rsq', 'bias', 'chi', 'for_err', 'fmr_err', 'alpha1_err', 'tau1_err', 'alpha2_err', 'tau2_err', 'alpha3_err', 'tau3_err', 'nfl']
		res['datetime'] = unique(dt)

	return res

