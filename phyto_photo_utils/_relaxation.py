#!/usr/bin/env python

from ._fitting import __fit_single_decay__, __fit_triple_decay__
from numpy import array, unique
from pandas import Series, concat
from tqdm import tqdm

def fit_relaxation(flevel, seq_time, seq, datetime, blank=0, sat_len=100, rel_len=60, sat_flashlets=None, single_decay=False, bounds=True, single_lims=[100,50000], tau1_lims=[100, 800], tau2_lims=[800, 2000], tau3_lims=[2000, 50000], method='trf', loss='soft_l1', f_scale=0.1, max_nfev=None, xtol=1e-9):
	"""

	Process the raw transient data and perform the Kolber et al. 1998 relaxation model.

	Parameters
	----------
	seq_time : np.array, dtype=float, shape=[n,] 
		The sequence time of the flashlets in μs.
	flevel : np.array, dtype=float, shape=[n,] 
		The fluorescence yield of the instrument.
	seq : np.array, dtype=int, shape=[n,] 
		The measurement number.
	datetime : np.array, dtype=datetime64, shape=[n,]
		The date & time of each measurement in the numpy datetime64 format.
	blank : np.array, dtype=float, shape=[n,]
		The blank value, must be the same length as flevel.
	sat_len : int, default=100
		The number of flashlets in the saturation sequence.
	rel_len : int, default=60
		The number of flashlets in the relaxation sequence.
	sat_flashlets : int, default=0
		The number of saturation flashlets to include at the start.
	single_decay : bool, default=False
		If True, will fit a single decay relaxation.
	bounds : bool, default=True
		If True, will set lower and upper limit bounds for the estimation, not suitable for methods 'lm'.
	single_lims : [int, int], default=[100, 50000]
	 	The lower and upper limit bounds for fitting τ, only required if single_decay is True. 
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
	max_nfev : int, default=None		
		The number of iterations to perform fitting routine. If None, the value is chosen automatically. See ``scipy.optimize.least_squares`` documentation for more information on non-linear least squares fitting options.
	xtol : float, default=1e-9			
		The tolerance for termination by the change of the independent variables. See ``scipy.optimize.least_squares`` documentation for more information on non-linear least squares fitting options.

	Returns
	-------
	res: pandas.DataFrame
		The results of the fitting routine with columns as below:
	fo_r : np.array, dtype=float, shape=[n,]
		The minimum fluorescence level of relaxation phase.
	fm_r : np.array, dtype=float, shape=[n,]
		The maximum fluorescence level of relaxation phase
	tau : np.array, dtype=float, shape=[n,]
		The rate of QA\ :sup:`-` reoxidation in μs, only returned if single_decay is True.
	alpha1 : np.array, dtype=float, shape=[n,]
		The decay coefficient of τ\ :sub:`1`, only returned if single_decay is False.
	tau1 : np.array, dtype=float, shape=[n,]
		The rate of QA\ :sup:`-` reoxidation in μs, only returned if single_decay is False.
	alpha2 : np.array, dtype=float, shape=[n,]
		The decay coefficient of τ\ :sub:`2`.
	tau2 : np.array, dtype=float, shape=[n,]
		The rate of QB\ :sup:`-` reoxidation in μs, only returned if single_decay is False.
	alpha3 : np.array, dtype=float, shape=[n,]
		The decay coefficient of τ\ :sub:`3`, only returned if single_decay is False.
	tau3 : np.array, dtype=float, shape=[n,]
		The rate of PQ reoxidation in μs, only returned if single_decay is False.
	bias : np.array, dtype=float, shape=[n,]
		The bias of fit.
	rmse : np.array, dtype=float, shape=[n,]
		The root mean squared error of the fit.
	fo_err : np.array, dtype=float, shape=[n,]
		The fit error of Fo_relax.
	fm_err : np.array, dtype=float, shape=[n,]
		The fit error of Fm_relax.
	tau_err : np.array, dtype=float, shape=[n,]
		The fit error of τ, only returned if single_decay is True.
	alpha1_err : np.array, dtype=float, shape=[n,]
		The fit error of α\ :sub:`1`, only returned if single_decay is False.
	tau1_err : np.array, dtype=float, shape=[n,]
		The fit error of τ\ :sub:`1`, only returned if single_decay is False.
	alpha2_err : np.array, dtype=float, shape=[n,]
		The fit error of α\ :sub:`2`, only returned if single_decay is False.
	tau2_err : np.array, dtype=float, shape=[n,]
		The fit error of τ\ :sub:`2`, only returned if single_decay is False.
	alpha3_err : np.array, dtype=float, shape=[n,]
		The fit error of α\ :sub:`3`, only returned if single_decay is False.
	tau3_err : np.array, dtype=float, shape=[n,]
		The fit error of τ\ :sub:`3`, only returned if single_decay is False.
	nfl : np.array, dtype=int, shape=[n,]
		The number of flashlets used for fitting.
	niters : np.array, dype=int, shape=[n,]
		The number of functional evaluations done on the fitting routine.
	flag : np.array, dtype=int, shape=[n,]
		The code associated with the fitting routine success, positive values = SUCCESS, negative values = FAILURE.
		-3 : Unable to calculate parameter errors
		-2 : F\ :sub:`o` Relax > F\ :sub:`m` Relax
		-1 : improper input parameters status returned from MINPACK.
		0 : the maximum number of function evaluations is exceeded.
		1 : gtol termination condition is satisfied.
		2 : ftol termination condition is satisfied.
		3 : xtol termination condition is satisfied.
		4 : Both ftol and xtol termination conditions are satisfied.
	success : np.array, dtype=bool, shape=[n,]
		A boolean array reporting whether fit was successful (TRUE) or if not successful (FALSE)
	datetime : np.array, dtype=datetime64, shape=[n,]
		The date and time associated with the measurement.

	Example
	-------
	>>> rel = ppu.calculate_relaxation(flevel, seq_time, seq, datetime, blank=0, sat_len=100, rel_len=40, single_decay=True, bounds=True, tau_lims=[100, 50000])
	"""

		
	seq_time = array(seq_time)
	flevel = array(flevel)
	seq = array(seq)
	dt = array(datetime)
	if single_decay:
		opts = {'sat_flashlets':sat_flashlets, 'bounds':bounds, 'single_lims':single_lims, 'method':method,'loss':loss, 'f_scale':f_scale, 'max_nfev':max_nfev, 'xtol':xtol} 
	else:
		opts = {'sat_flashlets':sat_flashlets, 'bounds':bounds, 'tau1_lims':tau1_lims, 'tau2_lims':tau2_lims, 'tau3_lims':tau3_lims, 'method':method,'loss':loss, 'f_scale':f_scale, 'max_nfev':max_nfev, 'xtol':xtol} 
	res = []
    
	for s in tqdm(unique(seq)):

		i = seq == s
		x = seq_time[i]
		y = flevel[i]
		x_min = min(x[sat_len:])
		x = x[sat_len-sat_flashlets:sat_len+rel_len] - x_min
		y = y[sat_len-sat_flashlets:sat_len+rel_len]
		if single_decay:
			rel = __fit_single_decay__(x, y, **opts)
		else:
			rel = __fit_triple_decay__(x, y, **opts)
		res.append(Series(rel))

	res = concat(res, axis=1)
	res = res.T

	if res.empty:
		pass
	
	else:
		if single_decay: 
			res.columns = ['fo_r','fm_r','tau','bias','rmse','fo_err','fm_err','tau_err','nfl','niters','flag','success']
		else:
			res.columns = ['fo_r','fm_r','alpha1','tau1','alpha2','tau2','alpha3','tau3','bias','rsme','for_err','fmr_err','alpha1_err','tau1_err','alpha2_err','tau2_err','alpha3_err','tau3_err','nfl','niters','flag','success']
		res['datetime'] = unique(dt)

	return res



