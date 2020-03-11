#!/usr/bin/env python
	
from ._fitting import __fit_fixed_p_model__, __fit_calc_p_model__, __fit_no_p_model__
from pandas import Series, concat
from numpy import array, unique
from tqdm import tqdm


def fit_saturation(pfd, flevel, seq, datetime, blank=0, sat_len=100, skip=0, ro=0.3, no_ro=False, fixed_ro=False, bounds=True, sig_lims =[100, 2200], ro_lims=[0.0, 1.0], datetime_unique=False, method='trf', loss='soft_l1', f_scale=0.1, max_nfev=None, xtol=1e-9):
    
	"""
	Process the raw transient data and perform the Kolber et al. 1998 saturation model.


	Parameters
	----------
	pfd : np.array, dtype=float, shape=[n,] 
		The photon flux density of the instrument in μmol photons m\ :sup:`2` s\ :sup:`-1`.
	flevel : np.array, dtype=float, shape=[n,] 
		The fluorescence yield of the instrument.
	seq : np.array, dtype=int, shape=[n,] 
		The measurement number.
	datetime : np.array, dtype=datetime64, shape=[n,]
		The date & time of each measurement.
	blank : np.array, dype=float, shape=[n,]
		The blank value, must be the same length as flevel.
	sat_len : int, default=100
		The number of flashlets in saturation sequence.
	skip : int, default=0
		the number of flashlets to skip at start.
	ro : float, default=0.3
		The fixed value of the connectivity coefficient. Not required if fixed_ro is False.
	no_ro : bool, default=False
		If True, this processes the raw transient data and performs the no connectivity saturation model.
	fixed_ro : bool, default=False
		If True, this sets a user defined fixed value for ro (the connectivity factor) when fitting the saturation model.
	bounds : bool, default=True
		If True, will set lower and upper limit bounds for the estimation, not suitable for methods 'lm'.
	sig_lims : [int, int], default=[100, 2200]
	 	The lower and upper limit bounds for fitting sigmaPSII.
	ro_lims: [float, float], default=[0.0, 0.1]
		The lower and upper limit bounds for fitting the connectivity coefficient. Not required if no_ro and fixed_ro are False.
	method : str, default='trf'
		The algorithm to perform minimization. See ``scipy.optimize.least_squares`` documentation for more information on non-linear least squares fitting options.
	loss : str, default='soft_l1'
		The loss function to be used. Note: Method ‘lm’ supports only ‘linear’ loss. See ``scipy.optimize.least_squares`` documentation for more information on non-linear least squares fitting options.
	f_scale : float, default=0.1
	 	The soft margin value between inlier and outlier residuals. See ``scipy.optimize.least_squares`` documentation for more information on non-linear least squares fitting options.
	max_nfev : int, default=None		
		The number of iterations to perform fitting routine. If None, the value is chosen automatically. See ``scipy.optimize.least_squares`` documentation for more information on non-linear least squares fitting options.
	xtol : float, default=1e-9			
		The tolerance for termination by the change of the independent variables. See ``scipy.optimize.least_squares`` documentation for more information on non-linear least squares fitting options.

	Returns
	-------

	res: pandas.DataFrame
		The results of the fitting routine with columns as below:
	fo : np.array, dtype=float, shape=[n,]
		The minimum fluorescence level.
	fm : np.array, dtype=float, shape=[n,]
		The maximum fluorescence level.
	sigma : np.array, dtype=float, shape=[n,]
		The effective absorption cross-section of PSII in Å\ :sup:`2`.
	fvfm : np.array, dtype=float, shape=[n,]
		The maximum photochemical efficiency.
	ro : np.array, dtype=float, shape=[n,]
		The connectivity coefficient, ρ, only returned if no_ro and fixed_ro are False.
	bias : np.array, dtype=float, shape=[n,]
		The bias of fit.
	rmse : np.array, dtype=float, shape=[n,]
		The root mean squared error of the fit.
	fo_err : np.array, dtype=float, shape=[n,]
		The fit error of F\ :sub:`o`.
	fm_err : np.array, dtype=float, shape=[n,]
		The fit error of F\ :sub:`m`.
	sigma_err : np.array, dtype=float, shape=[n,]
		The fit error of σ\ :sub:`PSII`.
	ro_err : np.array, dtype=float, shape=[n,]
		The fit error of ρ, only returned if no_ro and fixed_ro are False.
	nfl : np.array, dtype=int, shape=[n,]
		The number of flashlets used for fitting.
	niters : np.array, dype=int, shape=[n,]
		The number of functional evaluations done on the fitting routine.
	flag : np.array, dtype=int, shape=[n,]
		The code associated with the fitting routine success, positive values = SUCCESS, negative values = FAILURE.
		-3 : Unable to calculate parameter errors
		-2 : F\ :sub:`o` > F\ :sub:`m`
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
	>>> sat = ppu.calculate_saturation(pfd, flevel, seq, datetime, blank=0, sat_len=100, skip=0, ro=0.3, no_ro=False, fixed_ro=True, sig_lims =[100,2200])
	"""

	pfd = array(pfd)
	flevel = array(flevel)
	seq = array(seq)
	dt = array(datetime)
	
	res = []
	
	if no_ro:
		opts = {'bounds':bounds, 'sig_lims':sig_lims, 'method':method,'loss':loss, 'f_scale':f_scale, 'max_nfev':max_nfev, 'xtol':xtol} 
	elif fixed_ro:
		opts = {'bounds':bounds, 'sig_lims':sig_lims, 'method':method,'loss':loss, 'f_scale':f_scale, 'max_nfev':max_nfev, 'xtol':xtol} 
	else:
		opts = {'bounds':bounds, 'sig_lims':sig_lims, 'ro_lims':ro_lims, 'method':method,'loss':loss, 'f_scale':f_scale, 'max_nfev':max_nfev, 'xtol':xtol} 
	
	for s in tqdm(unique(seq)):
	    
		i = seq == s
		x = pfd[i]
		y = flevel[i]
		if no_ro:
			sat = __fit_no_p_model__(x[skip:sat_len], y[skip:sat_len], **opts)
		elif fixed_ro:
			sat = __fit_fixed_p_model__(x[skip:sat_len], y[skip:sat_len], ro, **opts)
		else:
			sat = __fit_calc_p_model__(x[skip:sat_len], y[skip:sat_len], **opts)

		res.append(Series(sat))

	res = concat(res, axis=1)
	res = res.T

	if res.empty:
		pass
	
	else: 
		if no_ro:
			res.columns = ['fo','fm','sigma','bias','rmse','fo_err','fm_err','sigma_err','nfl','niters','flag','success']
		if fixed_ro:
			res.columns = ['fo','fm','sigma','ro','bias','rmse','fo_err','fm_err','sigma_err','nfl','niters','flag','success']
		else:
			res.columns = ['fo','fm','sigma','ro','bias','rmse','fo_err','fm_err','sigma_err','ro_err','nfl','niters','flag','success']
	
		# Subtract blank from Fo and Fm 
		res['fo'] -= blank
		res['fm'] -= blank

		# Calculate Fv/Fm after blank subtraction
		fvfm = (res.fm - res.fo)/res.fm
		res.insert(3, "fvfm", fvfm)
		if datetime_unique:
			res['datetime'] = unique(dt)
		else:
			res['datetime'] = dt[0]
	
	return res