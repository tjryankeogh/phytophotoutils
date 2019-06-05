#!/usr/bin/env python
	
from ._fitting import __fit_fixed_p_model__, __fit_calc_p_model__, __fit_no_p_model__
from numpy import array, unique
from tqdm import tqdm
from pandas import Series, concat

def calculate_saturation_with_fixedpmodel(pfd, fyield, seq, datetime, blank=0, sat_len=100, skip=0, ro=0.3, bounds=True, sig_lims =[100,2200], method='trf', loss='soft_l1', f_scale=0.1, max_nfev=1000, xtol=1e-9):
    
	"""
	Process the raw transient data and perform the Kolber et al. 1998 saturation model.
	
	Note: This sets a user defined fixed value for ro (the connectivity factor).

	Parameters
	----------
	pfd : np.array, dtype=float, shape=[n,] 
		The photon flux density of the instrument in μmol photons m\ :sup:`2` s\ :sup:`-1`.
	fyield : np.array, dtype=float, shape=[n,] 
		The fluorescence yield of the instrument.
	seq : np.array, dtype=int, shape=[n,] 
		The measurement number.
	datetime : np.array, dtype=datetime64, shape=[n,]
		The date & time of each measurement.
	blank : np.array, dype=float, shape=[n,]
		The blank value, must be the same length as fyield.
	sat_len : int, default=100
		The number of flashlets in saturation sequence.
	skip : int, default=0
		the number of flashlets to skip at start.
	ro : float, default=0.3
		The fixed value of the connectivity coefficient.
	bounds : bool, default=True
		If True, will set lower and upper limit bounds for the estimation, not suitable for methods 'lm'.
	sig_lims : [int, int], default=[100,2200]
	 	The lower and upper limit bounds for fitting sigmaPSII.
	method : str, default='trf'
		The algorithm to perform minimization. See ``scipy.optimize.least_squares`` documentation for more information on non-linear least squares fitting options.
	loss : str, default='soft_l1'
		The loss function to be used. Note: Method ‘lm’ supports only ‘linear’ loss. See ``scipy.optimize.least_squares`` documentation for more information on non-linear least squares fitting options.
	f_scale : float, default=0.1
	 	The soft margin value between inlier and outlier residuals. See ``scipy.optimize.least_squares`` documentation for more information on non-linear least squares fitting options.
	max_nfev : int, default=1000			
		The number of iterations to perform fitting routine.
	xtol : float, default=1e-9			
		The tolerance for termination by the change of the independent variables. See ``scipy.optimize.least_squares`` documentation for more information on non-linear least squares fitting options.

	Returns
	-------

	res : pandas.DataFrame 
		The results of the fitting routine with columns as below:
	fo : np.array, dtype=float, shape=[n,]
		The minimum fluorescence level.
	fm : np.array, dtype=float, shape=[n,]
		The maximum fluorescence level.
	fvfm : np.array, dtype=float, shape=[n,]
		The maximum photochemical efficiency.
	sigma : np.array, dtype=float, shape=[n,]
		The effective absorption cross-section of PSII in Å\ :sup:`2`.
	rsq : np.array, dtype=float, shape=[n,]
		The r\ :sup:`2` value.
	bias : np.array, dtype=float, shape=[n,]
		The bias of fit.
	chi : np.array, dtype=float, shape=[n,]
		The chi-squared goodness of fit.
	fo_err : np.array, dtype=float, shape=[n,]
		The fit error of F\ :sub:`o`.
	fm_err : np.array, dtype=float, shape=[n,]
		The fit error of F\ :sub:`m`.
	sigma_err : np.array, dtype=float, shape=[n,]
		The fit error of σ\ :sub:`PSII`.
	nfl : np.array, dtype=int, shape=[n,]
		The number of flashlets used for fitting.

	Example
	-------
	>>> sat = ppu.calculate_saturation_with_fixedpmodel(pfd, fyield, seq, datetime, blank=0, sat_len=100, skip=0, ro=0.3, sig_lims =[100,2200])
	"""

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

def calculate_saturation_with_pmodel(pfd, fyield, seq, datetime, blank=0, sat_len=100, skip=0, bounds=True, ro_lims=[0.0,1.0], sig_lims =[100,2200], method='trf', loss='soft_l1', f_scale=0.1, max_nfev=1000, xtol=1e-9):
    
	"""
	Process the raw transient data and perform the Kolber et al. 1998 saturation model, including modelling the connectivity coefficient.

	Parameters
	----------
	pfd : np.array, dtype=float, shape=[n,] 
		The photon flux density of the instrument in μmol photons m\ :sup:`2` s\ :sup:`-1`.
	fyield : np.array, dtype=float, shape=[n,] 
		The fluorescence yield of the instrument.
	seq : np.array, dtype=int, shape=[n,] 
		The measurement number.
	datetime : np.array, dtype=datetime64, shape=[n,]
		The date & time of each measurement.
	blank : np.array, dype=float, shape=[n,]
		The blank value, must be the same length as fyield.
	sat_len : int, default=100
		The number of flashlets in saturation sequence.
	skip : int, default=0
		The number of flashlets to skip at start.
	bounds: bool
		If True, will set lower and upper limit bounds for the estimation, not suitable for methods 'lm'.
	ro_lims: [float, float]
		The lower and upper limit bounds for fitting the connectivity coefficient.
	sig_lims: [int, int]
	 	The lower and upper limit bounds for fitting sigmaPSII.
	fit_method : str, default='trf'
		The algorithm to perform minimization. See ``scipy.optimize.least_squares`` documentation for more information on non-linear least squares fitting options.
	loss_method : str, default='soft_l1'
		The loss function to be used. Note: Method ‘lm’ supports only ‘linear’ loss. See ``scipy.optimize.least_squares`` documentation for more information on non-linear least squares fitting options.
	fscale : float, default=0.1
	 	The soft margin value between inlier and outlier residuals. See ``scipy.optimize.least_squares`` documentation for more information on non-linear least squares fitting options.
	max_nfev : int, default=1000			
		The number of iterations to perform fitting routine.
	xtol : float, default=1e-9			
		The tolerance for termination by the change of the independent variables. See ``scipy.optimize.least_squares`` documentation for more information on non-linear least squares fitting options.

	Returns
	-------

	res : pandas.DataFrame 
		The results of the fitting routine with columns as below:
	fo : np.array, dtype=float, shape=[n,]
		The minimum fluorescence level.
	fm : np.array, dtype=float, shape=[n,]
		The maximum fluorescence level.
	fvfm : np.array, dtype=float, shape=[n,]
		The maximum photochemical efficiency.
	sigma : np.array, dtype=float, shape=[n,]
		The effective absorption cross-section of PSII in Å\ :sup:`2`.
	rsq : np.array, dtype=float, shape=[n,]
		The r\ :sup:`2` value.
	bias : np.array, dtype=float, shape=[n,]
		The bias of the fit.
	chi : np.array, dtype=float, shape=[n,]
		The chi-squared goodness of the fit.
	fo_err : np.array, dtype=float, shape=[n,]
		The fit error of F\ :sub:`o`.
	fm_err : np.array, dtype=float, shape=[n,]
		The fit error of F\ :sub:`m`.
	sigma_err : np.array, dtype=float, shape=[n,]
		The fit error of σ\ :sub:`PSII`.
	ro_err : np.array, dtype=float, shape=[n,]
		The fit error of ρ.
	nfl : np.array, dtype=int, shape=[n,]
		The number of flashlets used for fitting.

	Example
	-------
	>>> sat = ppu.calculate_saturation_with_pmodel(pfd, fyield, seq, datetime, blank=0, sat_len=100, skip=0, ro_lims=[0.0,1.0], sig_lims =[100,2200])
	"""

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

def calculate_saturation_with_nopmodel(pfd, fyield, seq, datetime, ro=None, blank=0, sat_len=100, skip=0, bounds=True, sig_lims =[100,2200], method='trf', loss='soft_l1', f_scale=0.1, max_nfev=1000, xtol=1e-9):
      
	"""
	
	Process the raw transient data and perform the no connectivity saturation model.

	Parameters
	----------
	pfd : np.array, dtype=float, shape=[n,] 
		The photon flux density of the instrument in μmol photons m\ :sup:`2` s\ :sup:`-1`.
	fyield : np.array, dtype=float, shape=[n,] 
		The fluorescence yield of the instrument.
	seq : np.array, dtype=int, shape=[n,] 
		The measurement number.
	datetime : np.array, dtype=datetime64, shape=[n,]
		The date & time of each measurement in the numpy datetime64 format.
	blank : np.array, dype=float, shape=[n,]
		The blank value, must be the same length as fyield.
	sat_len : int, default=100
		The number of flashlets in saturation sequence.
	skip : int, default=0
		the number of flashlets to skip at start.
	bounds: bool
		If True, will set lower and upper limit bounds for the estimation, not suitable for methods 'lm'.
	sig_lims: [int, int]
	 	The lower and upper limit bounds for fitting sigmaPSII. 
	fit_method : str, default='trf'
		The algorithm to perform minimization. See ``scipy.optimize.least_squares`` documentation for more information on non-linear least squares fitting options.
	loss_method : str, default='soft_l1'
		The loss function to be used. Note: Method ‘lm’ supports only ‘linear’ loss. See ``scipy.optimize.least_squares`` documentation for more information on non-linear least squares fitting options.
	fscale : float, default=0.1
	 	The soft margin value between inlier and outlier residuals. See ``scipy.optimize.least_squares`` documentation for more information on non-linear least squares fitting options.
	max_nfev : int, default=1000			
		The number of iterations to perform fitting routine.
	xtol : float, default=1e-9			
		The tolerance for termination by the change of the independent variables. See ``scipy.optimize.least_squares`` documentation for more information on non-linear least squares fitting options.

	Returns
	-------
	res : pandas.DataFrame, shape=[n,11] 
		The results of the fitting routine with as below:
	fo : np.array, dtype=float, shape=[n,]
		The minimum fluorescence level.
	fm : np.array, dtype=float, shape=[n,]
		The maximum fluorescence level.
	fvfm : np.array, dtype=float, shape=[n,]
		The maximum photochemical efficiency.
	sigma : np.array, dtype=float, shape=[n,]
		The effective absorption cross-section of PSII in Å\ :sup:`2`.
	rsq : np.array, dtype=float, shape=[n,]
		The r\ :sup:`2` value.
	bias : np.array, dtype=float, shape=[n,]
		The bias of the fit.
	chi : np.array, dtype=float, shape=[n,]
		The chi-squared goodness of the fit.
	fo_err : np.array, dtype=float, shape=[n,]
		The fit error of F\ :sub:`o`.
	fm_err : np.array, dtype=float, shape=[n,]
		The fit error of F\ :sub:`m`.
	sigma_err : np.array, dtype=float, shape=[n,]
		The fit error of σ\ :sub:`PSII`.
	nfl : np.array, dtype=int, shape=[n,]
		The number of flashlets used for fitting.

	Example
	-------
	>>> sat = ppu.calculate_saturation_with_nopmodel(pfd, fyield, seq, datetime, blank=0, sat_len=100, skip=0, sig_lims =[100,2200])
	"""

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

