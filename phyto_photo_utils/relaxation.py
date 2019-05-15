#!/usr/bin/env python
"""
@package phyto_photo_utils.relaxation
@file phyto_photo_utils/relaxation.py
@author Thomas Ryan-Keogh
@brief module containing the processing functions for calculating the relaxation of fluorescence transients
"""

def calc_single(fyield, seq_time, seq, datetime, blank=10, sat_len=100, rel_len=60, sat_flashlets=0, bounds=True, tau_lims=[100,50000], fit_method='trf', loss_method='soft_l1', fscale=0.1, n_iter=1000, xtol=1e-9):
	"""

	Process the raw transient data and perform the Kolber et al. 1998 relaxation model.

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
	from scipy.optimize import least_squares
	from numpy import count_nonzero, mean, exp, isnan, sum, min, array, unique, sqrt, diag, inf, linalg
	from pandas import Series, concat
	from tqdm import tqdm
	#from sklearn.metrics import mean_squared_error
    
    # This is from ???
	def single_decay(seq_time, fyield):
		def rel_func(seq_time, fo_relax, fm_relax, tau):
			return (fm_relax - (fm_relax - fo_relax) * (1 - exp(-seq_time/tau)))
        
		# Count number of flashlets excluding NaNs
		fyield = fyield[~isnan(fyield)]
		nfl = count_nonzero(fyield)

		# Estimates of relaxation parameters
		fo_relax = fyield[-3:].mean()
		fm_relax = fyield[:3].mean()
		
		# Rule checking to see if both are positive and that fm_relax is greater than fo_relax
		#if fm_relax < fo_relax:
		#	raise UserWarning('Fm_relax is not greater than Fo_relax - unable to perform optimization.')
		#if fm_relax < 0:
		#	raise UserWarning('Fm_relax is not positive - unable to perform optimization.')
		#if fo_relax < 0:
		#	raise UserWarning('Fo_relax is not positive - unable to perform optimization.')

		fo10 = fo_relax * 0.1
		fm10 = fm_relax * 0.1
		tau = 4000
		p0 = [fo_relax, fm_relax, tau]

		bds = [-inf, inf]
		if bounds:
			bds = [fo_relax-fo10, fm_relax-fm10, tau_lims[0]],[fo_relax+fo10, fm_relax+fm10, tau_lims[1]]
        
		# See scipy.optimize.least_squares documentation for more information on non-linear least squares fitting options
		opts = {'method':fit_method, 'loss':loss_method, 'f_scale':fscale, 'max_nfev':n_iter, 'xtol':xtol} 

		def residuals(p, seq_time, fyield):
			fo_relax, fm_relax, tau = p
			err = fyield - (fm_relax - (fm_relax - fo_relax) * (1 - exp(-seq_time/tau)))
			return err

		try:
			popt = least_squares(residuals, p0, bounds = (bds), args=(seq_time, fyield), **opts)

			fo_r =  popt.x[0]
			fm_r = popt.x[1]
			tau = popt.x[2]

			# Calculate curve fitting statistical metrics
			res = fyield - rel_func(seq_time, fo_r, fm_r, tau)
			rsq = 1 - (sum(res**2)/sum((fyield - mean(fyield))**2))
			bias = sum((1-res)/fyield) / (len(fyield)*100)
			chi = sum(res**2 / fyield)	
			J = popt.jac
			pcov = linalg.inv(J.T.dot(J)) * mean(res**2)
			perr = sqrt(diag(pcov))
			fo_err = perr[0]
			fm_err = perr[1]
			tau_err = perr[2]

			return  fo_r, fm_r, tau, rsq, bias, chi, fo_err, fm_err, tau_err, nfl
		except Exception:
			#print('Optimization was unable to be performed, parameters set to NaN.')
			pass

	seq_time = array(seq_time)
	fyield = array(fyield)
	seq = array(seq)
	dt = array(datetime)

	res = []
    
	for s in tqdm(unique(seq)):

		i = seq == s
		x = seq_time[i]
		y = fyield[i]
		x_min = min(x[sat_len:])
		x = x[sat_len-sat_flashlets:sat_len+rel_len] - x_min
		y = y[sat_len-sat_flashlets:sat_len+rel_len]

		rel = single_decay(x, y)
		res.append(Series(rel))

	res = concat(res, axis=1)
	res = res.T

	if res.empty:
		pass
	
	else: 
		res.columns = ['fo_r','fm_r','tau','rsq','bias','chi','fo_err','fm_err','tau_err','nfl']
		res['datetime'] = unique(dt)

	return res

def calc_triple(fyield, seq_time, seq, datetime, blank=10, sat_len=100, rel_len=60, sat_flashlets=0, bounds=True, tau1_lims=[100, 800], tau2_lims=[800, 2000], tau3_lims=[2000, 50000], fit_method='trf', loss_method='soft_l1', fscale=0.1, n_iter=1000, xtol=1e-9):
    
	"""
	
	Process the raw transient data and perform the Kolber et al. 1998 relaxation model.

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
	from scipy.optimize import least_squares
	from numpy import count_nonzero, mean, exp, isnan, sum, min, array, unique, sqrt, diag, inf, linalg
	from pandas import Series, concat
	from tqdm import tqdm
	#from sklearn.metrics import mean_squared_error

	# This is from Kolber et al. 1998 Equation 6
	def triple_decay(seq_time, fyield):
		def rel_func(seq_time, p):
			return (p[0] + (p[1] - p[0]) *(p[2] * exp(-seq_time / p[3]) 
				+ p[4] * exp(-seq_time / p[5]) + p[6] * exp(-seq_time / p[7])))
	    
		# Count number of flashlets excluding NaNs
		fyield = fyield[~isnan(fyield)]
		nfl = count_nonzero(fyield)

		# Estimates of relaxation parameters
		fo_relax = fyield[-3:].mean()
		fm_relax = fyield[:3].mean()

		# Rule checking to see if both are positive and that fm_relax is greater than fo_relax
		#if fm_relax < fo_relax:
		#	raise UserWarning('Fm_relax is not greater than Fo_relax - unable to perform optimization.')
		#if fm_relax < 0:
		#	raise UserWarning('Fm_relax is not positive - unable to perform optimization.')
		#if fo_relax < 0:
		#	raise UserWarning('Fo_relax is not positive - unable to perform optimization.')

		fo10 = fo_relax * 0.1
		fm10 = fm_relax * 0.1
		alpha1 = 0.3
		tau1 = 600
		alpha2 = 0.3
		tau2 = 2000
		alpha3 = 0.3
		tau3 = 30000
		# initial estimates of the parameters
		p0 = [fo_relax, fm_relax, alpha1, tau1, alpha2, tau2, alpha3, tau3]
	    
		bds = [-inf, inf]
		if bounds:
			bds = [fo_relax-fo10, fm_relax-fm10, 0.1, tau1_lims[0], 0.1, tau2_lims[0], 0.1, tau3_lims[0]],[fo_relax+fo10, fm_relax+fm10, 1, tau1_lims[1], 1, tau2_lims[1], 1, tau3_lims[1]]

		# See scipy.optimize.least_squares documentation for more information on non-linear least squares fitting options
		opts = {'method':fit_method,'loss':loss_method, 'f_scale':fscale, 'max_nfev':n_iter, 'xtol':xtol} 

		def residuals(p, seq_time, fyield):
			fo_relax, fm_relax, alpha1, tau1, alpha2, tau2, alpha3, tau3 = p
			err = fyield - (fo_relax + (fm_relax - fo_relax) *(alpha1 * 
			exp(-seq_time / tau1) + alpha2 * exp(-seq_time / tau2) + alpha3 * exp(-seq_time / tau3)))
			return err

		try:
			popt = least_squares(residuals, p0, bounds=(bds), args=(seq_time, fyield), **opts)

			fo_r =  popt.x[0]
			fm_r = popt.x[1]
			a1 = popt.x[2]
			t1 = popt.x[3]
			a2 = popt.x[4]
			t2 = popt.x[5]
			a3 = popt.x[6]
			t3 = popt.x[7]
	        
			# Calculate curve fitting statistical metrics
			res = fyield - rel_func(seq_time, [fo_r, fm_r, a1, t1, a2, t2, a3, t3])
			rsq = 1 - (sum(res**2) / sum((fyield - mean(fyield))**2))
			bias = sum((1-res)/fyield) / (len(fyield)*100)
			chi = sum(res**2 / fyield)
			#rms = sqrt(mean_squared_error(fyield, rel_func(seq_time, [fo_r, fm_r, a1, t1, a2, t2, a3, t3])))		
			J = popt.jac
			pcov = linalg.inv(J.T.dot(J)) * mean(res**2)
			perr = sqrt(diag(pcov))
			fo_err = perr[0]
			fm_err = perr[1]
			a1_err = perr[2]
			t1_err = perr[3]
			a2_err = perr[4]
			t2_err = perr[5]
			a3_err = perr[6]
			t3_err = perr[7]

			return  fo_r, fm_r, a1, t1, a2, t2, a3, t3, rsq, bias, chi, fo_err, fm_err, a1_err, t1_err, a2_err, t2_err, a3_err, t3_err, nfl
	    
		except Exception:
			#print('Optimization was unable to be performed, parameters set to NaN.')
			pass

	seq_time = array(seq_time)
	fyield = array(fyield)
	seq = array(seq)
	dt = array(datetime)

	res = []
	for s in tqdm(unique(seq)):

		i = seq == s
		x = seq_time[i]
		y = fyield[i]
		x_min = min(x[sat_len:])
		x = x[sat_len-sat_flashlets:sat_len+rel_len] - x_min
		y = y[sat_len-sat_flashlets:sat_len+rel_len]
	    
		rel = triple_decay(x, y)
		res.append(Series(rel))

	res = concat(res, axis=1)
	res = res.T
	
	if res.empty:
		pass
	
	else:
		res.columns = ['fo_r', 'fm_r', 'alpha1', 'tau1', 'alpha2', 'tau2', 'alpha3', 'tau3', 'rsq', 'bias', 'chi','for_err', 'fmr_err', 'alpha1_err', 'tau1_err', 'alpha2_err', 'tau2_err', 'alpha3_err', 'tau3_err', 'nfl']
		res['datetime'] = unique(dt)

	return res

