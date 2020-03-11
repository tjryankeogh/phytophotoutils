#!/usr/bin/env python

from numpy import count_nonzero, isnan, inf, linalg, arange, repeat, nan
from scipy.optimize import least_squares
from sklearn import linear_model
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from ._equations import __fit_kolber_nop__, __fit_kolber_p__, __fit_single_relaxation__, __fit_triple_relaxation__, __calculate_residual_saturation_p__, __calculate_residual_saturation_nop__, __calculate_residual_saturation_fixedp__, __calculate_residual_single_relaxation__, __calculate_residual_triple_relaxation__, __calculate_bias__, __calculate_rmse__, __calculate_fit_errors__
	
	
def __fit_fixed_p_model__(pfd, flevel, ro, bounds=False, sig_lims=None, method='trf', loss='soft_l1', f_scale=0.1, max_nfev=None, xtol=1e-9):

	# Count number of flashlets excluding NaNs
	nfl = count_nonzero(~isnan(flevel))
	m = ~isnan(flevel)
	flevel = flevel[m]
	pfd = pfd[m]

	# Estimates of saturation parameters
	model = linear_model.HuberRegressor()
	try:
		y = flevel[:8]
		x = arange(1,9)[:,None]
		fo_model = model.fit(x,y)
		fo = fo_model.intercept_
	except Exception:
		fo = flevel[:3].mean()
	
	try:
		y = flevel[-24:]
		x = arange(1,25)[:,None]
		fm_model = model.fit(x,y)
		fm = fm_model.intercept_
	except Exception:
		fm = flevel[-3:].mean()
	
	if (fo > fm) | (fo <= 0):
		(print('Fo greater than Fm - skipping fit.'))
		fo, fm, sigma, ro, bias, rmse, fo_err, fm_err, sigma_err, nfl, nfev = repeat(nan, 11)
		flag = -2
		success = 'False'
		return fo, fm, sigma, ro, bias, rmse, fo_err, fm_err, sigma_err, nfl, nfev, flag, success
		pass
	
	else:
		fo10 = fo * 0.1
		fm10 = fm * 0.1
		sig = 500                   
		x0 = [fo, fm, sig]

		bds = [-inf, inf]
		if bounds:
			bds = [fo-fo10, fm-fm10, sig_lims[0]], [fo+fo10, fm+fm10, sig_lims[1]]
			if (bds[0][0] > bds[1][0]) | (bds[0][1] > bds[1][1]) | (bds[0][2] > bds[1][2]):
				print('Lower bounds greater than upper bounds - fitting with no bounds.')
				bds = [-inf, inf]
		
		if max_nfev is None:
			opts = {'method':method, 'loss':loss, 'f_scale':f_scale, 'xtol':xtol} 
		else:
			opts = {'method':method, 'loss':loss, 'f_scale':f_scale, 'max_nfev':max_nfev, 'xtol':xtol} 

		try:
			popt = least_squares(__calculate_residual_saturation_fixedp__, x0, bounds=(bds), args=(pfd, flevel, ro), **opts)
			fo = popt.x[0]
			fm = popt.x[1]
			sigma = popt.x[2]

			# Calculate curve fitting statistical metrics
			sol = __fit_kolber_p__(pfd, *popt.x, ro)
			bias = __calculate_bias__(sol, flevel)
			rmse = __calculate_rmse__(popt.fun, flevel)			
			perr = __calculate_fit_errors__(popt.jac, popt.fun)
			fo_err = perr[0]
			fm_err = perr[1]
			sigma_err = perr[2]

			if max_nfev is None:
				nfev = popt.nfev
			else:
				nfev = max_nfev
			
			flag = popt.status
			success = popt.success
			
			return fo, fm, sigma, ro, bias, rmse, fo_err, fm_err, sigma_err, nfl, nfev, flag, success
		
		except linalg.LinAlgError as err:
			if str(err) == 'Singular matrix':
				print('Unable to calculate fitting errors, skipping sequence.'),
				fo, fm, sigma, ro, bias, rmse, fo_err, fm_err, sigma_err, nfl, nfev = repeat(nan, 11)
				flag = -3
				success = 'False'
				return fo, fm, sigma, ro, bias, rmse, fo_err, fm_err, sigma_err, nfl, nfev, flag, success
				pass
		except Exception:
			print('Unable to calculate fit, skipping sequence.'),
			fo, fm, sigma, ro, bias, rmse, fo_err, fm_err, sigma_err, nfl, nfev = repeat(nan, 11)
			flag = -1
			success = 'False'
			return fo, fm, sigma, ro, bias, rmse, fo_err, fm_err, sigma_err, nfl, nfev, flag, success
			pass

def __fit_calc_p_model__(pfd, flevel, bounds=False, sig_lims=None, ro_lims=None, method='trf', loss='soft_l1', f_scale=0.1, max_nfev=None, xtol=1e-9):

	# Count number of flashlets excluding NaNs
	nfl = count_nonzero(~isnan(flevel))
	m = ~isnan(flevel)
	flevel = flevel[m]
	pfd = pfd[m]
	
	# Estimates of saturation parameters
	model = linear_model.HuberRegressor()
	try:
		y = flevel[:8]
		x = arange(0,8)[:,None]
		fo_model = model.fit(x,y)
		fo = fo_model.intercept_
	except Exception:
		fo = flevel[:3].mean()
	
	try:
		y = flevel[-24:]
		x = arange(0,24)[:,None]
		fm_model = model.fit(x,y)
		fm = fm_model.intercept_
	except Exception:
		fm = flevel[-3:].mean()

	if (fo > fm) | (fo <= 0):
		(print('Fo greater than Fm - skipping fit.'))
		fo, fm, sigma, ro, bias, rmse, fo_err, fm_err, sigma_err, ro_err, nfl, nfev = repeat(nan, 12)
		flag = -2
		success = 'False'
		return fo, fm, sigma, ro, bias, rmse, fo_err, fm_err, sigma_err, ro_err, nfl, nfev, flag, success
		pass
	
	else:
		fo10 = fo * 0.1
		fm10 = fm * 0.1
		sig = 500
		ro = 0.3
		x0 = [fo, fm, sig, ro]

		bds = [-inf, inf]
		if bounds:
			bds = [fo-fo10, fm-fm10, sig_lims[0], ro_lims[0]],[fo+fo10, fm+fm10, sig_lims[1], ro_lims[1]]
			if (bds[0][0] > bds[1][0]) | (bds[0][1] > bds[1][1]) | (bds[0][2] > bds[1][2]) | (bds[0][3] > bds[1][3]): #| (bds[0][0] == 0):
				print('Lower bounds greater than upper bounds - fitting with no bounds.')
				bds = [-inf, inf]
		
		if max_nfev is None:
			opts = {'method':method, 'loss':loss, 'f_scale':f_scale, 'xtol':xtol} 
		else:
			opts = {'method':method, 'loss':loss, 'f_scale':f_scale, 'max_nfev':max_nfev, 'xtol':xtol} 

		try:
			popt = least_squares(__calculate_residual_saturation_p__, x0, bounds=(bds), args=(pfd, flevel), **opts)
			fo = popt.x[0]
			fm = popt.x[1]
			sigma = popt.x[2]
			ro = popt.x[3]

			# Calculate curve fitting statistical metrics
			sol = __fit_kolber_p__(pfd, *popt.x)
			bias = __calculate_bias__(sol, flevel)
			rmse = __calculate_rmse__(popt.fun, flevel)			
			perr = __calculate_fit_errors__(popt.jac, popt.fun)
			fo_err = perr[0]
			fm_err = perr[1]
			sigma_err = perr[2]
			ro_err = perr[3]
			
			if max_nfev is None:
				nfev = popt.nfev
			else:
				nfev = max_nfev
			
			flag = popt.status
			status = popt.success
			
			return fo, fm, sigma, ro, bias, rmse, fo_err, fm_err, sigma_err, ro_err, nfl, nfev, flag, status
		
		except linalg.LinAlgError as err:
			if str(err) == 'Singular matrix':
				print('Unable to calculate fitting errors, skipping sequence.'),
				fo, fm, sigma, ro, bias, rmse, fo_err, fm_err, sigma_err, ro_err, nfl, nfev = repeat(nan, 12)
				flag = -3
				success = 'False'
				return fo, fm, sigma, ro, bias, rmse, fo_err, fm_err, sigma_err, ro_err, nfl, nfev, flag, success
				pass

		except Exception:
			print('Unable to calculate fit, skipping sequence.'),
			fo, fm, sigma, ro, bias, rmse, fo_err, fm_err, sigma_err, ro_err, nfl, nfev = repeat(nan, 12)
			flag = -1
			success = 'False'
			return fo, fm, sigma, ro, bias, rmse, fo_err, fm_err, sigma_err, ro_err, nfl, nfev, flag, success
			pass

def __fit_no_p_model__(pfd, flevel, ro=None, bounds=False, sig_lims=None, method='trf', loss='soft_l1', f_scale=0.1, max_nfev=None, xtol=1e-9):
	
	# Count number of flashlets excluding NaNs
	nfl = count_nonzero(~isnan(flevel))
	m = ~isnan(flevel)
	flevel = flevel[m]
	pfd = pfd[m]

	# Estimates of saturation parameters
	model = linear_model.HuberRegressor()
	try:
		y = flevel[:8]
		x = arange(0,8)[:,None]
		fo_model = model.fit(x,y)
		fo = fo_model.intercept_
	except Exception:
		fo = flevel[:3].mean()
	
	try:
		y = flevel[-24:]
		x = arange(0,24)[:,None]
		fm_model = model.fit(x,y)
		fm = fm_model.intercept_
	except Exception:
		fm = flevel[-3:].mean()

	if (fo > fm) | (fo <= 0):
		(print('Fo greater than Fm - skipping fit.'))
		fo, fm, sigma, bias, rmse, fo_err, fm_err, sigma_err, nfev = repeat(nan, 9)
		flag = -2
		success = False
		return fo, fm, sigma, bias, rmse, fo_err, fm_err, sigma_err, nfl, nfev, flag, success
		pass

	else:
		fo10 = fo * 0.1
		fm10 = fm * 0.1
		sig = 500                   
		x0 = [fo, fm, sig]

		bds = [-inf, inf]
		if bounds:
			bds = [fo-fo10, fm-fm10, sig_lims[0]],[fo+fo10, fm+fm10, sig_lims[1]]
			if (bds[0][0] > bds[1][0]) | (bds[0][1] > bds[1][1]) | (bds[0][2] > bds[1][2]): #| (bds[0][0] == 0):
				print('Lower bounds greater than upper bounds - fitting with no bounds.')
				bds = [-inf, inf]

		if max_nfev is None:
			opts = {'method':method, 'loss':loss, 'f_scale':f_scale, 'xtol':xtol} 
		else:
			opts = {'method':method, 'loss':loss, 'f_scale':f_scale, 'max_nfev':max_nfev, 'xtol':xtol} 

		try:
			popt = least_squares(__calculate_residual_saturation_nop__, x0, bounds=(bds), args=(pfd, flevel), **opts)
			fo = popt.x[0] 
			fm = popt.x[1]
			sigma = popt.x[2]

			# Calculate curve fitting statistical metrics
			sol = __fit_kolber_nop__(pfd, *popt.x)
			bias = __calculate_bias__(sol, flevel)
			rmse = __calculate_rmse__(popt.fun, flevel)			
			perr = __calculate_fit_errors__(popt.jac, popt.fun)
			fo_err = perr[0]
			fm_err = perr[1]
			sigma_err = perr[2]

			if max_nfev is None:
				nfev = popt.nfev
			else:
				nfev = max_nfev
			
			flag = popt.status
			success = popt.success
			
			return fo, fm, sigma, bias, rmse, fo_err, fm_err, sigma_err, nfl, nfev, flag, success
		
		except linalg.LinAlgError as err:
			if str(err) == 'Singular matrix':
				print('Unable to calculate fitting errors, skipping sequence.'),
				fo, fm, sigma, bias, rmse, fo_err, fm_err, sigma_err, nfl, nfev = repeat(nan, 10)
				flag = -3
				success = 'False'
				return fo, fm, sigma, bias, rmse, fo_err, fm_err, sigma_err, nfl, nfev, flag, success
				pass

		except Exception:
			print('Unable to calculate fit, skipping sequence.'),
			fo, fm, sigma, bias, rmse, fo_err, fm_err, sigma_err, nfl, nfev = repeat(nan, 10)
			flag = -1
			success = 'False'
			return fo, fm, sigma, bias, rmse, fo_err, fm_err, sigma_err, nfl, nfev, flag, success
			pass

def __fit_single_decay__(seq_time, flevel, sat_flashlets=None, bounds=False, single_lims=None, method='trf', loss='soft_l1', f_scale=0.1, max_nfev=None, xtol=1e-9):
   
	# Count number of flashlets excluding NaNs
	nfl = count_nonzero(~isnan(flevel))
	m = ~isnan(flevel)
	flevel = flevel[m]
	seq_time = seq_time[m]

	# Estimates of relaxation parameters
	fo_relax = flevel[-3:].mean()
	
	if sat_flashlets is None:
		fm_relax = flevel[:3].mean()
	else:
		fm_relax = flevel[:3+sat_flashlets].mean()

	if (fo_relax > fm_relax):
		(print('Fo_relax greater than Fm_relax - skipping fit.'))
		fo_r, fm_r, tau, bias, rmse, fo_err, fm_err, tau_err, nfev = repeat(nan, 9)
		flag = -2
		success = 'False'
		return fo_r, fm_r, tau, bias, rmse, fo_err, fm_err, tau_err, nfl, nfev, flag, success
		pass

	fo10 = fo_relax * 0.1
	fm10 = fm_relax * 0.1
	tau = 4000
	x0 = [fo_relax, fm_relax, tau]

	bds = [-inf, inf]
	if bounds:
		bds = [fo_relax-fo10, fm_relax-fm10, single_lims[0]],[fo_relax+fo10, fm_relax+fm10, single_lims[1]]
		if (bds[0][0] > bds[1][0]) | (bds[0][1] > bds[1][1]) | (bds[0][2] > bds[1][2]):
			print('Lower bounds greater than upper bounds - fitting with no bounds.')
			bds = [-inf, inf]
	
	if max_nfev is None:
		opts = {'method':method, 'loss':loss, 'f_scale':f_scale, 'xtol':xtol} 
	else:
		opts = {'method':method, 'loss':loss, 'f_scale':f_scale, 'max_nfev':max_nfev, 'xtol':xtol} 

	try: 	
		popt = least_squares(__calculate_residual_single_relaxation__, x0, bounds=(bds), args=(seq_time, flevel), **opts)
		fo_r =  popt.x[0]
		fm_r = popt.x[1]
		tau = popt.x[2]

		# Calculate curve fitting statistical metrics
		sol = __fit_single_relaxation__(seq_time, *popt.x)
		bias = __calculate_bias__(sol, flevel)
		rmse = __calculate_rmse__(popt.fun, flevel)			
		perr = __calculate_fit_errors__(popt.jac, popt.fun)
		fo_err = perr[0]
		fm_err = perr[1]
		tau_err = perr[2]

		if max_nfev is None:
			nfev = popt.nfev
		else:
			nfev = max_nfev
			
		flag = popt.status
		success = popt.success

		return  fo_r, fm_r, tau, bias, rmse, fo_err, fm_err, tau_err, nfl, nfev, flag, success
	
	except linalg.LinAlgError as err:
		if str(err) == 'Singular matrix':
			print('Unable to calculate fitting errors, skipping sequence.'),
			fo_r, fm_r, tau, bias, rmse, fo_err, fm_err, tau_err, nfev = repeat(nan, 9)
			flag = -3
			success = 'False'
			return fo_r, fm_r, tau, bias, rmse, fo_err, fm_err, tau_err, nfl, nfev, flag, success
			pass
	
	except Exception:
		print('Unable to calculate fit, skipping sequence.'),
		fo_r, fm_r, tau, bias, rmse, fo_err, fm_err, tau_err, nfev = repeat(nan, 9)
		flag = -1
		success = 'False'
		return fo_r, fm_r, tau, bias, rmse, fo_err, fm_err, tau_err, nfl, nfev, flag, success
		pass


def __fit_triple_decay__(seq_time, flevel, sat_flashlets=None, bounds=False, tau1_lims=None, tau2_lims=None, tau3_lims=None, method='trf', loss='soft_l1', f_scale=0.1, max_nfev=None, xtol=1e-9):
    
	# Count number of flashlets excluding NaNs
	nfl = count_nonzero(~isnan(flevel))
	m = ~isnan(flevel)
	flevel = flevel[m]
	seq_time = seq_time[m]

	# Estimates of relaxation parameters
	fo_relax = flevel[-3:].mean()
	
	if sat_flashlets is None:
		fm_relax = flevel[:3].mean()
	else:
		fm_relax = flevel[:3+sat_flashlets].mean()

	if (fo_relax > fm_relax):
		(print('Fo_relax greater than Fm_relax - skipping fit.'))
		fo_r, fm_r, a1, t1, a2, t2, a3, t3, bias, rmse, fo_err, fm_err, a1_err, t1_err, a2_err, t2_err, a3_err, t3_err, nfl, nfev = repeat(nan, 20)
		flag = -2
		success = 'False'
		return fo_r, fm_r, a1, t1, a2, t2, a3, t3, bias, rmse, fo_err, fm_err, a1_err, t1_err, a2_err, t2_err, a3_err, t3_err, nfl, nfev, flag, success
		pass

	fo10 = fo_relax * 0.1
	fm10 = fm_relax * 0.1
	alpha1 = 0.3
	tau1 = 600
	alpha2 = 0.3
	tau2 = 2000
	alpha3 = 0.3
	tau3 = 30000
	x0 = [fo_relax, fm_relax, alpha1, tau1, alpha2, tau2, alpha3, tau3]
    
	bds = [-inf, inf]
	if bounds:
		bds = [fo_relax-fo10, fm_relax-fm10, 0.1, tau1_lims[0], 0.1, tau2_lims[0], 0.1, tau3_lims[0]],[fo_relax+fo10, fm_relax+fm10, 1, tau1_lims[1], 1, tau2_lims[1], 1, tau3_lims[1]]
		if (bds[0][0] > bds[1][0]) | (bds[0][1] > bds[1][1]) | (bds[0][2] > bds[1][2]) | (bds[0][3] > bds[1][3]) | (bds[0][4] > bds[1][4]) | (bds[0][5] > bds[1][5]) | (bds[0][6] > bds[1][6]) | (bds[0][7] > bds[1][7]):
			print('Lower bounds greater than upper bounds - fitting with no bounds.')
			bds = [-inf, inf]
	
	if max_nfev is None:
		opts = {'method':method, 'loss':loss, 'f_scale':f_scale, 'xtol':xtol} 
	else:
		opts = {'method':method, 'loss':loss, 'f_scale':f_scale, 'max_nfev':max_nfev, 'xtol':xtol} 

	try: 	
		popt = least_squares(__calculate_residual_triple_relaxation__, x0, bounds=(bds), args=(seq_time, flevel), **opts)
		fo_r =  popt.x[0]
		fm_r = popt.x[1]
		a1 = popt.x[2]
		t1 = popt.x[3]
		a2 = popt.x[4]
		t2 = popt.x[5]
		a3 = popt.x[6]
		t3 = popt.x[7]

		# Calculate curve fitting statistical metrics
		sol = __fit_single_relaxation__(seq_time, *popt.x)
		bias = __calculate_bias__(sol, flevel)
		rmse = __calculate_rmse__(popt.fun, flevel)			
		perr = __calculate_fit_errors__(popt.jac, popt.fun)
		fo_err = perr[0]
		fm_err = perr[1]
		a1_err = perr[2]
		t1_err = perr[3]
		a2_err = perr[4]
		t2_err = perr[5]
		a3_err = perr[6]
		t3_err = perr[7]

		if max_nfev is None:
			nfev = popt.nfev
		else:
			nfev = max_nfev
		
		flag = popt.status
		success = popt.success

		return  fo_r, fm_r, a1, t1, a2, t2, a3, t3, bias, rmse, fo_err, fm_err, a1_err, t1_err, a2_err, t2_err, a3_err, t3_err, nfl, nfev, flag, success
	
	except linalg.LinAlgError as err:
		if str(err) == 'Singular matrix':
			print('Unable to calculate fitting errors, skipping sequence.'),
			fo_r, fm_r, a1, t1, a2, t2, a3, t3, bias, rmse, fo_err, fm_err, a1_err, t1_err, a2_err, t2_err, a3_err, t3_err, nfl, nfev = repeat(nan, 20)
			flag = -3
			success = 'False'
			return fo_r, fm_r, a1, t1, a2, t2, a3, t3, bias, rmse, fo_err, fm_err, a1_err, t1_err, a2_err, t2_err, a3_err, t3_err, nfl, nfev, flag, success
			pass
	
	except Exception:
		print('Unable to calculate fit, skipping sequence.'),
		fo_r, fm_r, a1, t1, a2, t2, a3, t3, bias, rmse, fo_err, fm_err, a1_err, t1_err, a2_err, t2_err, a3_err, t3_err, nfl, nfev = repeat(nan, 20)
		flag = -1
		success = 'False'
		return fo_r, fm_r, a1, t1, a2, t2, a3, t3, bias, rmse, fo_err, fm_err, a1_err, t1_err, a2_err, t2_err, a3_err, t3_err, nfl, nfev, flag, success
		pass

