#!/usr/bin/env python

from numpy import count_nonzero, isnan, inf, linalg, arange
from scipy.optimize import least_squares
from sklearn import linear_model
from ._equations import __fit_kolber__, __fit_single_relaxation__, __fit_triple_relaxation__,__calculate_residual_saturation__, __calculate_residual_saturation_pmodel__, __calculate_residual_single_relaxation__, __calculate_residual_triple_relaxation__, __calculate_rsquared__, __calculate_bias__, __calculate_chisquared__, __calculate_fit_errors__
	
	
def __fit_fixed_p_model__(pfd, fyield, ro, bounds=False, sig_lims=None, method='trf', loss='soft_l1', f_scale=0.1, max_nfev=1000, xtol=1e-9):

	# Count number of flashlets excluding NaNs
	fyield = fyield[~isnan(fyield)]
	nfl = count_nonzero(fyield)

	# Estimates of saturation parameters
	model = linear_model.HuberRegressor()
	try:
	y = fyield[:8]
		x = arange(0,8)[:,None]
		fo_model = model.fit(x,y)
		fo = fo_model.intercept_
	except Exception:
		fo = fyield[:3].mean()
	
	try:
		y = fyield[76:]
		x = arange(0,24)[:,None]
		fm_model = model.fit(x,y)
		fm = fm_model.intercept_
	except Exception:
		fm = fyield[-3:].mean()
	
	if (fo > fm) | (fo <= 0):
		(print('Fo greater than Fm - skipping fit.'))
		pass
	
	else:
		fo10 = fo * 0.1
		fm10 = fm * 0.1
		sig = 500                   
		x0 = [fo, fm, sig]

		bds = [-inf, inf]
		if bounds:
			bds = [fo-fo10, fm-fm10, sig_lims[0]],[fo+fo10, fm+fm10, sig_lims[1]]
			if (bds[0][0] > bds[1][0]) | (bds[0][1] > bds[1][1]) | (bds[0][2] > bds[1][2]):
				print('Lower bounds greater than upper bounds - fitting with no bounds.')
				bds = [-inf, inf]
		
		opts = {'method':method, 'loss':loss, 'f_scale':f_scale, 'max_nfev':max_nfev, 'xtol':xtol} 
		try:
			popt = least_squares(__calculate_residual_saturation__, x0, bounds=(bds), args=(pfd, fyield, ro), **opts)
			fo = popt.x[0]
			fm = popt.x[1]
			sigma = popt.x[2]

			# Calculate curve fitting statistical metrics
			res = fyield - __fit_kolber__(pfd, *popt.x, ro)
			rsq = __calculate_rsquared__(res, fyield)
			bias = __calculate_bias__(res, fyield)
			chi = __calculate_chisquared__(res, fyield)		
			perr = __calculate_fit_errors__(popt, res)
			fo_err = perr[0]
			fm_err = perr[1]
			sigma_err = perr[2]

			return fo, fm, sigma, rsq, bias, chi, fo_err, fm_err, sigma_err, nfl
		
		except linalg.LinAlgError as err:
			if str(err) == 'Singular matrix':
				print('Unable to calculate fitting errors, skipping sequence.'),
				pass

def __fit_calc_p_model__(pfd, fyield, bounds=False, sig_lims=None, ro_lims=None, method='trf', loss='soft_l1', f_scale=0.1, max_nfev=1000, xtol=1e-9):

	# Count number of flashlets excluding NaNs
	fyield = fyield[~isnan(fyield)]
	nfl = count_nonzero(fyield)
    
	# Estimates of saturation parameters
	model = linear_model.HuberRegressor()
	try:
		y = fyield[:8]
		x = arange(0,8)[:,None]
		fo_model = model.fit(x,y)
		fo = fo_model.intercept_
	except Exception:
		fo = fyield[:3].mean()
	
	try:
		y = fyield[76:]
		x = arange(0,24)[:,None]
		fm_model = model.fit(x,y)
		fm = fm_model.intercept_
	except Exception:
		fm = fyield[-3:].mean()

	if (fo > fm) | (fo <= 0):
		(print('Fo greater than Fm - skipping fit.'))
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

		opts = {'method':method, 'loss':loss, 'f_scale':f_scale, 'max_nfev':max_nfev, 'xtol':xtol} 

		try:
			popt = least_squares(__calculate_residual_saturation_pmodel__, x0, bounds=(bds), args=(pfd, fyield), **opts)
			fo = popt.x[0]
			fm = popt.x[1]
			sigma = popt.x[2]
			ro = popt.x[3]

			# Calculate curve fitting statistical metrics
			res = fyield - __fit_kolber__(pfd, *popt.x)
			rsq = __calculate_rsquared__(res, fyield)
			bias = __calculate_bias__(res, fyield)
			chi = __calculate_chisquared__(res, fyield)		
			perr = __calculate_fit_errors__(popt, res)
			fo_err = perr[0]
			fm_err = perr[1]
			sigma_err = perr[2]
			ro_err = perr[3]

			return fo, fm, sigma, ro, rsq, bias, chi, fo_err, fm_err, sigma_err, ro_err, nfl
		
		except linalg.LinAlgError as err:
			if str(err) == 'Singular matrix':
				print('Unable to calculate fitting errors, skipping sequence.'),
				pass

def __fit_no_p_model__(pfd, fyield, ro=None, bounds=False, sig_lims=None, method='trf', loss='soft_l1', f_scale=0.1, max_nfev=1000, xtol=1e-9):
	
	# Count number of flashlets excluding NaNs
	fyield = fyield[~isnan(fyield)]
	nfl = count_nonzero(fyield)

	# Estimates of saturation parameters
	model = linear_model.HuberRegressor()
	try:
		y = fyield[:8]
		x = arange(0,8)[:,None]
		fo_model = model.fit(x,y)
		fo = fo_model.intercept_
	except Exception:
		fo = fyield[:3].mean()
	
	try:
		y = fyield[76:]
		x = arange(0,24)[:,None]
		fm_model = model.fit(x,y)
		fm = fm_model.intercept_
	except Exception:
		fm = fyield[-3:].mean()

	if (fo > fm) | (fo <= 0):
		(print('Fo greater than Fm - skipping fit.'))
		pass

	else:
		fo10 = fo * 0.1
		fm10 = fm * 0.1
		sig = 500                   
		x0 = [fo, fm, sig]

		opts = {'method':method, 'loss':loss, 'f_scale':f_scale, 'max_nfev':max_nfev, 'xtol':xtol} 

		bds = [-inf, inf]
		if bounds:
			bds = [fo-fo10, fm-fm10, sig_lims[0]],[fo+fo10, fm+fm10, sig_lims[1]]
			if (bds[0][0] > bds[1][0]) | (bds[0][1] > bds[1][1]) | (bds[0][2] > bds[1][2]): #| (bds[0][0] == 0):
				print('Lower bounds greater than upper bounds - fitting with no bounds.')
				bds = [-inf, inf]

		try:
			popt = least_squares(__calculate_residual_saturation__, x0, bounds=(bds), args=(pfd, fyield, ro), **opts)
			fo = popt.x[0] 
			fm = popt.x[1]
			sigma = popt.x[2]

			# Calculate curve fitting statistical metrics
			res = fyield - __fit_kolber__(pfd, *popt.x, ro)
			rsq = __calculate_rsquared__(res, fyield)
			bias = __calculate_bias__(res, fyield)
			chi = __calculate_chisquared__(res, fyield)		
			perr = __calculate_fit_errors__(popt, res)
			fo_err = perr[0]
			fm_err = perr[1]
			sigma_err = perr[2]

			return fo, fm, sigma, rsq, bias, chi, fo_err, fm_err, sigma_err, nfl
		
		except linalg.LinAlgError as err:
			if str(err) == 'Singular matrix':
				print('Unable to calculate fitting errors, skipping sequence.'),
				pass

def __fit_single_decay__(seq_time, fyield, bounds=False, tau_lims=None, method='trf', loss='soft_l1', f_scale=0.1, max_nfev=1000, xtol=1e-9):
   
	# Count number of flashlets excluding NaNs
	fyield = fyield[~isnan(fyield)]
	nfl = count_nonzero(fyield)

	# Estimates of relaxation parameters
	fo_relax = fyield[-3:].mean()
	fm_relax = fyield[:3].mean()
	fo10 = fo_relax * 0.1
	fm10 = fm_relax * 0.1
	tau = 4000
	x0 = [fo_relax, fm_relax, tau]

	bds = [-inf, inf]
	if bounds:
		bds = [fo_relax-fo10, fm_relax-fm10, tau_lims[0]],[fo_relax+fo10, fm_relax+fm10, tau_lims[1]]
		if (bds[0][0] > bds[1][0]) | (bds[0][1] > bds[1][1]) | (bds[0][2] > bds[1][2]):
			print('Lower bounds greater than upper bounds - fitting with no bounds.')
			bds = [-inf, inf]
	
	opts = {'method':method,'loss':loss, 'f_scale':f_scale, 'max_nfev':max_nfev, 'xtol':xtol} 

	try: 	
		popt = least_squares(__calculate_residual_single_relaxation__, x0, bounds=(bds), args=(seq_time, fyield), **opts)
		fo_r =  popt.x[0]
		fm_r = popt.x[1]
		tau = popt.x[2]

		# Calculate curve fitting statistical metrics
		res = fyield - __fit_single_relaxation__(seq_time, *popt.x)
		rsq = __calculate_rsquared__(res, fyield)
		bias = __calculate_bias__(res, fyield)
		chi = __calculate_chisquared__(res, fyield)		
		perr = __calculate_fit_errors__(popt, res)
		fo_err = perr[0]
		fm_err = perr[1]
		tau_err = perr[2]

		return  fo_r, fm_r, tau, rsq, bias, chi, fo_err, fm_err, tau_err, nfl
	except linalg.LinAlgError as err:
		if str(err) == 'Singular matrix':
			print('Unable to calculate fitting errors, skipping sequence.'),
			pass
	except Exception:
		print('Unable to calculate fit, skipping sequence.'),
		pass


def __fit_triple_decay__(seq_time, fyield, bounds=False, tau1_lims=None, tau2_lims=None, tau3_lims=None, method='trf', loss='soft_l1', f_scale=0.1, max_nfev=1000, xtol=1e-9):
    
	# Count number of flashlets excluding NaNs
	fyield = fyield[~isnan(fyield)]
	nfl = count_nonzero(fyield)

	# Estimates of relaxation parameters
	fo_relax = fyield[-3:].mean()
	fm_relax = fyield[:3].mean()
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
	
	opts = {'method':method,'loss':loss, 'f_scale':f_scale, 'max_nfev':max_nfev, 'xtol':xtol} 

	try: 	
		popt = least_squares(__calculate_residual_triple_relaxation__, x0, bounds=(bds), args=(seq_time, fyield), **opts)
		fo_r =  popt.x[0]
		fm_r = popt.x[1]
		a1 = popt.x[2]
		t1 = popt.x[3]
		a2 = popt.x[4]
		t2 = popt.x[5]
		a3 = popt.x[6]
		t3 = popt.x[7]

		# Calculate curve fitting statistical metrics
		res = fyield - __fit_triple_relaxation__(seq_time, *popt.x)
		rsq = __calculate_rsquared__(res, fyield)
		bias = __calculate_bias__(res, fyield)
		chi = __calculate_chisquared__(res, fyield)		
		perr = __calculate_fit_errors__(popt, res)
		fo_err = perr[0]
		fm_err = perr[1]
		a1_err = perr[2]
		t1_err = perr[3]
		a2_err = perr[4]
		t2_err = perr[5]
		a3_err = perr[6]
		t3_err = perr[7]

		return  fo_r, fm_r, a1, t1, a2, t2, a3, t3, rsq, bias, chi, fo_err, fm_err, a1_err, t1_err, a2_err, t2_err, a3_err, t3_err, nfl
	except linalg.LinAlgError as err:
		if str(err) == 'Singular matrix':
			print('Unable to calculate fitting errors, skipping sequence.'),
			pass
	except Exception:
		print('Unable to calculate fit, skipping sequence.'),
		pass
