#!/usr/bin/env python

from numpy import exp, sum, mean, sqrt, diag, linalg, arange, cumsum
from sklearn.metrics import mean_squared_error

def __fit_kolber_nop__(pfd, fo, fm, sig):
	return fo + (fm - fo) * (1 - exp(-sig * cumsum(pfd)))

def __calculate_residual_saturation_nop__(p, pfd, fyield):
	return  fyield - __fit_kolber_nop__(pfd, *p)

def __fit_kolber_p__(pfd, fo, fm, sig, ro):
	c = pfd[:] * 0.
	c[0] = pfd[0] * sig
	for i in arange(1, len(pfd)):
		c[i] = c[i-1] + pfd[i] * sig * (1 - c[i-1])/(1 - ro * c[i-1])
	return fo + (fm - fo) * c * (1-ro) / (1-c*ro)

def __calculate_residual_saturation_p__(p, pfd, fyield):
	return  fyield - __fit_kolber_p__(pfd, *p)

def __calculate_residual_saturation_fixedp__(p, pfd, fyield, ro):
	return  fyield - __fit_kolber_p__(pfd, *p, ro)

def __fit_single_relaxation__(seq_time, fo_relax, fm_relax, tau):
	return (fm_relax - (fm_relax - fo_relax) * (1 - exp(-seq_time/tau)))

def __calculate_residual_single_relaxation__(p, seq_time, fyield):
	return fyield - __fit_single_relaxation__(seq_time, *p)

def __fit_triple_relaxation__(seq_time, fo_relax, fm_relax, alpha1, tau1, alpha2, tau2, alpha3, tau3):
	return (fo_relax + (fm_relax - fo_relax) *(alpha1 * 
			exp(-seq_time / tau1) + alpha2 * exp(-seq_time / tau2) + alpha3 * exp(-seq_time / tau3)))

def __calculate_residual_triple_relaxation__(p, seq_time, fyield):
		return fyield - __fit_triple_relaxation__(seq_time, *p)

def __calculate_Webb_model__(E, P, a):
	return P * (1 - exp(-a * E/P))

def __calculate_residual_etr__(p, E, P):
	return P - __calculate_Webb_model__(E, *p)

def __calculate_modified_Webb_model__(E, P, a):
	return P * (1 - exp(-a * E/P)) * (E**-1)

def __calculate_residual_phi__(p, E, P):
	return P - __calculate_modified_Webb_model__(E, *p)

def __calculate_rsquared__(res, fyield):
	return 1 - (sum(res**2)/sum((fyield - mean(fyield))**2))

def __calculate_bias__(res, fyield):
	return sum((1 - res)/fyield) / (len(fyield)*100)

def __calculate_chisquared__(res):
	return sum(res**2)

def __calculate_reduced_chisquared__(chi, fyield, nvars):
	return chi/(len(fyield) - nvars)

def __calculate_rmse__(res, fyield):
	return sqrt(mean_squared_error(fyield, res+fyield))	

def __calculate_fit_errors__(jac, res):
	#J = popt.jac
	pcov = linalg.inv(jac.T.dot(jac)) * mean(res**2)
	return sqrt(diag(pcov))
