#!/usr/bin/env python

from numpy import exp, sum, mean, sqrt, diag, linalg, arange, cumsum, nansum, isnan, log10, max, min
from sklearn.metrics import mean_squared_error

def __fit_kolber_nop__(pfd, fo, fm, sig):
	return fo + (fm - fo) * (1 - exp(-sig * cumsum(pfd)))

def __calculate_residual_saturation_nop__(p, pfd, flevel):
	return  flevel - __fit_kolber_nop__(pfd, *p)

def __fit_kolber_p__(pfd, fo, fm, sig, ro):
	c = pfd[:] * 0.
	c[0] = pfd[0] * sig
	for i in arange(1, len(pfd)):
		c[i] = c[i-1] + pfd[i] * sig * (1 - c[i-1])/(1 - ro * c[i-1])
	return fo + (fm - fo) * c * (1-ro) / (1-c*ro)

def __calculate_residual_saturation_p__(p, pfd, flevel):
	return  flevel - __fit_kolber_p__(pfd, *p)

def __calculate_residual_saturation_fixedp__(p, pfd, flevel, ro):
	return  flevel - __fit_kolber_p__(pfd, *p, ro)

def __fit_kolber_transient__(pfd, seq_time, fo, fm, sig, ro, tau):
	c = pfd[:] * 0.
	c[0] = pfd[0] * sig
	for i in arange(1, len(pfd)):
		c[i] = c[i-1] + pfd[i] * sig * (((1 - c[i-1])/(1 - ro * c[i-1])) - c(i-1) * (seq_time[i] - seq_time[i-1])/tau)
	return fo + (fm - fo) * c * (1-ro) / (1-c*ro)

def __calculate_residual_transient__(p, pfd, flevel, seq_time):
	return  flevel - __fit_kolber_transient__(pfd, *p)

def __fit_single_relaxation__(seq_time, fo_relax, fm_relax, tau):
	return (fm_relax - (fm_relax - fo_relax) * (1 - exp(-seq_time/tau)))

def __calculate_residual_single_relaxation__(p, seq_time, flevel):
	return flevel - __fit_single_relaxation__(seq_time, *p)

def __fit_double_relaxation__(seq_time, fo_relax, fm_relax, alpha1, tau1, alpha2, tau2):
	return (fo_relax + (fm_relax - fo_relax) *(alpha1 * 
			exp(-seq_time / tau1) + alpha2 * exp(-seq_time / tau2)))

def __calculate_residual_double_relaxation__(p, seq_time, flevel):
		return flevel - __fit_double_relaxation__(seq_time, *p)

def __fit_triple_relaxation__(seq_time, fo_relax, fm_relax, alpha1, tau1, alpha2, tau2, alpha3, tau3):
	return (fo_relax + (fm_relax - fo_relax) *(alpha1 * 
			exp(-seq_time / tau1) + alpha2 * exp(-seq_time / tau2) + alpha3 * exp(-seq_time / tau3)))

def __calculate_residual_triple_relaxation__(p, seq_time, flevel):
		return flevel - __fit_triple_relaxation__(seq_time, *p)

def __calculate_alpha_model__(E, P, a):
	return P * (1 - exp(-a * E/P))

def __calculate_residual_etr__(p, E, P):
	return P - __calculate_alpha_model__(E, *p)

def __calculate_modified_alpha_model__(E, P, a):
	return P * (1 - exp(-a * E/P)) * (E**-1)

def __calculate_residual_phi__(p, E, P):
	return P - __calculate_modified_alpha_model__(E, *p)

def __calculate_beta_model__(E, P, B, a, Ek):
	return P * (1 - exp(-a * E/P) -B * Ek * (1- exp(-((E-P)/Ek))))

def __calculate_residual_beta__(p, E, P, a, Ek):
	return P - __calculate_beta_model__(E, *p, a, Ek)

def __calculate_modified_beta_model__(E, P, B, a, Ek):
	return P * (1 - exp(-a * E/P) -B * Ek * (1- exp(-((E-P)/P)))) * (E**-1)

def __calculate_residual_mbeta__(p, E, P, a, Ek):
	return P - __calculate_modified_beta_model__(E, *p, a, Ek)

def __calculate_bias__(sol, flevel):
	m = (isnan(sol)) | (isnan(flevel)) | (sol <= 0) | (flevel <= 0)
	return 10 ** ((nansum((sol[~m]) - (flevel[~m])) / mean((flevel[~m]))))

def __calculate_rmse__(res, flevel):
	return sqrt(mean_squared_error(flevel, res+flevel))	

def __calculate_nrmse__(res, flevel):
	return sqrt((mean_squared_error(flevel, res+flevel)) / (mean(flevel)))

def __calculate_fit_errors__(jac, res):
	pcov = linalg.inv(jac.T.dot(jac)) * mean(res**2)
	return sqrt(diag(pcov))

#def __calculate_bias__(sol, flevel):
#	m = (isnan(sol)) | (isnan(flevel)) | (sol <= 0) | (flevel <= 0)
#	return 10 ** ((nansum(log10(sol[~m]) - log10(flevel[~m])) / len(flevel[~m])))

#def __calculate_range_nrmse__(res, flevel):
#	return sqrt((mean_squared_error(flevel, res+flevel)) / (max(flevel) - min(flevel)))	

#def __calculate__chisquared__(res):
#	return sum(res**2)

#def __calculate_reduced_chisquared__(chi, flevel, nvars):
#	return chi/(len(flevel) - nvars)
