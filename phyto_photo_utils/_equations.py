#!/usr/bin/env python

from numpy import exp, sum, mean, sqrt, diag, linalg, arange, cumsum

def __fit_kolber__(pfd, fo, fm, sig, ro):
	if ro is None:
		return fo + (fm - fo) * (1 - exp(-sig * cumsum(pfd)))
	else:
		c = pfd[:] * 0.
		c[0] = pfd[0] * sig
		for i in arange(1, len(pfd)):
			c[i] = c[i-1] + pfd[i] * sig * (1 - c[i-1])/(1 - ro * c[i-1])
		return fo + (fm - fo) * c * (1-ro) / (1-c*ro)

def __calculate_residual_saturation__(p, pfd, fyield, ro=None):
	return fyield - __fit_kolber__(pfd, *p, ro)

def __calculate_residual_saturation_pmodel__(p, pfd, fyield):
	return fyield - __fit_kolber__(pfd, *p)

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

def __calculate_chisquared__(res, fyield):
	return sum(res**2 / fyield)	

def __calculate_fit_errors__(popt, res):
	J = popt.jac
	pcov = linalg.inv(J.T.dot(J)) * mean(res**2)
	return sqrt(diag(pcov))
