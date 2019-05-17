#!/usr/bin/env python
"""
@package phyto_photo_utils.saturation
@file phyto_photo_utils/saturation.py
@author Thomas Ryan-Keogh
@brief module containing statistical and fitting equations.
"""

def __fit_kolber__(pfd, fo, fm, sig, ro):
	from numpy import arange, cumsum, exp

	if ro is None:
		return fo + (fm - fo) * (1 - exp(-sig * cumsum(pfd)))
	else:
		c = pfd[:] * 0.
		c[0] = pfd[0] * sig
		for i in arange(1, len(pfd)):
			c[i] = c[i-1] + pfd[i] * sig * (1 - c[i-1])/(1 - ro * c[i-1])
		return fo + (fm - fo) * c * (1-ro) / (1-c*ro)


def __fit_single_relaxation__(seq_time, fo_relax, fm_relax, tau):
	from numpy import exp

	return (fm_relax - (fm_relax - fo_relax) * (1 - exp(-seq_time/tau)))


def __fit_triple_relaxation__(seq_time, fo_relax, fm_relax, alpha1, tau1, alpha2, tau2, alpha3, tau3):
	from numpy import exp

	return (fo_relax + (fm_relax - fo_relax) *(alpha1 * 
			exp(-seq_time / tau1) + alpha2 * exp(-seq_time / tau2) + alpha3 * exp(-seq_time / tau3)))


#def __calculate_residual_saturation__(p, pfd, fyield, ro=None):
#	if ro is None:
#		return fyield - __fit_kolber__(pfd, *p)
#	else:
#		return fyield - __fit_kolber__(pfd, *p, ro)


#def __calculate_residual_relaxation__(p, seq_time, fyield):
#	if len(p) == 3:
#		return fyield - __fit_single_relaxation__(seq_time, *p)
#	else:
#		return fyield - __fit_triple_relaxation__(seq_time, *p)

def __calculate_Webb_model__(E, P, a):
	from numpy import exp

	return P * (1 - exp(-a * E/P))

def __calculate_modified_Webb_model__(E, P, a):
	from numpy import exp
	
	return P * (1 - exp(-a * E/P)) * (E**-1)

def __calculate_rsquared__(res, fyield):
	from numpy import sum, mean
	
	return 1 - (sum(res**2)/sum((fyield - mean(fyield))**2))


def __calculate_bias__(res, fyield):
	from numpy import sum

	return sum((1 - res)/fyield) / (len(fyield)*100)


def __calculate_chisquared__(res, fyield):
	from numpy import sum

	return sum(res**2 / fyield)	


def __calculate_fit_errors__(popt, res):
	from numpy import sqrt, diag, linalg, mean

	J = popt.jac
	pcov = linalg.inv(J.T.dot(J)) * mean(res**2)
	return sqrt(diag(pcov))

