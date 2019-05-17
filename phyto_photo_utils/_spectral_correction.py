#!/usr/bin/env python
"""
@package phyto_photo_utils.spectral_correction
@file phyto_photo_utils/spectral_correction.py
@author Thomas Ryan-Keogh
@brief module containing the processing functions reading raw absorption data and calculating spectral correction

NOTE: Standard manufacturer LED spectra are used for determining the spectral correction.

"""

def calc_chl_specific_absorption(aptot, depig, blank, ap_lambda, chl=1.0, vol=2000, beta=2, diam=15.0, bricaud_slope=True):
	"""

	Process the raw absorbance data to produce chlorophyll specific phytoplankton absorption.

	Parameters
	----------

	aptot: numpy.ndarray
		The raw absorbance data.
	depig: numpy.ndarray 
		The raw depigmented absorbance data.
	blank: numpy.ndarray
	 	The blank absorbance data.
	ap_lambda: np.ndarray
		The wavelengths corresponding to measurements.
	chl: float
		The chlorophyll concentration associated with the measurement.
	vol: numpy.ndarray
		The volume of water filtered in mL.
	beta: int
		The pathlength amplification factor (see Roesler et al. 1998).
	diam: float			
		The diameter of filtrate in mm.
	bricaud_slope: bool
	 	If True, will theoretically calculate detrital slope (see Bricaud & Stramski 1990). If False, will subtract depigmented absorption from total absorption.



	Returns
	-------

	aphy: numpy.ndarray
		The chlorophyll specific phytoplankton absorption data.
	   
	"""
	from numpy import exp, argmin, abs, min
	from math import pi
    
	aptot -= blank # subtract blank from data

	# Convert from absorbance to absorption
	diam = pi * ((diam / 2000)**2) 
	divol = beta * ((vol / 1e6) / diam) 
	aptot *= 2.303 # conversion from log10 to loge
	aptot /= divol

	# Normalise to minimum value (~750 nm)
	dmin = min(aptot)
	aptot -= dmin

	if bricaud_slope: # See Bricaud & Stramski 1990 for more details
	    
	    # Find unique indices based upon wavelengths measured
	    idx380 = argmin(abs(ap_lambda - 380))
	    idx505 = argmin(abs(ap_lambda - 505))
	    idx580 = argmin(abs(ap_lambda - 580))
	    idx692 = argmin(abs(ap_lambda - 692))
	    idx750 = argmin(abs(ap_lambda - 750))
	    
	    ap750 = aptot[idx750]
	    R1 = (0.99 * aptot[idx380]) - aptot[idx505]
	    R2 = (aptot[idx580]) - 0.92 * aptot[idx692]

	    if (R1 <= 0) | (R2 <= 0):
	        S = 0

	    else:
	        R = R1/R2
	        S = 0.0001
	        L1 = (0.99 * exp(-380 * S)) - exp(-505 * S)
	        L2 = exp(-580 * S) - 0.92 * exp(-692 * S)
	        L = L1/L2

	        while (S < 0.03):
	                S += 0.0001
	                L = (0.99 * exp(-380 * S) - 0.99 * exp(-505 * S)) 
	                L = L / (exp(-580 * S) - 0.92 * exp(-692 * S))
	                if (L/R) >= 1:
	                    break
	        else: 
	            S = 0
	    if (S == 0 or S == 0.03):
	        A = 0 
	    else:
	        A = (0.99 * aptot[idx380] - aptot[idx505]) / (0.99 * exp(-380 * S) - exp(-505 * S)) 

	    slope = A*exp(-S*ap_lambda) - A*exp(-750*S)

	    # Calculate phytoplankton specific absorption
	    aphy = aptot - slope

	else:

		# Convert from absorbance to absorption
	    depig -= blank
	    depig *= 2.303 # Conversion from log10 to loge
	    depig /= divol

	    # Normalise to minimum value (~750 nm)
	    depmin = min(depig)
	    depig -= depmin
	    
	    # Calculate phytoplankton specific absorption
	    aphy = aptot - depig

	aphy /= chl

	return aphy

def instrument_led_correction(e_insitu, e_led, aphy):
	"""

	Calculate the spectral correction factor
	NOTE: All arrays must be the same length and resolution for calculation

	Parameters
	----------

	e_led: numpy.ndarray 
		The excitation spectra of the instrument.
	e_insitu: numpy.ndarray 
		The in situ irradiance field.
	aphy: numpy.ndarray
		The chlorophyll specific phytoplankton absorption.

	Returns
	-------

	scf: float
		The spectral correction factor.
	   
	"""
	from numpy import max, sum

	# Normalise spectra
	e_insitu = e_insitu / max(e_insitu) 
	e_led = e_led / max(e_led)
	aphy = aphy / max(aphy)

	# Perform SCF calculation
	scf = (sum(aphy * e_insitu) * sum(e_led)) / (sum(aphy * e_led) * sum(e_insitu))

	return scf

