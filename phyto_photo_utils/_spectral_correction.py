#!/usr/bin/env python

from numpy import exp, argmin, abs, nanmin, nanmax, nansum, array
from math import pi
from pandas import read_csv

def calculate_chl_specific_absorption(aptot, blank, ap_lambda, depig=None, chl=None, vol=None, beta=None, diam=None, bricaud_slope=True):
	"""

	Process the raw absorbance data to produce chlorophyll specific phytoplankton absorption.

	Parameters
	----------

	aptot : np.array, dtype=float, shape=[n,]
		The raw absorbance data.
	blank : np.array, dtype=float, shape=[n,]
	 	The blank absorbance data.
	ap_lambda : np.array, dtype=float, shape=[n,]
		The wavelengths corresponding to the measurements.
	depig : np.array, dtype=float, shape=[n,] 
		The raw depigmented absorbance data.
	chl : float
		The chlorophyll concentration associated with the measurement.
	vol : int
		The volume of water filtered in mL.
	beta : int
		The pathlength amplification factor (see Roesler et al. 1998).
	diam : float			
		The diameter of filtrate in mm.
	bricaud_slope: bool, default=True
	 	If True, will theoretically calculate detrital slope (see Bricaud & Stramski 1990). If False, will subtract depigmented absorption from total absorption.



	Returns
	-------

	aphy : np.array, dtype=float, shape=[n,]
		The chlorophyll specific phytoplankton absorption data.

	Example
	-------
	>>> aphy = ppu.calculate_chl_specific_absorption(pa_data, blank, wavelength, chl=0.19, vol=2000, beta=2, diam=15, bricaud_slope=True)
	   
	"""

	aptot -= blank # subtract blank from data

	# Convert from absorbance to absorption
	diam = pi * ((diam / 2000)**2) 
	divol = beta * ((vol / 1e6) / diam) 
	aptot *= 2.303 # conversion from log10 to loge
	aptot /= divol

	# Normalise to minimum value (~750 nm)
	dmin = nanmin(aptot)
	aptot -= dmin

	if bricaud_slope: # See Bricaud & Stramski 1990 for more details
	    
	    # Find unique indices based upon wavelengths measured
	    idx380 = abs(ap_lambda - 380).idxmin()
	    idx505 = abs(ap_lambda - 505).idxmin()
	    idx580 = abs(ap_lambda - 580).idxmin()
	    idx692 = abs(ap_lambda - 692).idxmin()
	    idx750 = abs(ap_lambda - 750).idxmin()
	    
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
	    depmin = nanmin(depig)
	    depig -= depmin
	    
	    # Calculate phytoplankton specific absorption
	    aphy = aptot - depig

	aphy /= chl

	return aphy

def calculate_instrument_led_correction(aphy, ap_lambda, e_insitu=None, depth=None, e_led=None):
	"""

	Calculate the spectral correction factor
	TO DO: Create method to convert all arrays to same length and resolution

	Parameters
	----------

	aphy : np.array, dtype=float, shape=[n,]
		The chlorophyll specific phytoplankton absorption.
	ap_lambda : np.array, dtype=int, shape=[n,]
		The wavelengths associated with the aphy.
	e_insitu : np.array, dtype=int, shape=[n,]
		The in situ irradiance field, if None is passed then will theoretically calculated in situ light field.
	depth : float
		The depth of the corresponding aphy measurement.
	e_led : 'fire','fasttracka_ii'
		The excitation spectra of the instrument.

	Returns
	-------

	scf : float
		The spectral correction factor.

	Example
	-------
	>>> ppu.calculate_instrument_led_correction(aphy, wavelength, e_led='fire')
	   
	"""

	idx400 = abs(ap_lambda - 400).idxmin()
	idx700 = abs(ap_lambda - 700).idxmin()+1
	aphy = array(aphy[idx400:idx700])


	if e_insitu is None:
		if depth is None:
			print('User must define depth for calculating in situ light spectra')
		else:
			df = read_csv('./data/output/spectral_correction_factors/spectral_correction_constants.csv', index_col=0)
			kd = df.a_W + df.a_gt + aphy
			Ez = df.Ezero * exp(-kd * depth)
			Erange = Ez.max() - Ez.min()
			e_insitu = (Ez - Ez.min())/Erange

	else:
		e_insitu = e_insitu / max(e_insitu) 

	if e_led is None:
		print('No instrument selected. Unable to calculate spectral correction factor.')

	elif e_led == 'fire':
		 e_led = df.fire.values
		 e_led = e_led / nanmax(e_led)
	elif e_led == 'fasttracka_ii':
		 e_led = df.fasttracka_ii.values
		 e_led = e_led / nanmax(e_led)

	aphy = aphy / nanmax(aphy)	

	# Perform SCF calculation
	scf = (nansum(aphy * e_insitu) * nansum(e_led)) / (nansum(aphy * e_led) * nansum(e_insitu))

	return scf

