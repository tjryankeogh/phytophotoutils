#!/usr/bin/env python

from numpy import exp, argmin, abs, nanmin, nanmax, nansum, array
from math import pi
from pandas import read_csv

def calculate_chl_specific_absorption(aptot, blank, ap_lambda, depig=None, chl=None, vol=None, betac1=None, betac2=None, diam=None, bricaud_slope=False, phycobilin=False, norm_750=False):
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
	betac1 : int
		The pathlength amplification coefficient 1 (see Stramski et al. 2015). For transmittance mode, 0.679, for transmittance-reflectance mode, 0.719, and for integrating sphere mode, 0.323.
	betac2 : int
		The pathlength amplification coefficient 2 (see Stramski et al. 2015). For transmittance mode, 1.2804, for transmittance-reflectance mode, 1.2287, and for integrating sphere mode, 1.0867.
	diam : float			
		The diameter of filtrate in mm.
	bricaud_slope: bool, default=True
	 	If True, will theoretically calculate detrital slope (see Bricaud & Stramski 1990). If False, will subtract depigmented absorption from total absorption.
	phycobilin: bool, default=False
		If True, will account for high absorption in the green wavelengths (580 - 600 nm) by phycobilin proteins when performing the bricaud_slope detrital correction.
	norm_750: bool, default=False
		If True, will normalise the data to the value at 750 nm.



	Returns
	-------

	aphy : np.array, dtype=float, shape=[n,]
		The chlorophyll specific phytoplankton absorption data.

	Example
	-------
	>>> aphy = ppu.calculate_chl_specific_absorption(pa_data, blank, wavelength, chl=0.19, vol=2000, betac1=0.323, betac2=1.0867, diam=15, bricaud_slope=True, phycobilin=True, norm750=False)
	   
	"""
	aptot = array(aptot)
	ap_lambda = array(ap_lambda)

	aptot -= blank # subtract blank from data

	# Convert from absorbance to absorption
	diam = pi * ((diam / 2000)**2) 
	divol = ((vol / 1e6) / diam) 
	aptot *= 2.303 # conversion from log10 to loge
	aptot /= divol
	aptot = betac1 * aptot**betac2

	# Normalise to minimum value (~750 nm)
	if norm_750:
		dmin = argmin(abs(ap_lambda - 750))
		dmin = aptot[dmin]
		aptot -= dmin

	if bricaud_slope: # See Bricaud & Stramski 1990 for more details
	    
	    # Find unique indices based upon wavelengths measured
		idx380 = argmin(abs(ap_lambda - 380))
		idx505 = argmin(abs(ap_lambda - 505))
		idx580 = argmin(abs(ap_lambda - 580))
		idx600 = argmin(abs(ap_lambda - 600))
		idx692 = argmin(abs(ap_lambda - 692))
		idx750 = argmin(abs(ap_lambda - 750))
	    
		ap750 = aptot[idx750]
		R1 = (0.99 * aptot[idx380]) - aptot[idx505]
	    
		if phycobilin:

			phyco = aptot[idx580:idx600]
			ratio = []

			for i in range(len(phyco)):

				r = phyco.values[i] / aptot[idx692]
				ratio.append(r)

			val, idx = min((val, idx) for (idx, val) in enumerate(ratio))
			idx = phyco.index[idx]
			widx = ap_lambda[idx]
			R2 = (aptot[idx]) - 0.92 * aptot[idx692]

		else:
			R2 = (aptot[idx580]) - 0.92 * aptot[idx692]

		if (R1 <= 0) | (R2 <= 0):
			S = 0

		else:
			R = R1/R2
			S = 0.0001
			L1 = (0.99 * exp(-380 * S)) - exp(-505 * S)

			if phycobilin:
				L2 = exp(-widx * S) - 0.92 * exp(-widx * S)
			else:
				L2 = exp(-580 * S) - 0.92 * exp(-692 * S)
			L = L1/L2

			while (S < 0.03):
				S += 0.0001
				L = (0.99 * exp(-380 * S) - 0.99 * exp(-505 * S)) 
				if phycobilin:
					L = L / (exp(-widx * S) - 0.92 * exp(-692 * S))
				else:
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
		if depig is None:
			print('UserError - no depigmented data provided.')
		# Convert from absorbance to absorption
		depig -= blank
		depig *= 2.303 # Conversion from log10 to loge
		depig /= divol

 		# Normalise to minimum value (~750 nm)
		if norm_750:
			
			depmin = argmin(abs(ap_lambda - 750))
			depmin = depig[depmin]
			depig -= depmin
	    
	    # Calculate phytoplankton specific absorption
		aphy = aptot - depig
	
	if chl is None:

		return aphy
	
	else:
		aphy /= chl

		return aphy
	

def calculate_instrument_led_correction(aphy, ap_lambda, method=None, chl=None, e_background=None, e_insitu=None, e_actinic=None, depth=None, e_led=None, wl=None, constants=None):
	"""

	Calculate the spectral correction factor
	TO DO: Create method to convert all arrays to same length and resolution
	TO DO: Add in functionality to calculate mixed excitation wavelength spectra for FastOcean when using 1+ wavelength

	Parameters
	----------

	aphy : np.array, dtype=float, shape=[n,]
		The wavelength specific phytoplankton absorption coefficients.
	ap_lambda : np.array, dtype=int, shape=[n,]
		The wavelengths associated with the aphy and aphy_star.
	method : 'sigma', 'actinic'
		Choose spectral correction method to either correct SigmaPSII or correct the background actinic light.
	e_background : 'insitu', 'actinic'
		For sigma spectral correction factor, select either insitu light (e.g. for underway or insitu measurements) or actinic light (e.g. for fluorescence light curves) as the background light source
	e_insitu : np.array, dtype=int, shape=[n,]
		The in situ irradiance field, if None is passed then will theoretically calculate in situ light field.
	chl : dtype=float
		Chlorophyll concentration for estimation of Kbio for theoretical in situ light field. If None is passed then chl is set to 1 mg/m3.
	e_actinic : 'fastact'
		Actinic light spectrum e.g. Spectra of the Actinic lights within the FastAct illuminating during Fluorescence Light Curves etc. Must defined for 'actinic' method.
	depth : float, default=None
		The depth of the measurement. Must be set if theoretically calculating e_insitu.
	e_led : 'fire','fasttracka_ii', 'fastocean'
		The excitation spectra of the instrument.
	wl : '450nm', '530nm', 624nm', None
		For FastOcean only. Select the excitation wavelength. Future PPU versions will provide option to mix LEDs.

	Returns
	-------

	scf : float
		The spectral correction factor to correct SigmaPSII or actinic background light depending on method.

	Example
	-------
	>>> ppu.calculate_instrument_led_correction(aphy, wavelength, e_led='fire')

	"""
	aphy = array(aphy)
	ap_lambda = array(ap_lambda)

	idx400 = argmin(abs(ap_lambda - 400))
	idx700 = argmin(abs(ap_lambda - 700))+1
	aphy = array(aphy[idx400:idx700])
	aphy = aphy / nanmax(aphy)

	if method is None:
		print('User must select spectral correction method for correcting sigma or correcting actinic light')

	if constants is None:
		df = read_csv('./data/output/spectral_correction_factors/spectral_correction_constants.csv', index_col=0)
		df = df.sort_index()
	else:
		df = read_csv(constants, index_col=0)
		df = df.sort_index()

	if method == 'sigma':

		if chl is None:
			print(r'Chlorophyll concentration set to 1 mg m$^-$$^3$')
			chl = 1
		else:
			chl = chl

		if e_background == 'insitu':

			if e_insitu is None:
				if depth is None:
					print('User must define depth for calculating in situ light spectra')
				else:
					kd = df.a_W + df.bb_W + (df.chi * chl ** df.e)
					Ez = df.Ezero * exp(-kd * depth)
					e_background = Ez / max(Ez)

			else:
				e_background = e_insitu / max(e_insitu)

		elif e_background == 'actinic':

			if e_actinic == 'fastact':
				e_actinic = df.fastact.values
				e_background = e_actinic / max(e_actinic)
			else:
				e_background = e_actinic / max(e_actinic)

		if e_led is None:
			print('No instrument selected. Unable to calculate sigma spectral correction factor.')
		elif e_led == 'fire':
		 	e_led = df.fire.values
		 	e_led = e_led / nanmax(e_led)
		elif e_led == 'fasttracka_ii':
		 	e_led = df.fasttracka_ii.values
		 	e_led = e_led / nanmax(e_led)
		elif e_led == 'fastocean':
		 	if wl == None:
		 		print('User must select single excitation wavelength for FastOcean')
		 	elif wl == '450nm':
		 		e_led = df.fastocean_450.values
		 	elif wl == '530nm':
		 		e_led = df.fastocean_530.values
		 	elif wl == '624nm':
		 		e_led = df.fastocean_624.values

		# Perform SCF calculation for sigma
		scf = (nansum(aphy * e_background) * nansum(e_led)) / (nansum(aphy * e_led) * nansum(e_background))

	elif method == 'actinic':
		if e_insitu is None:
			if depth is None:
				print('User must define depth for calculating in situ light spectra')
			else:
				kd = df.a_W + df.bb_W + (df.chi * chl ** df.e)
				Ez = df.Ezero * exp(-kd * depth)
				e_insitu = Ez / max(Ez)
		else:
			e_insitu = e_insitu / max(e_insitu)

		if e_actinic == 'fastact':
			e_actinic = df.fastact.values
			e_actinic = e_actinic / max(e_actinic)
		else:
			e_actinic = e_actinic / max(e_actinic)

		# Perform SCF calculation for the actinic light

		scf = (nansum(aphy * e_actinic) * nansum(e_insitu)) / (nansum(aphy * e_insitu) * nansum(e_actinic))


	return scf


