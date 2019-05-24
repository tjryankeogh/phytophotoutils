Phytoplankton Photophysiology Utils
===================================

This is a tool to read and process FIRe and FRRf data from raw format to a level2 QC phase.
For more information see the documentation, below is a short example of how to use the data to read in and process variables.


EXAMPLE USAGE
-------------
This package is meant to be used in an interactive environment - ideally Jupyter Notebook

```python
%pylab inline
import PhytoPhotoUtils as ppu

fname = '/path_to_data/data'
output = '/output_path'

# Load all variables needed for fitting saturation and relaxation models
df = ppu.load_FASTTrackaI_files(fname, append=False, save_files=True, res_path=output, seq_len=120, irrad=545.62e10)

# Perform a no ρ saturation model fit on the data
sat = ppu.calculate_saturation_with_nopmodel(df, blank=10, sat_len=100, skip=1, n_iter=1000)

# Perform a single decay relaxation model fit on the data
rel = ppu.calculate_single_relaxation(df, blank=10, sat_len=100, rel_len=60, sat_flashlets=0, n_iter=1000)

# Perform time averaging on raw transients, including the removal of outliers (mean + stdev * 3)
dfm = ppu.remove_outlier_from_time_average(df, time=2, multiplier=3)

# Correct for FIRe instrument detector bias
dfb = ppu.correct_fire_bias_correction(file, sat=False, pos=1, sat_len=100)

# See the demo file for more info
```


ABOUT
-----
This work was funded by the CSIR and Curtin University.

- Version: 0.5
- Author:  Thomas Ryan-Keogh, Charlotte Robinson
- Email:   tjryankeogh@gmail.com
- Date:    2018-12-06
- Institution: Council for Scientific and Industrial Research, Curtin University
- Research group: Southern Ocean Carbon - Climate Observatory (SOCCO)

Please use the guidlines given on https://integrity.mit.edu/handbook/writing-code to cite this code.

**Example citation:**
Source: phyto_photo_utils [https://gitlab.com/socco/BuoyancyGliderUtils](https://gitlab.com/tjryankeogh/phytophotoutils) retrieved on 18 December 2018.

PACKAGE STRUCTURE
-----------------
NOTE: This package structure is defined by the `__init__.py` file
- load
	- load_FIRe_files
	- load_FastTrackaI_files
	- load_FastOcean_files
- saturation
	- calculate_saturation_with_fixedpmodel
	- calculate_saturation_with_pmodel
	- calculate_saturation_with_nopmodel
- relaxation
	- calculate_single_relaxation
	- calculate_triple_relaxation
- tools
	- remove_outlier_from_time_average
	- correct_fire_bias_correction
	- calculate_blank_FastOcean
	_ calculate_blank_FIRe
- spectral_correction
	- calculate_chl_specific_absorption
	- calculate_instrument_led_correction
- flc
	- calculate_e_dependent_etr
	- calculate_e_independent_etr
- plot
	- plot_saturation_data
	- plot_relaxation_data
	- plot_fluorescence_light_curve
- equations
	- fit_kolber
	- fit_single_relaxation
	- fit_triple_relaxation
	- calculate_Webb_model
	- calculate_modified_Webb_model
	- calculate_rsquared
	- calculate_bias
	- calculate_chisquared
	- calculate_fit_errors
- fitting
	- fit_fixed_p_model
	- fit_calc_p_model
	- fit_no_p_model
	- fit_single_decay
	- fit_triple_decay


ACKNOWLEDGEMENTS
----------------
- 


TO DO
-----
- Add the option using different production models in FLC processing
- Add methods to load in discrete FIRe files
- Add additional methods for newer FIRe instruments
- Unnest residual functions from _fitting.py
- Update docstrings to match read_the_docs requirements
- Publish documentation on read_the_docs
