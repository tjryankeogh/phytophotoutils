Phytoplankton Photophysiology Utils
===================================

This is a tool to read and process active chlorophyll fluorescence data from raw format and apply the biophysical model of Kolber et al. (1998).
For more information see the documentation and demo file, below is a short example of how to use the data to read in and process variables.


EXAMPLE USAGE
-------------
This package is meant to be used in an interactive environment - ideally Jupyter Notebook

```python
import phyto_photo_utils as ppu

fname = '/path_to_data/data'
output = '/output_path'

# Load all variables needed for fitting saturation and relaxation models
df = ppu.load_FASTTrackaI_files(fname, append=False, save_files=True, res_path=output, seq_len=120, irrad=545.62e10)

# Perform a ρ saturation model fit on the data
sat = ppu.fit_saturation(pfd, fyield, seq, datetime, blank=0, sat_len=100, skip=0, ro_lims=[0.0,1.0], sig_lims =[100,2200])

# Perform a single decay relaxation model fit on the data
rel = ppu.fit_single(fyield, seq_time, seq, datetime, blank=0, sat_len=100, rel_len=40, single_decay=True, bounds=True, tau_lims=[100, 50000])

# Perform time averaging (5 minute averages) on raw transients, including the removal of outliers (mean + stdev * 3)
dfm = ppu.remove_outlier_from_time_average(df, time=5, multiplier=3)

# Correct for FIRe instrument detector bias
dfb = ppu.correct_fire_instrument_bias(df, sat=False, pos=1, sat_len=100)

# See the demo file for more info
```


ABOUT
-----
This work was funded by the CSIR. This research was partially supported by the Australian Government through the Australian Research Council's Discovery Projects funding scheme (DP160103387).

- Version: 1.3.3
- Author:  Thomas Ryan-Keogh, Charlotte Robinson
- Email:   tjryankeogh@gmail.com
- Date:    2018-12-06
- Institution: Council for Scientific and Industrial Research, Curtin University
- Research group: Southern Ocean Carbon - Climate Observatory (SOCCO), Remote Sensing and Satellite Research Group

Please use the guidlines given on https://integrity.mit.edu/handbook/writing-code to cite this code.

**Example citation:**
Source: phyto_photo_utils [https://gitlab.com/tjryankeogh/phytophotoutils] retrieved on 30 May 2019.

PACKAGE STRUCTURE
-----------------
NOTE: This package structure is defined by the `__init__.py` file
- load
	- load_FIRe_files
	- load_FASTTrackaI_files
	- load_FastOcean_files
	- load_LIFT_FRR_files
- saturation
	- fit_saturation
- relaxation
	- fit_relaxation
- tools
	- remove_outlier_from_time_average
	- correct_fire_instrument_bias
	- calculate_blank_FastOcean
	_ calculate_blank_FIRe
- spectral_correction
	- calculate_chl_specific_absorption
	- calculate_instrument_led_correction
- etr
	- calculate_etr
- plot
	- plot_saturation_data
	- plot_relaxation_data
	- plot_fluorescence_light_curve
- equations
	- __fit_kolber_nop__
	- __calculatate_residual_saturation_nop__
	- __fit_kolber_p__
	- __calculate_residual_saturation_p__
	- __fit_single_relaxation__
	- __calculate_residual_single_relaxation__
	- __fit_triple_relaxation__
	- __calculate_residual_triple_relaxation__
	- __calculate_Webb_model__
	- __calculate_residual_etr__
	- __calculate_modified_Webb_model__
	- __calculate_residual_phi__
	- __calculate_bias__
	- __calculate_rmse__
	- __calculate_fit_errors__
- fitting
	- __fit_fixed_p_model__
	- __fit_calc_p_model__
	- __fit_no_p_model__
	- __fit_single_decay__
	- __fit_triple_decay__


ACKNOWLEDGEMENTS
----------------
- Kevin Oxborough (Chelsea Technology Groups) for linear methods to estimate Fo and Fm


TO DO
-----
- Add in function to read FastTracka I binary files
- Add deconvolution method for triple decay relaxation
- Add the option using different production models in FLC processing
- Add additional load method for CTG LabSTAF

