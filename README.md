Phytoplankton Photophysiology Utils
=====================

This is a tool to read and process FIRe and FRRf data from raw format to a level2 QC phase.
For more information see the documentation of the main variable processing functions with the `calc_` prefix.
Below is a short example of how to use the data to read in and process variables.


EXAMPLE USAGE
-------------
This package is meant to be used in an interactive environment - ideally Jupyter Notebook

```python
%pylab inline
import PhytoPhotoUtils as ppu

fname = '/path_to_data/data'
res_path = '/output_path'

# Load all variables needed for fitting saturation and relaxation models
#FIRe
df = ppu.load.load_FIRe_files(fname, append=False, save_files=True, res_path=res_path,
                             seq_len=160, flen=1e-6, sigscale=1e-20, irrad=47248)

#FastTracka I
df = ppu.load.load_FastTrackaI_files(append=False, save_files=True, res_path=res_path, 
                                 seq_len=120, sigscale=1e-20, irrad=545.62e10)

#FastOcean
df = ppu.load.load_FastOcean_files(fname, append=False, save_files=True, led_separate=True, res_path=res_path, 
                       seq_len=140, sigscale=1e-20, flen=1e-6)


# Perform a no p saturation model fit on the data
sat = ppu.saturation.calc_nopmodel(df, blank=10, sat_len=100, skip=1, n_iter=1000)


# Perform a single decay relaxation model fit on the data
rel = ppu.relaxation.calc_single(df, blank=10, sat_len=100, rel_len=60, sat_flashlets=0, n_iter=1000)


# Perform time averaging on raw transients, including the removal of outliers (mean + stdev * 3)
dfm = ppu.tools.outlier_bounds_time_average(df, time=10, mutliplier=3, seq_len=160)


# Correct for FIRe instrument detector bias
dfb = ppu.tools.fire_bias_correction(df, pos=1, sat=True, sat_len=100)

# See the demo file for more info
```


ABOUT
-----
This work was funded by the CSIR and Curtin University.

- Version: 0.1
- Author:  Thomas Ryan-Keogh, Charlotte Robinson
- Email:   tjryankeogh@gmail.com
- Date:    2018-12-06
- Institution: Council for Scientific and Industrial Research, Curtin University
- Research group: Southern Ocean Carbon - Climate Observatory (SOCCO)

Please use the guidlines given on https://integrity.mit.edu/handbook/writing-code to cite this code.

**Example citation:**
Source: phyto_photo_utils [https://gitlab.com/socco/BuoyancyGliderUtils](https://gitlab.com/tjryankeogh/phytophotoutils) retrieved on 18 December 2018.


CHANGE LOG
----------
**v0.3** (2019-05-17)

- restructured package to avoid nested functions
- added outlier removal tool to FLC function

**v0.2** (2018-12-07)

- added functionality for FLCs


PACKAGE STRUCTURE
-----------------
NOTE: This package structure is defined by the `__init__.py` file
- load
	- load_FIRe_files
	- load_FastTrackaI_files
	- load_FastOcean_files
- saturation
	- calc_fixedpmodel
	- calc_pmodel
	- calc_nopmodel
- relaxation
	- calc_single
	- calc_triple
- tools
	- outlier_bounds_time_average
	- fire_bias_correction
	- calc_blank_FastOcean
	_ calc_blank_FIRe
- spectral_correction
	- calc_chl_specific_absorption
	- instrument_led_correction
- flc
	- e_dependent_etr
	- e_independent_etr


ACKNOWLEDGEMENTS
----------------
- 


TO DO
-----
- Add the FLC data processing functions with different options for fitting e_dependent and e_independent models
- Add methods to load in discrete files
- Add blank method
- Add additional methods for newer FIRe instruments
