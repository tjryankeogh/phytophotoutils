Change Log
----------

**v0.4.2** (2018-11-12)

- cleaning functions now seperated into seperate script
- plotting made more robust with catches if x is not psuedo-discrete
- blob outlier detection added

**v0.4.1** (2018-10-31)

- `seperate_spikes` has been moved to tools and has been renamed to `despike`
- quenching correction has been fixed
- `bin_depths` renamed to `grid_data` and input is now `x, y, var` instead of `var, x, y`
- `neighbourhood_iqr` has been deprecated! It wasn't doing anything
- fixed despiking and quenching reports (figures)

**v0.4** (2018-10-19)

- Added `calc_oxygen` module for both sbe and aanderaa oxygen optodes
- Added 3D plotting function `plot.section3D`
- `optics.seperate_spikes` added to calc_physics
- Updated `optics.find_bad_profiles` to either use deep mean or median
- User control on existing functions more explicit and robust


**v0.3.4** (2018-08-31)

- Photic depth function has been updated and is now more robust
- Package now available on pypi


**v0.3** (2018-07-23)

- Added calibration module that lets users calibrate gliders from bottle data
- A new density function that calculates density from corrected salinity and uses surface pressure as a reference pressure
- seaglider module now has a `load_basestation_netCDF_files` function that makes loading data more explicit.
- updated the demo file to include the new functions.

**v0.2** (2018-07-13)

- fixed bug when correcting for quenching (`index[0] does not exist` reported by Tommy)
- made an MLD function that works on ungridded data that returns output as a mask or depths
- `plot.bin_size` now accounts for nans that slocum glider data has
- quenching correction can now be calculated without PAR as input
- only one backscatter function - now the wavelength needs to be set
- section plots are rasterized
- deprication warning (`DataFrame.from_items`) in `optics.sunrise_sunset`
- Slocum output from GEOMAR MATLAB scripts can be imported with `slocum.load_geomar_slocum_matfile`
- `SeaGliderDataset.load_multiple_vars` returns the DataFrame if one dimension otherwise a dictionary of DataFrames
- Added a function `tools.mask_to_depth` to convert boolean layers to depths as a series indexed by dive number
