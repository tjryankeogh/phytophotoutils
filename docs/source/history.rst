Change Log
----------
**v1.4.1**(2020-09-17)
- F\ :sub:`o` and F\ :sub:`m` errors are now reported as percentages.
- Model bias is now normalised.
- Additional statistical output included in the form of normalised RMSE, where RMSE is normalised to the mean fluorescence level.

**v1.4**(2020-07-01)

- Added an option to ETR fitting to apply Serodio corrections for samples that have dark relaxation.
- Added an option for the fit to return E\ :sub:`k` and manually calculate ETR\ :sup:`max`.
- Added an option to fit ETR\ :sup:`max` using a beta model.

**v1.3.5**(2020-05-22)

- Bug fix for ETR fitting when the number of measurements per light level is 1.

**v1.3.4**(2020-05-05)

- Bug fix for fitting with no connectivity model.

**v1.3.3**(2020-04-17)

- Bug fix for handling single FIRe files
- Loading FIRe files can now split single turnover and multiple turnover measurements

**v1.3.2**(2020-03-26)

- Bug fix to tools functions for change in name from fyield to flevel
- Bug fix for spectral correction to sort wavelengths to always be ascending

**v1.3.1**(2020-03-05)

- Added functionality to load raw data files of Single Acquisitions from Chelsea FastAct1 laboratory system
- Added functionality to calculate ETR by only using the last 3 measurements of each time step

**v1.3**(2020-02-20)

- Added functionality to load raw data files from Soliense LIFT-FRR
- Added functionality to load raw data files from Chelsea FastAct2 laboratory systems
- Bug fix to fit triple relaxation

**v1.2.1**(2020-01-08)

- Bug fix for fixed ro calculation

**v1.2.1**(2020-01-08)

- Bug fixes for fitting triple relaxation error codes
- Update to recommend bounds for tau1

**v1.2**(2019-12-09)

- Update to bias calculation.
- Update to spectral correction code for correcting for background light and using chlorophyll to calculate Kd.
- Update to spectral correction code to include FastOcean LED spectra.
- Update to spectral correction code that allows the user to include their own constants/spectra instead of the pre-included file.
- Plot functions now close any existing figure objects.
- Plot functions now include RMSE in the legend.
- Update to remove outliers code to make the datetime array 'datetime64'.
- Statistical metrics returned from FLC fitting procedure no longer include R\ :sup:`2`, Chi\ :sup:`2` or reduced Chi\ :sup:`2`.

**v1.1**(2019-10-15)

- Statistical metrics returned from fitting procedure no longer include R\ :sup:`2`, Chi\ :sup:`2` or reduced Chi\ :sup:`2`.

**v1.0.2**(2019-10-06)

- Saturation flashlets in relaxation fitting are now included in F\ :sub:`m`Relax estimation, rather than replacing relaxation flashlets.
- FIRe instrument relaxation bias now only uses the difference in relaxation flashlets to correct the large difference in flashlets.

**v1.0.1**(2019-10-04)

- Implementation of code for submission to PyPi. Package now available for installation using pip install phyto_photo_utils.

**v1.0** (2019-10-01)

- Syntax changes to saturation, relaxation and flc. Different models now called with optional arguments instead of separate functions.

**v0.9** (2019-10-01)

- Update to phytoplankton specific absorption code for handling phycobilin content
- Update to phytoplankton specific absorption code for updated pathlength amplification coefficients
- Update to phytoplankton specific absorption code for not normalising in the infra-red (750 nm) region

**v0.8** (2019-06-28)

- Bug fix to spectral correction for handling arrays
- Statistical metrics now outputs RMSE, reduced Chi squared
- Processing flags now included in output

**v0.7** (2019-06-20)

- F\ :sub:`o` and F\ :sub:`m` now estimated as intercepts of Huber Regression linear fits
- Fitting skipped if F\ :sub:`o` is greater than F\ :sub:`m`
- Spectral correction now calculates factor as a function of depth

**v0.6** (2019-05-30)

- read the docs formatting applied
- added warning messages when lower bounds are higher than upper bounds
- added demo file

**v0.5** (2019-05-23)

- various bug fixes
- spectral LED correction now estimates in situ light field

**v0.4** (2019-05-21)

- added plot function

**v0.3** (2019-05-17)

- restructured package to avoid nested functions
- added outlier removal tool to FLC function

**v0.2** (2018-12-07)

- added functionality for FLCs

**v0.1** (2018-12-01)

- Functions compiled in package format