Change Log
----------
**v1.1**(2019-10-15)

- Statistical metrics returned from fitting procedure no longer include R\ :sup:`2`, Chi\ :sup:`2` or reduced Chi\ :sup:`2`.

**v1.0.2**(2019-10-06)

- Saturation flashlets in relaxation fitting are now included in F\ :sub:`m`Relax estimation, rather than replacing relaxation flashlets.
- FIRe instrument relaxation bias now only uses the difference in relaxation flashlets to correct the large difference in flashlets.

**v1.0.1**(2019-10-04)

- Implementation of code for submission to PyPi. Package now available for installation using pip install phyto_photo_utils.

**v1.0**(2019-10-01)

- Syntax changes to saturation, relaxation and flc. Different models now called with optional arguments instead of separate functions.

**v0.9**(2019-09-30)

- Update to phytoplankton specific absorption code for handling phycobilin content
- Update to phytoplankton specific absorption code for updated pathlength amplification coefficients
- Update to phytoplankton specific absorption code for not normalising in the infra-red (750 nm) region

**v0.8**(2019-06-28)

- Bug fix to spectral correction for handling arrays
- Statistical metrics now outputs RMSE, reduced Chi squared
- Processing flags now included in output

**v0.7**(2019-06-20)

- Fo and Fm now estimated as intercepts of Huber Regression linear fits
- Fitting skipped if Fo is greater than Fm
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
