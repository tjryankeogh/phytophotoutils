Fluorescence Light Curves
=========================

Convert the processed transient data into an electron transport rate and perform a fit using the Webb Model.

Parameters
----------
fo : np.array, dtype=float, shape=[n,]
	The minimum fluorescence yield.
fm : np.array, dtype=float, shape=[n,] 
	The maximum fluorescence yield.
sigma : np.array, dtype=float, shape=[n,] 
	The effective absorption cross-section of PSII in Å\ :sup:`2`.
par : np.array, dtype=float, shape=[n,]
	The actinic light levels in μE m\ :sup:`2` s\ :sup:`-1`.
light_independent : bool, default=True
	If True, will use the method outlined in Silsbe & Kromkamp 2012. 
dark_sigma : bool
	If True, will use mean of σ\ :sub:`PSII` under 0 actinic light for calculation. If False, will use σ\ :sub:`PSII` and σ\ :sub:`PSII`' for calculation.
light_step_size : int
	The number of measurements for initial light step.
outlier_multiplier : int, default=3
	The multiplier to apply to the standard deviation for determining the upper and lower limits.
return_data : bool, default=False
	If True, will return the final data used for the fit.
bounds : bool, default=True
	If True, will set lower and upper limit bounds for the estimation, not suitable for methods 'lm'.
alpha_lims : [int, int], default=[0,4]
	The lower and upper limit bounds for fitting α\ :sup:`ETR`.
etrmax_lims : [int, int], default=[0,2000]
 	The lower and upper limit bounds for fitting ETR\ :sub:`max`.
fit_method : str, default='trf'
	The algorithm to perform minimization. See ``scipy.optimize.least_squares`` documentation for more information on non-linear least squares fitting options.
loss_method : str, default='soft_l1'
	The loss function to be used. Note: Method ‘lm’ supports only ‘linear’ loss. See ``scipy.optimize.least_squares`` documentation for more information on non-linear least squares fitting options.
fscale : float, default=0.1
 	The soft margin value between inlier and outlier residuals. See ``scipy.optimize.least_squares`` documentation for more information on non-linear least squares fitting options.
max_nfev : int, default=None		
	The number of iterations to perform fitting routine. If None, the value is chosen automatically. See ``scipy.optimize.least_squares`` documentation for more information on non-linear least squares fitting options.
xtol : float, default=1e-9			
	The tolerance for termination by the change of the independent variables. See ``scipy.optimize.least_squares`` documentation for more information on non-linear least squares fitting options.

Returns
-------

etr_max : float
	The maximum electron transport rate.
alpha : float
	The light limited slope of electron transport.
ek : float
	The photoacclimation of ETR.
rsq : np.array, dtype=float, shape=[n,]
	The r\ :sup:`2` value.
bias : np.array, dtype=float, shape=[n,]
	The bias of fit.
chi : np.array, dtype=float, shape=[n,]
	The chi-squared goodness of fit.
rchi : np.array, dtype=float, shape=[n,]
	The reduced chi-squared goodness of fit.
rmse : np.array, dtype=float, shape=[n,]
	The root mean squared error of the fit.
etrmax_err : float
	The fit error of ETR\ :sup:`max`.
alpha_err : float
	The fit error of α\ :sub:`ETR`.
data : [np.array, np.array]
	Optional, the final data used for the fitting procedure.


Example
-------
>>> etr_max, alpha, ek, rsq, bias, chi, etr_max_err, alpha_err = ppu.calculate_e_dependent_etr(fo, fm, fvfm, sigma, par, return_data=False)

