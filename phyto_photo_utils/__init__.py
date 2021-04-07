#!/usr/bin/env python

from ._load import load_FASTTrackaI_files, load_FastOcean_files, load_FIRe_files, load_LIFT_FRR_files
from ._saturation import fit_saturation
from ._relaxation import fit_relaxation
from ._tools import remove_outlier_from_time_average, correct_fire_instrument_bias, calculate_blank_FastOcean, calculate_blank_FIRe
from ._etr import calculate_etr
from ._spectral_correction import calculate_chl_specific_absorption, calculate_instrument_led_correction
from ._equations import __fit_kolber_p__, __fit_kolber_nop__, __calculate_residual_saturation_p__, __calculate_residual_saturation_nop__, __calculate_residual_saturation_fixedp__, __fit_single_relaxation__, __calculate_residual_single_relaxation__, __fit_triple_relaxation__, __calculate_residual_triple_relaxation__, __calculate_alpha_model__,  __calculate_beta_model__, __calculate_modified_alpha_model__, __calculate_modified_beta_model__, __calculate_residual_etr__, __calculate_residual_phi__, __calculate_residual_beta__, __calculate_residual_mbeta__, __calculate_rmse__, __calculate_nrmse__, __calculate_bias__
from ._fitting import __fit_fixed_p_model__, __fit_calc_p_model__, __fit_no_p_model__, __fit_single_decay__, __fit_triple_decay__
from ._plot import plot_saturation_data, plot_relaxation_data, plot_fluorescence_light_curve
