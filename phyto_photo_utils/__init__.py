#!/usr/bin/env python

from ._load import load_FASTTrackaI_files, load_FastOcean_files, load_FIRe_files
from ._saturation import calculate_saturation_with_fixedpmodel, calculate_saturation_with_pmodel, calculate_saturation_with_nopmodel
from ._relaxation import calculate_single_relaxation, calculate_triple_relaxation
from ._tools import remove_outlier_from_time_average, correct_fire_bias_correction, calculate_blank_FastOcean, calculate_blank_FIRe
from ._flc import calculate_e_dependent_etr, calculate_e_independent_etr
from ._spectral_correction import calculate_chl_specific_absorption, calculate_instrument_led_correction
from ._equations import __fit_kolber__, __calculate_residual_saturation__, __fit_single_relaxation__, __calculate_residual_single_relaxation__, __fit_triple_relaxation__, __calculate_residual_triple_relaxation__, __calculate_Webb_model__, __calculate_residual_etr__, __calculate_modified_Webb_model__, __calculate_residual_phi__
from ._fitting import __fit_fixed_p_model__, __fit_calc_p_model__, __fit_no_p_model__, __fit_single_decay__, __fit_triple_decay__
from ._plot import plot_saturation_data, plot_relaxation_data, plot_fluorescence_light_curve
