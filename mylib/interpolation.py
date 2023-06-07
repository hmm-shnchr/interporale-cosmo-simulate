#from reshape_dataset import ReshapeDataset
from scipy import interpolate as ip
import numpy as np


def interp_(data_input, input_size, output_size, interp_kind):
    interp_input_step   = np.concatenate([np.arange(input_size), np.arange(input_size + output_size, input_size * 2 + output_size)])
    interp_unit_size    = np.arange(input_size * 2 + output_size)
    interp_func         = ip.interp1d(interp_input_step, data_input, kind = interp_kind)
    data_prediction     = interp_func(interp_unit_size)[:, input_size:input_size + output_size]

    return data_prediction


def interp(data_input, sf_input, sf_output, interp_kind):
    data_interpolation  = []
    for idx in range(data_input.shape[0]):
        interp_func     = ip.interp1d(sf_input[idx], data_input[idx], kind = interp_kind)
        data_interpolation.append(interp_func(sf_output[idx]))
    data_interpolation  = np.array(data_interpolation)

    return data_interpolation


def hermite(x, dxdsf, sf_input, sf_output):
    data_interpolation  = []
    for idx in range(x.shape[0]):
        interp_func     = ip.CubicHermiteSpline(sf_input[idx], x[idx], dxdsf[idx])
        data_interpolation.append(interp_func(sf_output[idx]))
    data_interpolation  = np.array(data_interpolation)

    return data_interpolation
