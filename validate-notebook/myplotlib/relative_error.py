import numpy as np


def relative_error(origin, predict):
    eps = 1e-7

    origin              = origin.reshape(-1)
    predict             = predict.reshape(-1)
    div_mask            = origin == 0.0
    origin[div_mask]    += eps
    predict[div_mask]   += eps
    error               = np.abs((origin - predict) / origin)

    return error
