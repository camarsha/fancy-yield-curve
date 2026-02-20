from .fancy_yield_curve import yield_curve_basic, yield_curve_complete
import numpy as np


def yield_curve(
    e_res,
    width_res,
    beam_fwhm,
    height,
    width_target,
    staggle_const,
    start_energy,
    stop_energy,
    step_size,
):
    e, y = yield_curve_basic(
        e_res,
        width_res,
        beam_fwhm,
        height,
        width_target,
        staggle_const,
        start_energy,
        stop_energy,
        step_size,
    )
    return np.array(e), np.array(y)
