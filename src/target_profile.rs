/*
This module allows for different target distribution functions
 */

use crate::gaussian::Gaussian;
use crate::simple_yield::fwhm_to_std;

pub fn uniform_target(energy: f64, e_beam: f64, width_target: f64) -> f64 {
    if (energy >= e_beam) || (energy <= (e_beam - width_target)) {
        0.0
    } else {
        1.0 / width_target
    }
}



