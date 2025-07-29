use crate::gaussian::Gaussian;
use std::collections::VecDeque;
use std::f64::consts::PI;

const FWHM_TO_STD: f64 = 2.3548200450309493;

#[inline]
pub fn fwhm_to_std(fwhm: f64) -> f64 {
    fwhm / FWHM_TO_STD
}

/// Analytic function for a thick target yield curve.
/// Eq. 4.102 in Iliadis
/// Note that this equation can be in the COM or Lab frame depending
/// on the frame of the resonance energy, target thickness, and resonance width.
/// The height is equivalent to the factor (lambda^2/2pi * wg/eta_r)
#[inline]
pub fn thick_target_yield(
    energy: f64,
    e_res: f64,
    width_res: f64,
    width_target: f64,
    height: f64,
) -> f64 {
    let half_width = width_res / 2.0;
    (height / PI)
        * (f64::atan((energy - e_res) / half_width)
            - f64::atan((energy - e_res - width_target) / half_width))
}

/// Given a yield, apply a Gaussian spread to the point.
/// Beam energy spread is given in fwhm
pub fn beam_spread_point(
    energy: f64,
    beam_fwhm: f64,
    energy_grid: &[f64],
    yield_grid: &[f64],
) -> f64 {
    let g = Gaussian::new(0.0, beam_fwhm / FWHM_TO_STD);
    // Starting with +/- 10 sigma, the old code seemed excessive at +/- 30.
    let delta = 5.0 * g.std;
    let start = energy - delta;
    let stop = energy + delta;
    let mut result = 0.0;
    let mut psum = 0.0; //normalization

    for (&current, &y) in energy_grid.iter().zip(yield_grid.iter()) {
        if current < start {
            continue;
        }

        let p = g.pdf(energy - current);
        psum += p;
        result += p * y;
        if current > stop {
            break;
        }
    }

    result / psum
}

/// This function applies both the target and beam straggling effects to the yield.
/// If the beam energy is below the resonance energy it returns does not
/// have an effect on the yield function.
pub fn simple_straggle_point(
    energy: f64,
    e_res: f64,
    straggle_const: f64,
    energy_grid: &[f64],
    yield_grid: &[f64],
) -> f64 {
    if energy <= e_res {
        let idx = energy_grid.iter().position(|&e| e == energy).unwrap();
        return yield_grid[idx];
    }
    // The straggling changes depending on how deep in the target you are.
    let std = f64::sqrt(straggle_const * (energy - e_res)) / FWHM_TO_STD;
    let g = Gaussian::new(0.0, std);
    // Copying the old code beat for beat, so we default to 5 sigma for the full profile
    let delta = 5.0 * g.std;
    let start = energy - delta;
    let stop = energy + delta;
    let mut result = 0.0;
    let mut psum = 0.0; //normalization

    for (&current, &y) in energy_grid.iter().zip(yield_grid.iter()) {
        if current < start {
            continue;
        }
        let p = g.pdf(energy - current);
        psum += p;
        result += p * y;
        if current > stop {
            break;
        }
    }

    result / psum
}

/// This function applies both the target and beam straggling effects to the yield.
/// For a continuous cross section, the current energy must be less than the
/// beam energy for straggling to apply (i.e inside the target).
pub fn non_resonant_straggle(
    energy: f64,
    e_beam: f64,
    straggle_const: f64,
    energy_grid: &[f64],
    yield_grid: &[f64],
) -> f64 {
    if energy >= e_beam {
        let idx = energy_grid.iter().position(|&e| e == energy).unwrap();
        return yield_grid[idx];
    }
    // The straggling changes depending on how deep in the target you are.
    let std = f64::sqrt(straggle_const * (e_beam - energy)) / FWHM_TO_STD;
    let g = Gaussian::new(0.0, std);
    // Copying the old code beat for beat, so we default to 5 sigma for the full profile
    let delta = 5.0 * g.std;
    let start = energy - delta;
    let stop = energy + delta;
    let mut result = 0.0;
    let mut psum = 0.0; //normalization

    for (&current, &y) in energy_grid.iter().zip(yield_grid.iter()) {
        if current < start {
            continue;
        }
        let p = g.pdf(energy - current);
        psum += p;
        result += p * y;
        if current > stop {
            break;
        }
    }

    result / psum
}
