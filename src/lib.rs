use cross_section::CrossSection;
use pyo3::prelude::*;
use rayon::prelude::*;
mod cross_section;
mod gaussian;
mod simple_yield;
mod target_profile;

#[pyfunction]
fn yield_curve(
    e_res: f64,
    width_res: f64,
    beam_fwhm: f64,
    height: f64,
    width_target: f64,
    straggle_const: f64,
    start_energy: f64,
    stop_energy: f64,
    step_size: f64,
) -> (Vec<f64>, Vec<f64>) {
    let num_points = (stop_energy - start_energy) / step_size;
    // create the requested energy grid
    let energy_grid: Vec<f64> = (0..num_points as usize)
        .map(|i| {
            start_energy
                + (stop_energy - start_energy) * i as f64 / (num_points as usize - 1) as f64
        })
        .collect();

    // First we generate the simple yield curve
    let yield_grid: Vec<f64> = energy_grid
        .par_iter()
        .map(|&energy| {
            simple_yield::thick_target_yield(energy, e_res, width_res, width_target, height)
        })
        .collect();

    // Next apply the beam spread function
    let beam_spread: Vec<f64> = energy_grid
        .par_iter()
        .map(|&energy| {
            simple_yield::beam_spread_point(energy, beam_fwhm, &energy_grid, &yield_grid)
        })
        .collect();

    let beam_straggle: Vec<f64> = energy_grid
        .par_iter()
        .map(|&energy| {
            simple_yield::simple_straggle_point(
                energy,
                e_res,
                straggle_const,
                &energy_grid,
                &beam_spread,
            )
        })
        .collect();

    let target_straggle: Vec<f64> = energy_grid
        .par_iter()
        .map(|&energy| {
            simple_yield::simple_straggle_point(
                energy,
                e_res,
                straggle_const,
                &energy_grid,
                &beam_straggle,
            )
        })
        .collect();

    (energy_grid, target_straggle)
}
#[pyfunction]
fn calc_direct_capture_yield(
    e_beam: f64,
    beam_fwhm: f64,
    det_fwhm: f64,
    height: f64,
    width_target: f64,
    straggle_const: f64,
    cs_energies: Vec<f64>,
    cs_values: Vec<f64>,
    start_energy: f64,
    stop_energy: f64,
    step_size: f64,
) -> (Vec<f64>, Vec<f64>) {
    let mut cross_section = CrossSection::new(cs_energies, cs_values, e_beam);
    let num_points = (stop_energy - start_energy) / step_size;
    // create the requested energy grid
    let energy_grid: Vec<f64> = (0..num_points as usize)
        .map(|i| {
            start_energy
                + (stop_energy - start_energy) * i as f64 / (num_points as usize - 1) as f64
        })
        .collect();

    // we start with a uniform target profile
    let target_grid: Vec<f64> = energy_grid
        .iter()
        .map(|&energy| target_profile::uniform_target(energy, e_beam, width_target))
        .collect();


    // we add in straggling
    let beam_straggle: Vec<f64> = energy_grid
        .par_iter()
        .map(|&energy| {
            simple_yield::non_resonant_straggle(
                energy,
                e_beam,
                straggle_const,
                &energy_grid,
                &target_grid,
            )
        })
        .collect();

    
    // broadining before target
    let beam_spread: Vec<f64> = energy_grid
        .par_iter()
        .map(|&energy| {
            simple_yield::beam_spread_point(energy, beam_fwhm, &energy_grid, &beam_straggle)
        })
        .collect();

    
    // Generate the cross section yield for the target profile.
    let yield_grid: Vec<f64> = energy_grid
        .iter()
        .zip(beam_spread.iter())
        .map(|(&energy, &target_point)| cross_section.get_value(energy, target_point, height))
        .collect();

    // Energy spread coming from the detector, i.e it does not affect the yield only the shape of the yield
    let det_spread: Vec<f64> = energy_grid
        .par_iter()
        .map(|&energy| {
            simple_yield::beam_spread_point(energy, det_fwhm, &energy_grid, &yield_grid)
        })
        .collect();

    (energy_grid, det_spread)
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn fancy_yield_curve(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(yield_curve, m)?)?;
    m.add_function(wrap_pyfunction!(calc_direct_capture_yield, m)?)?;
    Ok(())
}
