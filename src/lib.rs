use cross_section::CrossSection;
use pyo3::{exceptions, prelude::*};
use rayon::prelude::*;
mod cross_section;
mod gaussian;
mod simple_yield;
mod target_profile;
use std::sync::Mutex;

/// Global target state
static TARGET_LAYERS: Mutex<Option<Vec<target_profile::TargetLayer>>> = Mutex::new(None);

#[pyfunction]
fn add_layer(dx: f64, concentrations: Vec<f64>, stopping_powers: Vec<f64>) -> usize {
    let new_layer = target_profile::TargetLayer::new(dx, concentrations, stopping_powers);
    let mut guard = TARGET_LAYERS.lock().unwrap();
    match &mut *guard {
        Some(ref mut v) => {
            v.push(new_layer);
            v.len() - 1
        }
        None => {
            *guard = Some(vec![new_layer]);
            0
        }
    }
}

#[pyfunction]
fn replace_layer(
    index: usize,
    dx: f64,
    concentrations: Vec<f64>,
    stopping_powers: Vec<f64>,
) -> PyResult<()> {
    let new_layer = target_profile::TargetLayer::new(dx, concentrations, stopping_powers);
    let mut guard = TARGET_LAYERS.lock().unwrap();
    match &mut *guard {
        Some(ref mut v) => match v.get_mut(index) {
            Some(layer) => {
                *layer = new_layer;
                Ok(())
            }
            None => Err(exceptions::PyIndexError::new_err(
                "layer index out of range",
            )),
        },
        None => Err(exceptions::PyIndexError::new_err("no layers defined")),
    }
}

#[pyfunction]
fn remove_layer(index: usize) -> PyResult<()> {
    let mut guard = TARGET_LAYERS.lock().unwrap();
    if let Some(ref mut v) = &mut *guard {
        if index < v.len() {
            v.remove(index);
            Ok(())
        } else {
            Err(exceptions::PyIndexError::new_err(
                "layer index out of range",
            ))
        }
    } else {
        Err(exceptions::PyIndexError::new_err("no layers defined"))
    }
}

#[pyfunction]
fn print_layer(index: usize) -> PyResult<()> {
    let guard = TARGET_LAYERS.lock().unwrap();
    match &*guard {
        Some(v) => match v.get(index) {
            Some(layer) => {
                println!("{:?}", layer);
                Ok(())
            }
            None => Err(exceptions::PyIndexError::new_err(
                "layer index out of range",
            )),
        },
        None => Err(exceptions::PyIndexError::new_err("no layers defined")),
    }
}

#[pyfunction]
fn clear_layers() -> PyResult<()> {
    let mut guard = TARGET_LAYERS.lock().unwrap();
    match &mut *guard {
        Some(v) => {
            *guard = None;
            Ok(())
        }
        None => Err(exceptions::PyIndexError::new_err("no layers defined")),
    }
}

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
fn yield_curve_complete(
    e_res: f64,
    width_res: f64,
    beam_fwhm: f64,
    height: f64,
    width_target: f64,
    straggle_const: f64,
    start_energy: f64,
    stop_energy: f64,
    step_size: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
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

    (
        energy_grid,
        yield_grid,
        beam_spread,
        beam_straggle,
        target_straggle,
    )
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
        .map(|&energy| simple_yield::beam_spread_point(energy, det_fwhm, &energy_grid, &yield_grid))
        .collect();

    (energy_grid, det_spread)
}
#[pyfunction]
fn calc_direct_capture_yield_complete(
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
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
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
        .map(|&energy| simple_yield::beam_spread_point(energy, det_fwhm, &energy_grid, &yield_grid))
        .collect();

    (
        energy_grid,
        target_grid,
        beam_straggle,
        beam_spread,
        yield_grid,
        det_spread,
    )
}

#[pyfunction]
fn calc_direct_capture_yield_target_profile(
    e_beam: f64,
    beam_fwhm: f64,
    det_fwhm: f64,
    height: f64,
    straggle_const: f64,
    cs_energies: Vec<f64>,
    cs_values: Vec<f64>,
    start_energy: f64,
    stop_energy: f64,
    step_size: f64,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let mut cross_section = CrossSection::new(cs_energies, cs_values, e_beam);

    let binding = TARGET_LAYERS.lock().unwrap();
    let layers = match (*binding).as_ref() {
        Some(v) => v,
        None => return Err(exceptions::PyIndexError::new_err("no layers defined")),
    };

    let mut profile_target = target_profile::TargetProfile::new(e_beam, layers);
    let num_points = (stop_energy - start_energy) / step_size;
    // create the requested energy grid
    let energy_grid: Vec<f64> = (0..num_points as usize)
        .map(|i| {
            start_energy
                + (stop_energy - start_energy) * i as f64 / (num_points as usize - 1) as f64
        })
        .collect();

    // calculate the based on target profile
    let target_grid: Vec<f64> = energy_grid
        .iter()
        .map(|&energy| profile_target.get_value(e_beam, energy))
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
        .map(|&energy| simple_yield::beam_spread_point(energy, det_fwhm, &energy_grid, &yield_grid))
        .collect();

    Ok((energy_grid, det_spread))
}

#[pyfunction]
fn calc_direct_capture_yield_target_profile_complete(
    e_beam: f64,
    beam_fwhm: f64,
    det_fwhm: f64,
    height: f64,
    straggle_const: f64,
    cs_energies: Vec<f64>,
    cs_values: Vec<f64>,
    start_energy: f64,
    stop_energy: f64,
    step_size: f64,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> {
    let mut cross_section = CrossSection::new(cs_energies, cs_values, e_beam);

    let binding = TARGET_LAYERS.lock().unwrap();
    let layers = match (*binding).as_ref() {
        Some(v) => v,
        None => return Err(exceptions::PyIndexError::new_err("no layers defined")),
    };

    let mut profile_target = target_profile::TargetProfile::new(e_beam, layers);
    let num_points = (stop_energy - start_energy) / step_size;
    // create the requested energy grid
    let energy_grid: Vec<f64> = (0..num_points as usize)
        .map(|i| {
            start_energy
                + (stop_energy - start_energy) * i as f64 / (num_points as usize - 1) as f64
        })
        .collect();

    // calculate the based on target profile
    let target_grid: Vec<f64> = energy_grid
        .iter()
        .map(|&energy| profile_target.get_value(e_beam, energy))
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
        .map(|&energy| simple_yield::beam_spread_point(energy, det_fwhm, &energy_grid, &yield_grid))
        .collect();

    Ok((
        energy_grid,
        target_grid,
        beam_straggle,
        beam_spread,
        yield_grid,
        det_spread,
    ))
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn fancy_yield_curve(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add_layer, m)?)?;
    m.add_function(wrap_pyfunction!(replace_layer, m)?)?;
    m.add_function(wrap_pyfunction!(remove_layer, m)?)?;
    m.add_function(wrap_pyfunction!(print_layer, m)?)?;
    m.add_function(wrap_pyfunction!(clear_layers, m)?)?;
    m.add_function(wrap_pyfunction!(yield_curve, m)?)?;
    m.add_function(wrap_pyfunction!(yield_curve_complete, m)?)?;
    m.add_function(wrap_pyfunction!(calc_direct_capture_yield, m)?)?;
    m.add_function(wrap_pyfunction!(calc_direct_capture_yield_complete, m)?)?;
    m.add_function(wrap_pyfunction!(
        calc_direct_capture_yield_target_profile,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        calc_direct_capture_yield_target_profile_complete,
        m
    )?)?;
    Ok(())
}
