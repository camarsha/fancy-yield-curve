use rgsl::interpolation;
use rgsl::{Interp, InterpAccel, InterpType};
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;

/// This module handles reading in a cross section file and producing a closure that
/// can be called by another function.

// The output is wrapped in a Result to allow matching on errors.
// Returns an Iterator to the Reader of the lines of the file.
pub fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where
    P: AsRef<Path>,
{
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}

pub struct CrossSection {
    energies: Vec<f64>,
    cs_values: Vec<f64>,
    interp: Interp,
    interp_acc: InterpAccel,
}

impl CrossSection {
    pub fn new(energies: Vec<f64>, cs_values: Vec<f64>, e_beam: f64) -> Self {
        // create the interpolation
        let interp_type = InterpType::cspline();
        let mut interp =
            Interp::new(interp_type, energies.len()).expect("Failed to initialize cubic spline.");
        interp.init(&energies, &cs_values).unwrap();
        let mut interp_acc = InterpAccel::new();
        // find the max value for normalization
        CrossSection {
            energies,
            cs_values,
            interp,
            interp_acc,
        }
    }

    /// Gets the interpolated value of the cross section at the energy point.
    pub fn get_value(&mut self, energy: f64, target_point: f64, height: f64) -> f64 {
        interpolation::eval(
            &self.interp,
            &self.energies,
            &self.cs_values,
            energy,
            &mut self.interp_acc,
        ) * height
            * target_point
    }
}
