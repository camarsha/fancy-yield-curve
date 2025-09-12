/*
This module allows for different target distribution functions
 */

use crate::gaussian::Gaussian;
use crate::simple_yield::fwhm_to_std;
use rgsl::interpolation;
use rgsl::{Interp, InterpAccel, InterpType};

fn cumsum(v: &[f64]) -> Vec<f64> {
    v.iter()
        .scan(0.0, |acc, x| {
            *acc += *x;
            Some(*acc)
        })
        .collect::<Vec<f64>>()
}

/// Create a uniform target slab with unit height.
pub fn uniform_target(energy: f64, e_beam: f64, width_target: f64) -> f64 {
    if (energy >= e_beam) || (energy <= (e_beam - width_target)) {
        0.0
    } else {
        1.0
    }
}

#[derive(Debug)]
pub struct TargetLayer {
    dx: f64,
    concentrations: Vec<f64>,
    stopping_powers: Vec<f64>,
    eff_stopping_power: f64,
    tot_stopping_power: f64,
}

impl TargetLayer {
    /// Make a new target layer of thickness, dx, with concentrations given by a Vec with sum = 1,
    /// and stopping powers for each nuclei units of dx. The first concentration/stopping power is
    /// considered to the active nucleus for the effective stopping power calculation.
    pub fn new(dx: f64, concentrations: Vec<f64>, stopping_powers: Vec<f64>) -> Self {
        assert!(
            f64::abs(concentrations.iter().sum::<f64>() - 1.0) < 1e-3,
            "Concentrations do not sum to 1.0 within a tolerance of 1e-3!"
        );
        let act_conc = concentrations[0];
        let tot_stopping_power = concentrations
            .iter()
            .zip(stopping_powers.iter())
            .fold(0.0, |acc, (c, sp)| c * sp);
        let eff_stopping_power = tot_stopping_power / act_conc;
        Self {
            dx,
            concentrations,
            stopping_powers,
            eff_stopping_power,
            tot_stopping_power,
        }
    }

    pub fn integrate_layer(&self, e_start: f64) -> (Vec<f64>, f64) {
        let steps = self.dx.floor() as usize; // right now just
        let mut energy_grid: Vec<f64> = Vec::with_capacity(steps);
        let step_size = self.tot_stopping_power * (self.dx / steps as f64);
        for i in 0..steps {
            energy_grid.push(e_start - (i as f64 * step_size));
        }
        (energy_grid, e_start - (steps as f64 * step_size))
    }
}

pub struct TargetProfile {
    energies: Vec<f64>,
    height_target: Vec<f64>,
    interp: Interp,
    interp_acc: InterpAccel,
    min_value: f64,
}

impl TargetProfile {
    pub fn new(e_beam: f64, layers: &[TargetLayer]) -> Self {
        // The user has given us a list of target layers, we know transform them into a continuous energy grid
        // for the convolutions.

        let mut current_energy = e_beam;
        let mut energies = Vec::new();
        let mut height_target = Vec::new();

        for layer in layers.iter() {
            // push the current values
            let (mut e, ce) = layer.integrate_layer(current_energy);
            current_energy = ce;
            let mut h = vec![1.0 / layer.eff_stopping_power; e.len()];
            energies.append(&mut e);
            height_target.append(&mut h);
        }

        // Finish up
        energies.push(current_energy);
        height_target.push(1.0 / layers.last().unwrap().eff_stopping_power);

        // Spline needs increasing values
        energies.reverse();
        height_target.reverse();
        // Interpolate
        let interp_type = InterpType::linear(); // Linear avoids issues for the steps
        let mut interp =
            Interp::new(interp_type, energies.len()).expect("Failed to initialize TargetProfile spline.");
        interp.init(&energies, &height_target).unwrap();
        let mut interp_acc = InterpAccel::new();
        let min_value = *energies.first().unwrap();
        Self {
            energies,
            height_target,
            interp,
            interp_acc,
            min_value,
        }
    }

    pub fn get_value(&mut self, e_beam: f64, energy: f64) -> f64 {
        if (energy < self.min_value) || (energy > e_beam) {
            return 0.0;
        }
        interpolation::eval(
            &self.interp,
            &self.energies,
            &self.height_target,
            energy,
            &mut self.interp_acc,
        )
    }
}
