use std::f64::consts::PI;

/// Provides a gaussian struct for convience in the rest of the library
#[derive(Debug)]
pub struct Gaussian {
    pub mean: f64,
    pub std: f64,
    pub var: f64,
}

impl Gaussian {
    pub fn new(mean: f64, std: f64) -> Self {
        Gaussian {
            mean,
            std,
            var: std.powi(2),
        }
    }

    pub fn pdf(&self, x: f64) -> f64 {
        let norm = 1.0 / (self.std * (2.0 * PI).sqrt());
        let num = (x - self.mean).powi(2);
        norm * f64::exp(-0.5 * num / self.var)
    }
}

pub fn gaussian_pdf(mean: f64, std: f64, x: f64) -> f64 {
    let norm = 1.0 / (std * (2.0 * PI).sqrt());
    let num = (x - mean).powi(2);
    norm * f64::exp(-0.5 * num / std.powi(2))
}
