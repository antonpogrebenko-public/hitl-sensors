//! Gaussian noise generation utilities for sensor simulation.

use rand::Rng;

/// Generate two Gaussian-distributed samples from two uniform samples using Box-Muller transform.
pub fn box_muller(u1: f64, u2: f64) -> (f64, f64) {
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f64::consts::PI * u2;
    (r * theta.cos(), r * theta.sin())
}

/// Gauss-Markov process for correlated noise generation.
/// Used for gyro bias drift and GPS position drift.
pub struct GaussMarkov {
    tau: f64,   // Time constant (seconds)
    sigma: f64, // Steady-state standard deviation
    state: f64, // Current state
}

impl GaussMarkov {
    /// Create a new Gauss-Markov process.
    ///
    /// # Arguments
    /// * `tau` - Time constant in seconds (larger = slower drift)
    /// * `sigma` - Steady-state standard deviation
    pub fn new(tau: f64, sigma: f64) -> Self {
        Self {
            tau,
            sigma,
            state: 0.0,
        }
    }

    /// Advance the process by dt seconds and return the new state.
    ///
    /// Uses the discrete-time Gauss-Markov update:
    /// x[k+1] = exp(-dt/tau) * x[k] + w[k]
    /// where w[k] ~ N(0, sigma^2 * (1 - exp(-2*dt/tau)))
    pub fn step<R: Rng>(&mut self, dt: f64, rng: &mut R) -> f64 {
        let alpha = (-dt / self.tau).exp();
        let noise_sigma = self.sigma * (1.0 - (-2.0 * dt / self.tau).exp()).sqrt();

        let u1: f64 = rng.gen_range(0.0001..1.0);
        let u2: f64 = rng.gen();
        let (z, _) = box_muller(u1, u2);

        self.state = alpha * self.state + noise_sigma * z;
        self.state
    }

    /// Reset the process state to zero.
    pub fn reset(&mut self) {
        self.state = 0.0;
    }

    /// Get the current state value.
    pub fn state(&self) -> f64 {
        self.state
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_box_muller_distribution() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let n = 10000;
        let mut samples = Vec::with_capacity(n * 2);

        for _ in 0..n {
            let u1: f64 = rng.gen_range(0.0001..1.0);
            let u2: f64 = rng.gen();
            let (z1, z2) = box_muller(u1, u2);
            samples.push(z1);
            samples.push(z2);
        }

        let mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;
        let variance: f64 =
            samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / samples.len() as f64;

        // Mean should be close to 0, variance close to 1
        assert!(mean.abs() < 0.05, "Mean {} should be near 0", mean);
        assert!(
            (variance - 1.0).abs() < 0.1,
            "Variance {} should be near 1",
            variance
        );
    }

    #[test]
    fn test_gauss_markov_convergence() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let sigma = 0.1;
        let tau = 10.0;
        let mut gm = GaussMarkov::new(tau, sigma);

        // Run for many steps
        let dt = 0.01;
        let n = 100000;
        let mut samples = Vec::with_capacity(n);

        for _ in 0..n {
            samples.push(gm.step(dt, &mut rng));
        }

        // After convergence, variance should approach sigma^2
        let mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;
        let variance: f64 =
            samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / samples.len() as f64;
        let actual_sigma = variance.sqrt();

        assert!(
            (actual_sigma - sigma).abs() < 0.02,
            "Sigma {} should be near {}",
            actual_sigma,
            sigma
        );
    }
}
