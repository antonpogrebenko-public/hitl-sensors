//! Gaussian noise generation utilities for sensor simulation.

use rand::Rng;

/// Generate two Gaussian-distributed samples from two uniform samples using Box-Muller transform.
pub fn box_muller(u1: f64, u2: f64) -> (f64, f64) {
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f64::consts::PI * u2;
    (r * theta.cos(), r * theta.sin())
}

/// A source of standard normal deviates that keeps Box-Muller's second value.
///
/// The transform produces two independent deviates for one `ln`, one `sqrt` and
/// one `sin`/`cos` pair, and every call site here destructured `let (z, _)` and
/// dropped the second. One IMU sample reached `box_muller` twelve times — six
/// white-noise draws plus one inside each of the six Gauss-Markov steps — so at
/// 400 Hz that discarded roughly 4,800 transcendental calls a second, and burned
/// two RNG draws per deviate instead of one.
///
/// **This changes the random stream.** The values remain correctly distributed —
/// the discarded half was always a valid N(0,1) sample — but a seeded run no
/// longer reproduces its previous sequence. That is a declared, deliberate
/// consequence, not a regression: see the change's design notes. Anything that
/// needs bit-reproducibility against an old recording must pin the old
/// behaviour, not assume it.
#[derive(Debug, Default, Clone)]
pub struct NormalSource {
    spare: Option<f64>,
}

impl NormalSource {
    pub fn new() -> Self {
        Self { spare: None }
    }

    /// Next standard normal deviate.
    pub fn next<R: Rng>(&mut self, rng: &mut R) -> f64 {
        if let Some(z) = self.spare.take() {
            return z;
        }
        let u1: f64 = rng.gen_range(0.0001..1.0);
        let u2: f64 = rng.gen();
        let (z0, z1) = box_muller(u1, u2);
        self.spare = Some(z1);
        z0
    }

    /// Drop any held deviate, so a reset process does not resume mid-pair.
    pub fn reset(&mut self) {
        self.spare = None;
    }
}

/// Gauss-Markov process for correlated noise generation.
/// Used for gyro bias drift and GPS position drift.
pub struct GaussMarkov {
    tau: f64,   // Time constant (seconds)
    sigma: f64, // Steady-state standard deviation
    state: f64, // Current state
    /// Coefficients for the last `dt` this process was stepped with.
    ///
    /// `alpha` and `noise_sigma` depend only on `dt`, `tau` and `sigma`; `dt` is
    /// the fixed simulation step and the other two never change after
    /// construction. Recomputing them per sample cost two `exp` and one `sqrt`
    /// each time — at 400 Hz across six processes in the IMU alone, about 4,800
    /// `exp` and 2,400 `sqrt` a second producing the same two numbers.
    cached_dt: f64,
    alpha: f64,
    noise_sigma: f64,
    normals: NormalSource,
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
            // NaN never equals the first real dt, so the first step always
            // computes rather than trusting an uninitialised coefficient.
            cached_dt: f64::NAN,
            alpha: 0.0,
            noise_sigma: 0.0,
            normals: NormalSource::new(),
        }
    }

    /// Advance the process by dt seconds and return the new state.
    ///
    /// Uses the discrete-time Gauss-Markov update:
    /// x[k+1] = exp(-dt/tau) * x[k] + w[k]
    /// where w[k] ~ N(0, sigma^2 * (1 - exp(-2*dt/tau)))
    pub fn step<R: Rng>(&mut self, dt: f64, rng: &mut R) -> f64 {
        // Exact equality on purpose: the loop steps with a constant `dt`, so
        // this hits on every sample after the first, and a tolerance would risk
        // reusing coefficients for a genuinely different step.
        if dt != self.cached_dt {
            self.alpha = (-dt / self.tau).exp();
            self.noise_sigma = self.sigma * (1.0 - (-2.0 * dt / self.tau).exp()).sqrt();
            self.cached_dt = dt;
        }

        let z = self.normals.next(rng);
        self.state = self.alpha * self.state + self.noise_sigma * z;
        self.state
    }

    /// Reset the process state to zero.
    pub fn reset(&mut self) {
        self.state = 0.0;
        self.normals.reset();
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
