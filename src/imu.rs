//! IMU (Inertial Measurement Unit) sensor simulation.
//!
//! Models gyroscope and accelerometer with realistic noise characteristics
//! based on Kalibr/MPU-6000 specifications.

use crate::noise::{box_muller, GaussMarkov};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Configuration for IMU noise characteristics.
#[derive(Debug, Clone)]
pub struct ImuConfig {
    /// Gyroscope noise density in rad/s/√Hz
    pub gyro_noise_density: f64,
    /// Accelerometer noise density in m/s²/√Hz
    pub accel_noise_density: f64,
    /// Gyroscope bias random walk sigma in rad/s
    pub gyro_bias_sigma: f64,
    /// Gyroscope bias time constant in seconds
    pub gyro_bias_tau: f64,
    /// Accelerometer bias stability sigma in m/s² (~0.3mg for typical MEMS IMU)
    pub accel_bias_sigma: f64,
    /// Accelerometer bias correlation time in seconds
    pub accel_bias_tau: f64,
}

impl Default for ImuConfig {
    fn default() -> Self {
        // Based on MPU-6000 / Kalibr typical values
        Self {
            gyro_noise_density: 0.0008726646, // rad/s/√Hz (~0.05 deg/s/√Hz)
            accel_noise_density: 0.00637,     // m/s²/√Hz (~650 µg/√Hz)
            gyro_bias_sigma: 1e-4,            // rad/s
            gyro_bias_tau: 100.0,             // seconds
            accel_bias_sigma: 0.003,          // m/s² (~0.3mg bias stability)
            accel_bias_tau: 300.0,            // seconds
        }
    }
}

/// Simulated IMU sensor with gyroscope and accelerometer.
pub struct ImuSensor {
    config: ImuConfig,
    gyro_bias: [GaussMarkov; 3],
    accel_bias: [GaussMarkov; 3],
    rng: StdRng,
}

/// IMU reading containing accelerometer and gyroscope measurements.
#[derive(Debug, Clone, Copy)]
pub struct ImuReading {
    /// Accelerometer measurement in body frame (m/s²)
    pub accel: [f32; 3],
    /// Gyroscope measurement in body frame (rad/s)
    pub gyro: [f32; 3],
}

impl ImuSensor {
    /// Create a new IMU sensor with default configuration.
    pub fn new() -> Self {
        Self::with_config(ImuConfig::default())
    }

    /// Create a new IMU sensor with custom configuration.
    pub fn with_config(config: ImuConfig) -> Self {
        Self::with_config_and_seed(config, rand::random())
    }

    /// Create a new IMU sensor with custom configuration and seed for reproducibility.
    pub fn with_config_and_seed(config: ImuConfig, seed: u64) -> Self {
        let gyro_bias = [
            GaussMarkov::new(config.gyro_bias_tau, config.gyro_bias_sigma),
            GaussMarkov::new(config.gyro_bias_tau, config.gyro_bias_sigma),
            GaussMarkov::new(config.gyro_bias_tau, config.gyro_bias_sigma),
        ];
        let accel_bias = [
            GaussMarkov::new(config.accel_bias_tau, config.accel_bias_sigma),
            GaussMarkov::new(config.accel_bias_tau, config.accel_bias_sigma),
            GaussMarkov::new(config.accel_bias_tau, config.accel_bias_sigma),
        ];

        Self {
            config,
            gyro_bias,
            accel_bias,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Sample the IMU sensor.
    ///
    /// # Arguments
    /// * `true_accel_body` - True specific force in body frame (m/s²)
    /// * `true_gyro` - True angular velocity in body frame (rad/s)
    /// * `dt` - Time step since last sample (seconds)
    ///
    /// # Returns
    /// Noisy IMU reading
    pub fn sample(
        &mut self,
        true_accel_body: &[f64; 3],
        true_gyro: &[f64; 3],
        dt: f64,
    ) -> ImuReading {
        let dt_sqrt = dt.sqrt();

        // Generate white noise + bias for accelerometer
        let accel_noise_sigma = self.config.accel_noise_density / dt_sqrt;
        let mut accel = [0f32; 3];
        for i in 0..3 {
            let bias = self.accel_bias[i].step(dt, &mut self.rng);
            let u1: f64 = self.rng.gen_range(0.0001..1.0);
            let u2: f64 = self.rng.gen();
            let (z, _) = box_muller(u1, u2);
            accel[i] = (true_accel_body[i] + bias + accel_noise_sigma * z) as f32;
        }

        // Generate white noise + bias for gyroscope
        let gyro_noise_sigma = self.config.gyro_noise_density / dt_sqrt;
        let mut gyro = [0f32; 3];
        for i in 0..3 {
            let bias = self.gyro_bias[i].step(dt, &mut self.rng);
            let u1: f64 = self.rng.gen_range(0.0001..1.0);
            let u2: f64 = self.rng.gen();
            let (z, _) = box_muller(u1, u2);
            gyro[i] = (true_gyro[i] + bias + gyro_noise_sigma * z) as f32;
        }

        ImuReading { accel, gyro }
    }

    /// Reset the sensor state (clears bias drift).
    pub fn reset(&mut self) {
        for bias in &mut self.gyro_bias {
            bias.reset();
        }
        for bias in &mut self.accel_bias {
            bias.reset();
        }
    }

    /// Get the current configuration.
    pub fn config(&self) -> &ImuConfig {
        &self.config
    }
}

impl Default for ImuSensor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_imu_noise_stats() {
        let config = ImuConfig::default();
        let mut sensor = ImuSensor::with_config_and_seed(config.clone(), 42);

        let dt = 0.001; // 1000 Hz
        let n = 10000;

        let true_accel = [0.0, 0.0, 9.81]; // Gravity in body frame
        let true_gyro = [0.0, 0.0, 0.0];

        let mut accel_samples: Vec<[f64; 3]> = Vec::with_capacity(n);
        let mut gyro_samples: Vec<[f64; 3]> = Vec::with_capacity(n);

        for _ in 0..n {
            let reading = sensor.sample(&true_accel, &true_gyro, dt);
            accel_samples.push([
                reading.accel[0] as f64,
                reading.accel[1] as f64,
                reading.accel[2] as f64,
            ]);
            gyro_samples.push([
                reading.gyro[0] as f64,
                reading.gyro[1] as f64,
                reading.gyro[2] as f64,
            ]);
        }

        // Check accelerometer noise standard deviation
        for axis in 0..3 {
            let mean: f64 = accel_samples.iter().map(|s| s[axis]).sum::<f64>() / n as f64;
            let variance: f64 = accel_samples
                .iter()
                .map(|s| (s[axis] - mean).powi(2))
                .sum::<f64>()
                / n as f64;
            let actual_sigma = variance.sqrt();
            let expected_sigma = config.accel_noise_density / dt.sqrt();

            // Allow 20% tolerance
            assert!(
                (actual_sigma - expected_sigma).abs() / expected_sigma < 0.2,
                "Accel axis {} sigma {} should be near {}",
                axis,
                actual_sigma,
                expected_sigma
            );
        }

        // Check gyroscope noise (includes bias, so more tolerance)
        for axis in 0..3 {
            let mean: f64 = gyro_samples.iter().map(|s| s[axis]).sum::<f64>() / n as f64;
            let variance: f64 = gyro_samples
                .iter()
                .map(|s| (s[axis] - mean).powi(2))
                .sum::<f64>()
                / n as f64;
            let actual_sigma = variance.sqrt();
            let expected_sigma = config.gyro_noise_density / dt.sqrt();

            // Allow 30% tolerance due to bias drift contribution
            assert!(
                (actual_sigma - expected_sigma).abs() / expected_sigma < 0.3,
                "Gyro axis {} sigma {} should be near {}",
                axis,
                actual_sigma,
                expected_sigma
            );
        }
    }

    #[test]
    fn test_imu_mean_values() {
        let mut sensor = ImuSensor::with_config_and_seed(ImuConfig::default(), 123);

        let dt = 0.001;
        let n = 5000;

        let true_accel = [1.0, -2.0, 9.81];
        let true_gyro = [0.1, -0.05, 0.02];

        let mut accel_sum = [0.0f64; 3];
        let mut gyro_sum = [0.0f64; 3];

        for _ in 0..n {
            let reading = sensor.sample(&true_accel, &true_gyro, dt);
            for i in 0..3 {
                accel_sum[i] += reading.accel[i] as f64;
                gyro_sum[i] += reading.gyro[i] as f64;
            }
        }

        // Check that means are close to true values
        for i in 0..3 {
            let accel_mean = accel_sum[i] / n as f64;
            let gyro_mean = gyro_sum[i] / n as f64;

            assert!(
                (accel_mean - true_accel[i]).abs() < 0.1,
                "Accel mean {} should be near {}",
                accel_mean,
                true_accel[i]
            );
            assert!(
                (gyro_mean - true_gyro[i]).abs() < 0.01,
                "Gyro mean {} should be near {}",
                gyro_mean,
                true_gyro[i]
            );
        }
    }
}
