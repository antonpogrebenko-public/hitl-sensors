//! Magnetometer sensor simulation.
//!
//! Models a 3-axis magnetometer with configurable Earth field and noise.

use crate::noise::box_muller;
use nalgebra::UnitQuaternion;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Conversion factor from nanotesla to Gauss
const NT_TO_GAUSS: f64 = 1e-5;

/// Configuration for magnetometer sensor.
#[derive(Debug, Clone)]
pub struct MagConfig {
    /// Earth's magnetic field in NED frame (nanotesla).
    /// Default is Boulder, CO (see the `Default` impl for the exact WMM values).
    pub field_ned_nt: [f64; 3],
    /// Measurement noise standard deviation in Gauss
    pub noise_sigma_gauss: f64,
}

impl Default for MagConfig {
    fn default() -> Self {
        Self {
            // Boulder, CO magnetic field (WMM 2025, 40.015°N 105.2705°W, 1655m)
            // Source: NOAA WMM Calculator — https://www.ngdc.noaa.gov/geomag/calculators/magcalc.shtml
            field_ned_nt: [21222.0, 3065.0, 49715.0],
            noise_sigma_gauss: 0.005, // 500 nT
        }
    }
}

/// Simulated magnetometer sensor.
pub struct MagSensor {
    config: MagConfig,
    field_ned_gauss: [f64; 3],
    rng: StdRng,
}

/// Magnetometer reading in body frame.
#[derive(Debug, Clone, Copy)]
pub struct MagReading {
    /// Magnetic field vector in body frame (Gauss)
    pub field: [f32; 3],
}

impl MagSensor {
    /// Create a new magnetometer sensor with default configuration.
    pub fn new() -> Self {
        Self::with_config(MagConfig::default())
    }

    /// Create a new magnetometer sensor with custom configuration.
    pub fn with_config(config: MagConfig) -> Self {
        Self::with_config_and_seed(config, rand::random())
    }

    /// Create a new magnetometer sensor with custom configuration and seed.
    pub fn with_config_and_seed(config: MagConfig, seed: u64) -> Self {
        // Convert field to Gauss
        let field_ned_gauss = [
            config.field_ned_nt[0] * NT_TO_GAUSS,
            config.field_ned_nt[1] * NT_TO_GAUSS,
            config.field_ned_nt[2] * NT_TO_GAUSS,
        ];

        Self {
            config,
            field_ned_gauss,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Sample the magnetometer sensor.
    ///
    /// # Arguments
    /// * `rotation_body_to_ned` - Unit quaternion representing rotation from body to NED frame
    ///
    /// # Returns
    /// Noisy magnetometer reading in body frame
    pub fn sample(&mut self, rotation_body_to_ned: &UnitQuaternion<f64>) -> MagReading {
        // Rotate NED field into body frame
        // R_body_to_ned * field_body = field_ned
        // field_body = R_body_to_ned.inverse() * field_ned = R_ned_to_body * field_ned
        let rotation_ned_to_body = rotation_body_to_ned.inverse();

        let field_ned = nalgebra::Vector3::new(
            self.field_ned_gauss[0],
            self.field_ned_gauss[1],
            self.field_ned_gauss[2],
        );

        let field_body = rotation_ned_to_body * field_ned;

        // Add noise
        let mut field = [0f32; 3];
        for i in 0..3 {
            let u1: f64 = self.rng.gen_range(0.0001..1.0);
            let u2: f64 = self.rng.gen();
            let (z, _) = box_muller(u1, u2);
            field[i] = (field_body[i] + self.config.noise_sigma_gauss * z) as f32;
        }

        MagReading { field }
    }

    /// Get the current configuration.
    pub fn config(&self) -> &MagConfig {
        &self.config
    }

    /// Reset accumulated sensor state to initial conditions.
    ///
    /// Currently the magnetometer has no accumulated drift or bias state, so
    /// this is a no-op. It exists so that `Sensors::reset()` can call it
    /// uniformly and callers are not broken when hard/soft iron drift state is
    /// added in the future.
    pub fn reset(&mut self) {
        // No accumulated state to reset at this time.
        // Add hard/soft iron bias resets here as the model evolves.
    }

    /// Update the magnetic field (e.g., for different location).
    pub fn set_field_ned(&mut self, field_ned_nt: [f64; 3]) {
        self.config.field_ned_nt = field_ned_nt;
        self.field_ned_gauss = [
            field_ned_nt[0] * NT_TO_GAUSS,
            field_ned_nt[1] * NT_TO_GAUSS,
            field_ned_nt[2] * NT_TO_GAUSS,
        ];
    }

    /// Get the expected field magnitude in Gauss.
    pub fn field_magnitude_gauss(&self) -> f64 {
        (self.field_ned_gauss[0].powi(2)
            + self.field_ned_gauss[1].powi(2)
            + self.field_ned_gauss[2].powi(2))
        .sqrt()
    }
}

impl Default for MagSensor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    #[test]
    fn test_mag_rotation() {
        let mut sensor = MagSensor::with_config_and_seed(MagConfig::default(), 42);

        // Level attitude (identity rotation) - body frame = NED frame
        let identity = UnitQuaternion::identity();
        let reading = sensor.sample(&identity);

        // Should be close to NED field (in Gauss)
        let expected_n = sensor.config.field_ned_nt[0] * NT_TO_GAUSS;
        let expected_e = sensor.config.field_ned_nt[1] * NT_TO_GAUSS;
        let expected_d = sensor.config.field_ned_nt[2] * NT_TO_GAUSS;

        assert!(
            (reading.field[0] as f64 - expected_n).abs() < 0.02,
            "X (North) should be ~{}, got {}",
            expected_n,
            reading.field[0]
        );
        assert!(
            (reading.field[1] as f64 - expected_e).abs() < 0.02,
            "Y (East) should be ~{}, got {}",
            expected_e,
            reading.field[1]
        );
        assert!(
            (reading.field[2] as f64 - expected_d).abs() < 0.02,
            "Z (Down) should be ~{}, got {}",
            expected_d,
            reading.field[2]
        );
    }

    #[test]
    fn test_mag_90deg_yaw() {
        // Use zero noise for deterministic test
        let config = MagConfig {
            field_ned_nt: [22200.0, 0.0, 41800.0], // Zero east component for clarity
            noise_sigma_gauss: 0.0,
        };
        let mut sensor = MagSensor::with_config_and_seed(config, 42);

        // 90 degree yaw (heading east)
        // Body X points East, Body Y points South
        let yaw_90 = UnitQuaternion::from_euler_angles(0.0, 0.0, PI / 2.0);
        let reading = sensor.sample(&yaw_90);

        let field_n = sensor.field_ned_gauss[0];
        let field_d = sensor.field_ned_gauss[2];

        // After 90 deg yaw:
        // Body X (forward) should see ~0 (was pointing North, now East)
        // Body Y (right) should see -field_n (was pointing East, now South)
        // Body Z (down) should still see field_d
        assert!(
            reading.field[0].abs() < 0.01,
            "Body X should be ~0 after 90deg yaw, got {}",
            reading.field[0]
        );
        assert_relative_eq!(reading.field[1] as f64, -field_n, epsilon = 0.01);
        assert_relative_eq!(reading.field[2] as f64, field_d, epsilon = 0.01);
    }

    #[test]
    fn test_mag_90deg_pitch() {
        let config = MagConfig {
            field_ned_nt: [22200.0, 0.0, 41800.0],
            noise_sigma_gauss: 0.0,
        };
        let mut sensor = MagSensor::with_config_and_seed(config, 42);

        // 90 degree pitch up
        // Body X points Up, Body Z points North
        let pitch_90 = UnitQuaternion::from_euler_angles(0.0, PI / 2.0, 0.0);
        let reading = sensor.sample(&pitch_90);

        let field_n = sensor.field_ned_gauss[0];
        let field_d = sensor.field_ned_gauss[2];

        // After 90 deg pitch up:
        // Body X (forward) should see -field_d (was pointing North, now Up)
        // Body Y (right) should see 0
        // Body Z (down) should see field_n (was pointing Down, now North)
        assert_relative_eq!(reading.field[0] as f64, -field_d, epsilon = 0.01);
        assert!(
            reading.field[1].abs() < 0.01,
            "Body Y should be ~0, got {}",
            reading.field[1]
        );
        assert_relative_eq!(reading.field[2] as f64, field_n, epsilon = 0.01);
    }

    #[test]
    fn test_mag_field_magnitude_preserved() {
        let config = MagConfig::default();
        let mut sensor = MagSensor::with_config_and_seed(config, 42);

        let expected_mag = sensor.field_magnitude_gauss();

        // Test various orientations
        let orientations = [
            UnitQuaternion::identity(),
            UnitQuaternion::from_euler_angles(0.3, 0.2, 0.5),
            UnitQuaternion::from_euler_angles(-0.5, 0.1, -0.3),
            UnitQuaternion::from_euler_angles(0.0, PI / 4.0, PI / 3.0),
        ];

        for orientation in &orientations {
            // Take multiple samples and average to reduce noise
            let mut mag_sum = 0.0;
            let n = 100;
            for _ in 0..n {
                let reading = sensor.sample(orientation);
                let mag = (reading.field[0].powi(2)
                    + reading.field[1].powi(2)
                    + reading.field[2].powi(2))
                .sqrt();
                mag_sum += mag as f64;
            }
            let avg_mag = mag_sum / n as f64;

            // Magnitude should be preserved (within noise tolerance)
            assert!(
                (avg_mag - expected_mag).abs() < 0.01,
                "Field magnitude {} should be ~{}",
                avg_mag,
                expected_mag
            );
        }
    }

    #[test]
    fn test_mag_noise_stats() {
        let config = MagConfig::default();
        let mut sensor = MagSensor::with_config_and_seed(config.clone(), 42);

        let identity = UnitQuaternion::identity();
        let n = 10000;
        let mut readings: Vec<[f32; 3]> = Vec::with_capacity(n);

        for _ in 0..n {
            readings.push(sensor.sample(&identity).field);
        }

        // Check noise standard deviation on each axis
        for axis in 0..3 {
            let mean: f64 = readings.iter().map(|r| r[axis] as f64).sum::<f64>() / n as f64;
            let variance: f64 = readings
                .iter()
                .map(|r| (r[axis] as f64 - mean).powi(2))
                .sum::<f64>()
                / n as f64;
            let actual_sigma = variance.sqrt();

            // Allow 30% tolerance
            assert!(
                (actual_sigma - config.noise_sigma_gauss).abs() / config.noise_sigma_gauss < 0.3,
                "Axis {} noise sigma {} should be near {}",
                axis,
                actual_sigma,
                config.noise_sigma_gauss
            );
        }
    }
}
