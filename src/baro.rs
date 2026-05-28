//! Barometer sensor simulation using ISA (International Standard Atmosphere) model.

use crate::noise::box_muller;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Sea level standard atmospheric pressure in Pascals
const ISA_P0: f64 = 101325.0;
/// Sea level standard temperature in Kelvin
const ISA_T0: f64 = 288.15;
/// Temperature lapse rate in K/m
const ISA_LAPSE_RATE: f64 = 0.0065;
/// Exponent for pressure calculation
const ISA_EXPONENT: f64 = 5.2561;

/// Configuration for barometer sensor.
#[derive(Debug, Clone)]
pub struct BaroConfig {
    /// Altitude measurement noise standard deviation in meters
    pub noise_sigma: f64,
    /// Reference pressure at sea level in Pascals
    pub reference_pressure_pa: f64,
}

impl Default for BaroConfig {
    fn default() -> Self {
        Self {
            noise_sigma: 0.15, // meters
            reference_pressure_pa: ISA_P0,
        }
    }
}

/// Simulated barometer sensor.
pub struct BaroSensor {
    config: BaroConfig,
    rng: StdRng,
}

/// Barometer reading containing pressure, altitude, and temperature.
#[derive(Debug, Clone, Copy)]
pub struct BaroReading {
    /// Atmospheric pressure in Pascals
    pub pressure_pa: f32,
    /// Pressure altitude in meters (above reference pressure level)
    pub altitude_m: f32,
    /// Temperature at altitude in Celsius
    pub temperature_c: f32,
}

impl BaroSensor {
    /// Create a new barometer sensor with default configuration.
    pub fn new() -> Self {
        Self::with_config(BaroConfig::default())
    }

    /// Create a new barometer sensor with custom configuration.
    pub fn with_config(config: BaroConfig) -> Self {
        Self::with_config_and_seed(config, rand::random())
    }

    /// Create a new barometer sensor with custom configuration and seed.
    pub fn with_config_and_seed(config: BaroConfig, seed: u64) -> Self {
        Self {
            config,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Calculate temperature at altitude using ISA model.
    ///
    /// T = T0 - L * h
    /// where T0 = 288.15 K, L = 0.0065 K/m
    ///
    /// Input is clamped to the valid troposphere range [-500m, 11000m] to prevent
    /// negative temperatures (and NaN in pressure calculations) from physics
    /// instabilities that push NED altitude far out of range.
    fn isa_temperature_k(altitude_m: f64) -> f64 {
        let clamped = altitude_m.clamp(-500.0, 11_000.0);
        ISA_T0 - ISA_LAPSE_RATE * clamped
    }

    /// Calculate pressure at altitude using ISA model.
    ///
    /// P = P0 * (T/T0)^5.2561
    ///
    /// Input is clamped to the valid troposphere range [-500m, 11000m] to prevent
    /// NaN from powf() when temperature goes negative at extreme altitudes.
    fn isa_pressure_pa(altitude_m: f64) -> f64 {
        let clamped_alt = altitude_m.clamp(-500.0, 11_000.0);
        let temp_k = Self::isa_temperature_k(clamped_alt);
        ISA_P0 * (temp_k / ISA_T0).powf(ISA_EXPONENT)
    }

    /// Calculate altitude from pressure using inverse ISA model.
    fn pressure_to_altitude(pressure_pa: f64, reference_pressure_pa: f64) -> f64 {
        // h = (T0/L) * (1 - (P/P0)^(1/5.2561))
        let pressure_ratio = pressure_pa / reference_pressure_pa;
        (ISA_T0 / ISA_LAPSE_RATE) * (1.0 - pressure_ratio.powf(1.0 / ISA_EXPONENT))
    }

    /// Sample the barometer sensor.
    ///
    /// # Arguments
    /// * `true_altitude_m` - True geometric altitude in meters MSL
    ///
    /// # Returns
    /// Noisy barometer reading
    pub fn sample(&mut self, true_altitude_m: f64) -> BaroReading {
        // Add noise to altitude measurement
        let u1: f64 = self.rng.gen_range(0.0001..1.0);
        let u2: f64 = self.rng.gen();
        let (z, _) = box_muller(u1, u2);
        let noisy_altitude = true_altitude_m + self.config.noise_sigma * z;

        // Calculate pressure from noisy altitude
        let pressure_pa = Self::isa_pressure_pa(noisy_altitude);

        // Calculate pressure altitude (what the barometer reports)
        let altitude_m =
            Self::pressure_to_altitude(pressure_pa, self.config.reference_pressure_pa);

        // Calculate temperature
        let temperature_k = Self::isa_temperature_k(noisy_altitude);
        let temperature_c = temperature_k - 273.15;

        BaroReading {
            pressure_pa: pressure_pa as f32,
            altitude_m: altitude_m as f32,
            temperature_c: temperature_c as f32,
        }
    }

    /// Get the current configuration.
    pub fn config(&self) -> &BaroConfig {
        &self.config
    }

    /// Reset accumulated sensor state to initial conditions.
    ///
    /// Currently the barometer has no accumulated drift or bias state, so this
    /// is a no-op. It exists so that `Sensors::reset()` can call it uniformly
    /// and callers are not broken when drift state is added in the future.
    pub fn reset(&mut self) {
        // No accumulated state to reset at this time.
        // Add drift/bias resets here as the model evolves.
    }

    /// Update the reference pressure (for QNH adjustment).
    pub fn set_reference_pressure(&mut self, pressure_pa: f64) {
        self.config.reference_pressure_pa = pressure_pa;
    }
}

impl Default for BaroSensor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_baro_isa_model() {
        // Test ISA model at known altitudes
        // At sea level: P = 101325 Pa, T = 288.15 K (15°C)
        let p_0 = BaroSensor::isa_pressure_pa(0.0);
        let t_0 = BaroSensor::isa_temperature_k(0.0);
        assert_relative_eq!(p_0, 101325.0, epsilon = 1.0);
        assert_relative_eq!(t_0, 288.15, epsilon = 0.01);

        // At 1000m: P ≈ 89874 Pa, T ≈ 281.65 K (8.5°C)
        let p_1000 = BaroSensor::isa_pressure_pa(1000.0);
        let t_1000 = BaroSensor::isa_temperature_k(1000.0);
        assert_relative_eq!(p_1000, 89874.0, epsilon = 100.0);
        assert_relative_eq!(t_1000, 281.65, epsilon = 0.01);

        // At 5000m: P ≈ 54019 Pa, T ≈ 255.65 K (-17.5°C)
        let p_5000 = BaroSensor::isa_pressure_pa(5000.0);
        let t_5000 = BaroSensor::isa_temperature_k(5000.0);
        assert_relative_eq!(p_5000, 54019.0, epsilon = 100.0);
        assert_relative_eq!(t_5000, 255.65, epsilon = 0.01);

        // At 11000m (tropopause): P ≈ 22632 Pa, T ≈ 216.65 K (-56.5°C)
        let p_11000 = BaroSensor::isa_pressure_pa(11000.0);
        let t_11000 = BaroSensor::isa_temperature_k(11000.0);
        assert_relative_eq!(p_11000, 22632.0, epsilon = 100.0);
        assert_relative_eq!(t_11000, 216.65, epsilon = 0.01);
    }

    #[test]
    fn test_baro_altitude_roundtrip() {
        // Test that pressure -> altitude -> pressure is consistent
        for alt in [0.0, 100.0, 500.0, 1000.0, 2000.0, 5000.0] {
            let pressure = BaroSensor::isa_pressure_pa(alt);
            let recovered_alt = BaroSensor::pressure_to_altitude(pressure, ISA_P0);
            assert_relative_eq!(recovered_alt, alt, epsilon = 0.01);
        }
    }

    #[test]
    fn test_baro_noise_stats() {
        let config = BaroConfig::default();
        let mut sensor = BaroSensor::with_config_and_seed(config.clone(), 42);

        let n = 10000;
        let true_altitude = 100.0;
        let mut readings = Vec::with_capacity(n);

        for _ in 0..n {
            readings.push(sensor.sample(true_altitude));
        }

        // Check altitude noise standard deviation
        let mean: f64 = readings.iter().map(|r| r.altitude_m as f64).sum::<f64>() / n as f64;
        let variance: f64 = readings
            .iter()
            .map(|r| (r.altitude_m as f64 - mean).powi(2))
            .sum::<f64>()
            / n as f64;
        let actual_sigma = variance.sqrt();

        // Allow 20% tolerance
        assert!(
            (actual_sigma - config.noise_sigma).abs() / config.noise_sigma < 0.2,
            "Altitude noise sigma {} should be near {}",
            actual_sigma,
            config.noise_sigma
        );

        // Check mean is close to true altitude
        assert!(
            (mean - true_altitude).abs() < 0.05,
            "Mean altitude {} should be near {}",
            mean,
            true_altitude
        );
    }

    #[test]
    fn test_baro_extreme_altitudes_no_nan() {
        // Physics instability can push NED altitude far outside normal UAV range.
        // Verify the ISA model never produces NaN or infinite values at extremes.
        for &alt in &[-10_000.0_f64, -1_000.0, 50_000.0, 100_000.0, 1_000_000.0] {
            let pressure = BaroSensor::isa_pressure_pa(alt);
            let temp_k = BaroSensor::isa_temperature_k(alt);
            assert!(
                pressure.is_finite(),
                "pressure should be finite at altitude {}m, got {}",
                alt,
                pressure
            );
            assert!(
                temp_k.is_finite() && temp_k > 0.0,
                "temperature should be finite and positive at altitude {}m, got {}",
                alt,
                temp_k
            );
        }
    }

    #[test]
    fn test_baro_temperature() {
        let mut sensor = BaroSensor::with_config_and_seed(BaroConfig::default(), 42);

        // At sea level, temperature should be around 15°C
        let reading = sensor.sample(0.0);
        assert!(
            (reading.temperature_c - 15.0).abs() < 1.0,
            "Temperature at sea level {} should be near 15°C",
            reading.temperature_c
        );

        // At 1000m, temperature should be around 8.5°C
        let reading = sensor.sample(1000.0);
        assert!(
            (reading.temperature_c - 8.5).abs() < 1.0,
            "Temperature at 1000m {} should be near 8.5°C",
            reading.temperature_c
        );
    }
}
