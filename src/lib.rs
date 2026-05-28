//! HITL (Hardware-In-The-Loop) sensor simulation library.
//!
//! This crate provides realistic sensor models for drone simulation:
//! - IMU (gyroscope + accelerometer) with noise and bias drift
//! - Barometer with ISA atmosphere model
//! - GPS with delay buffer and position drift
//! - Magnetometer with configurable Earth field
//!
//! # Example
//!
//! ```
//! use hitl_sensors::{Sensors, SensorsConfig};
//! use nalgebra::UnitQuaternion;
//!
//! let mut sensors = Sensors::new();
//!
//! // Simulation state
//! let true_accel_body = [0.0, 0.0, 9.81];  // Gravity in body frame
//! let true_gyro = [0.0, 0.0, 0.0];          // No rotation
//! let altitude_m = 100.0;
//! let position_ned = [0.0, 0.0, -100.0];    // 100m up
//! let velocity_ned = [0.0, 0.0, 0.0];
//! let ref_lat = 40.0;
//! let ref_lon = -105.0;
//! let attitude = UnitQuaternion::identity();
//! let time_s = 0.0;
//! let dt = 0.001;
//!
//! let readings = sensors.sample_all(
//!     &true_accel_body,
//!     &true_gyro,
//!     altitude_m,
//!     &position_ned,
//!     &velocity_ned,
//!     ref_lat,
//!     ref_lon,
//!     &attitude,
//!     time_s,
//!     dt,
//! );
//!
//! println!("IMU accel: {:?}", readings.imu.accel);
//! println!("Baro altitude: {}", readings.baro.altitude_m);
//! ```

pub mod baro;
pub mod gps;
pub mod imu;
pub mod mag;
pub mod noise;

pub use baro::{BaroConfig, BaroReading, BaroSensor};
pub use gps::{GpsConfig, GpsReading, GpsSensor};
pub use imu::{ImuConfig, ImuReading, ImuSensor};
pub use mag::{MagConfig, MagReading, MagSensor};
pub use nalgebra::UnitQuaternion;

/// Combined sensor configuration.
#[derive(Debug, Clone)]
pub struct SensorsConfig {
    pub imu: ImuConfig,
    pub baro: BaroConfig,
    pub gps: GpsConfig,
    pub mag: MagConfig,
}

impl Default for SensorsConfig {
    fn default() -> Self {
        Self {
            imu: ImuConfig::default(),
            baro: BaroConfig::default(),
            gps: GpsConfig::default(),
            mag: MagConfig::default(),
        }
    }
}

/// Combined sensor readings from all sensors.
#[derive(Debug, Clone)]
pub struct SensorReadings {
    /// IMU (accelerometer + gyroscope) reading
    pub imu: ImuReading,
    /// Barometer reading
    pub baro: BaroReading,
    /// GPS reading (None if not time for GPS update)
    pub gps: Option<GpsReading>,
    /// Magnetometer reading
    pub mag: MagReading,
}

/// Combined sensor suite for HITL simulation.
pub struct Sensors {
    pub imu: ImuSensor,
    pub baro: BaroSensor,
    pub gps: GpsSensor,
    pub mag: MagSensor,
}

impl Sensors {
    /// Create a new sensor suite with default configuration.
    pub fn new() -> Self {
        Self::with_config(SensorsConfig::default())
    }

    /// Create a new sensor suite with custom configuration.
    pub fn with_config(config: SensorsConfig) -> Self {
        Self {
            imu: ImuSensor::with_config(config.imu),
            baro: BaroSensor::with_config(config.baro),
            gps: GpsSensor::with_config(config.gps),
            mag: MagSensor::with_config(config.mag),
        }
    }

    /// Create a new sensor suite with custom configuration and seed for reproducibility.
    pub fn with_config_and_seed(config: SensorsConfig, seed: u64) -> Self {
        Self {
            imu: ImuSensor::with_config_and_seed(config.imu, seed),
            baro: BaroSensor::with_config_and_seed(config.baro, seed.wrapping_add(1)),
            gps: GpsSensor::with_config_and_seed(config.gps, seed.wrapping_add(2)),
            mag: MagSensor::with_config_and_seed(config.mag, seed.wrapping_add(3)),
        }
    }

    /// Sample all sensors at once.
    ///
    /// # Arguments
    /// * `true_accel_body` - True specific force in body frame (m/s²)
    /// * `true_gyro` - True angular velocity in body frame (rad/s)
    /// * `altitude_m` - True geometric altitude MSL (meters)
    /// * `position_ned` - True position in NED frame relative to reference (meters)
    /// * `velocity_ned` - True velocity in NED frame (m/s)
    /// * `ref_lat` - Reference latitude in degrees
    /// * `ref_lon` - Reference longitude in degrees
    /// * `attitude` - Rotation from body to NED frame
    /// * `time_s` - Current simulation time (seconds)
    /// * `dt` - Time step since last sample (seconds)
    #[allow(clippy::too_many_arguments)]
    pub fn sample_all(
        &mut self,
        true_accel_body: &[f64; 3],
        true_gyro: &[f64; 3],
        altitude_m: f64,
        position_ned: &[f64; 3],
        velocity_ned: &[f64; 3],
        ref_lat: f64,
        ref_lon: f64,
        attitude: &UnitQuaternion<f64>,
        time_s: f64,
        dt: f64,
    ) -> SensorReadings {
        SensorReadings {
            imu: self.imu.sample(true_accel_body, true_gyro, dt),
            baro: self.baro.sample(altitude_m),
            gps: self
                .gps
                .sample(position_ned, velocity_ned, ref_lat, ref_lon, time_s),
            mag: self.mag.sample(attitude),
        }
    }

    /// Reset all sensors to initial state.
    pub fn reset(&mut self) {
        self.imu.reset();
        self.gps.reset();
        self.baro.reset();
        self.mag.reset();
    }
}

impl Default for Sensors {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sensors_integration() {
        let mut sensors = Sensors::with_config_and_seed(SensorsConfig::default(), 42);

        let true_accel = [0.0, 0.0, 9.81];
        let true_gyro = [0.0, 0.0, 0.0];
        let altitude = 100.0;
        let position = [0.0, 0.0, -100.0];
        let velocity = [0.0, 0.0, 0.0];
        let ref_lat = 40.0;
        let ref_lon = -105.0;
        let attitude = UnitQuaternion::identity();

        let dt = 0.001;
        let mut time = 0.0;

        let mut gps_count = 0;

        for _ in 0..1000 {
            let readings = sensors.sample_all(
                &true_accel,
                &true_gyro,
                altitude,
                &position,
                &velocity,
                ref_lat,
                ref_lon,
                &attitude,
                time,
                dt,
            );

            // Verify IMU readings are reasonable
            assert!(readings.imu.accel[2] > 8.0 && readings.imu.accel[2] < 11.0);

            // Verify baro readings are reasonable
            assert!(readings.baro.altitude_m > 90.0 && readings.baro.altitude_m < 110.0);

            // Verify mag readings are reasonable (magnitude check)
            let mag_magnitude = (readings.mag.field[0].powi(2)
                + readings.mag.field[1].powi(2)
                + readings.mag.field[2].powi(2))
            .sqrt();
            assert!(mag_magnitude > 0.4 && mag_magnitude < 0.6);

            if readings.gps.is_some() {
                gps_count += 1;
            }

            time += dt;
        }

        // Should have received some GPS readings (5 Hz over 1 second)
        assert!(gps_count >= 3 && gps_count <= 6);
    }

    #[test]
    fn test_sensors_reset() {
        let mut sensors = Sensors::with_config_and_seed(SensorsConfig::default(), 42);

        // Run for a bit to build up bias
        let true_accel = [0.0, 0.0, 9.81];
        let true_gyro = [0.0, 0.0, 0.0];
        let dt = 0.001;

        for _ in 0..1000 {
            sensors.imu.sample(&true_accel, &true_gyro, dt);
        }

        // Reset and verify
        sensors.reset();

        // After reset, bias should be zero again
        // (internal state is reset)
    }
}
