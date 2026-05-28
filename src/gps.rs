//! GPS sensor simulation with delay buffer and Gauss-Markov drift.

use crate::noise::{box_muller, GaussMarkov};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::VecDeque;

/// Earth radius in meters for coordinate conversion
const EARTH_RADIUS_M: f64 = 6371000.0;

/// Configuration for GPS sensor.
#[derive(Debug, Clone)]
pub struct GpsConfig {
    /// Position drift time constant in seconds
    pub position_drift_tau: f64,
    /// Position drift sigma in meters
    pub position_drift_sigma: f64,
    /// Horizontal position noise standard deviation in meters
    pub horizontal_noise_sigma: f64,
    /// Altitude noise standard deviation in meters
    pub altitude_noise_sigma: f64,
    /// Velocity noise standard deviation in m/s
    pub velocity_noise_sigma: f64,
    /// Measurement delay in milliseconds
    pub delay_ms: f64,
    /// GPS update rate in Hz
    pub update_rate_hz: f64,
}

impl Default for GpsConfig {
    fn default() -> Self {
        Self {
            position_drift_tau: 30.0,       // seconds
            position_drift_sigma: 0.06,     // m/s equivalent drift
            horizontal_noise_sigma: 1.5,    // meters
            altitude_noise_sigma: 3.0,      // meters
            velocity_noise_sigma: 0.1,      // m/s
            delay_ms: 120.0,                // milliseconds
            update_rate_hz: 5.0,            // Hz
        }
    }
}

/// Internal struct for delay buffer
#[derive(Clone)]
struct GpsSample {
    time_s: f64,
    position_ned: [f64; 3],
    velocity_ned: [f64; 3],
    ref_lat: f64,
    ref_lon: f64,
}

/// Simulated GPS sensor.
pub struct GpsSensor {
    config: GpsConfig,
    drift: [GaussMarkov; 3],
    delay_buffer: VecDeque<GpsSample>,
    last_update_time: f64,
    last_output_time: f64,
    rng: StdRng,
}

/// GPS reading with position, velocity, and quality indicators.
#[derive(Debug, Clone, Copy)]
pub struct GpsReading {
    /// Latitude in degrees
    pub lat: f64,
    /// Longitude in degrees
    pub lon: f64,
    /// Altitude in meters AGL (height above launch/reference point, i.e. `-ned_down`).
    /// Does NOT include reference_alt — callers must add reference_alt to get MSL.
    pub alt: f32,
    /// Velocity North in m/s
    pub vel_n: f32,
    /// Velocity East in m/s
    pub vel_e: f32,
    /// Velocity Down in m/s
    pub vel_d: f32,
    /// Horizontal Dilution of Precision
    pub hdop: f32,
    /// Number of satellites
    pub satellites: u8,
}

impl GpsSensor {
    /// Create a new GPS sensor with default configuration.
    pub fn new() -> Self {
        Self::with_config(GpsConfig::default())
    }

    /// Create a new GPS sensor with custom configuration.
    pub fn with_config(config: GpsConfig) -> Self {
        Self::with_config_and_seed(config, rand::random())
    }

    /// Create a new GPS sensor with custom configuration and seed.
    pub fn with_config_and_seed(config: GpsConfig, seed: u64) -> Self {
        let drift = [
            GaussMarkov::new(config.position_drift_tau, config.position_drift_sigma),
            GaussMarkov::new(config.position_drift_tau, config.position_drift_sigma),
            GaussMarkov::new(
                config.position_drift_tau,
                config.altitude_noise_sigma * 0.1,
            ), // Smaller drift for altitude
        ];

        Self {
            config,
            drift,
            delay_buffer: VecDeque::new(),
            last_update_time: -1000.0, // Force first update
            last_output_time: -1000.0,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Convert NED position to lat/lon given reference point.
    fn ned_to_latlon(
        position_ned: &[f64; 3],
        ref_lat: f64,
        ref_lon: f64,
    ) -> (f64, f64, f64) {
        let ref_lat_rad = ref_lat.to_radians();

        // North offset to latitude change
        let lat = ref_lat + (position_ned[0] / EARTH_RADIUS_M).to_degrees();

        // East offset to longitude change (accounting for latitude)
        let lon =
            ref_lon + (position_ned[1] / (EARTH_RADIUS_M * ref_lat_rad.cos())).to_degrees();

        // Down to altitude (negative down = positive altitude)
        let alt = -position_ned[2];

        (lat, lon, alt)
    }

    /// Sample the GPS sensor.
    ///
    /// # Arguments
    /// * `position_ned` - True position in NED frame relative to reference (meters)
    /// * `velocity_ned` - True velocity in NED frame (m/s)
    /// * `ref_lat` - Reference latitude in degrees
    /// * `ref_lon` - Reference longitude in degrees
    /// * `time_s` - Current simulation time in seconds
    ///
    /// # Returns
    /// GPS reading if it's time for an update, None otherwise
    pub fn sample(
        &mut self,
        position_ned: &[f64; 3],
        velocity_ned: &[f64; 3],
        ref_lat: f64,
        ref_lon: f64,
        time_s: f64,
    ) -> Option<GpsReading> {
        let update_period = 1.0 / self.config.update_rate_hz;
        let delay_s = self.config.delay_ms / 1000.0;

        // Add sample to delay buffer at GPS rate
        if time_s - self.last_update_time >= update_period {
            self.last_update_time = time_s;

            // Update drift processes
            let dt = update_period;
            for d in &mut self.drift {
                d.step(dt, &mut self.rng);
            }

            // Add noisy sample to buffer
            let mut noisy_position = *position_ned;
            for i in 0..3 {
                noisy_position[i] += self.drift[i].state();

                // Add white noise
                let u1: f64 = self.rng.gen_range(0.0001..1.0);
                let u2: f64 = self.rng.gen();
                let (z, _) = box_muller(u1, u2);

                if i == 2 {
                    // Altitude noise (configurable)
                    noisy_position[i] += self.config.altitude_noise_sigma * z;
                } else {
                    // Horizontal noise (configurable)
                    noisy_position[i] += self.config.horizontal_noise_sigma * z;
                }
            }

            // Add velocity noise (configurable)
            let mut noisy_velocity = *velocity_ned;
            for v in &mut noisy_velocity {
                let u1: f64 = self.rng.gen_range(0.0001..1.0);
                let u2: f64 = self.rng.gen();
                let (z, _) = box_muller(u1, u2);
                *v += self.config.velocity_noise_sigma * z;
            }

            self.delay_buffer.push_back(GpsSample {
                time_s,
                position_ned: noisy_position,
                velocity_ned: noisy_velocity,
                ref_lat,
                ref_lon,
            });
        }

        // Output delayed sample at GPS rate
        if time_s - self.last_output_time >= update_period {
            // Find sample that should be output now (with delay)
            let target_time = time_s - delay_s;

            // Remove old samples and find the right one
            while self.delay_buffer.len() > 1 {
                if let Some(front) = self.delay_buffer.front() {
                    if front.time_s <= target_time {
                        self.delay_buffer.pop_front();
                    } else {
                        break;
                    }
                }
            }

            // Cap buffer size to avoid unbounded growth during startup (max 20 samples)
            while self.delay_buffer.len() > 20 {
                self.delay_buffer.pop_front();
            }

            if let Some(sample) = self.delay_buffer.front() {
                if sample.time_s <= target_time {
                    self.last_output_time = time_s;

                    let (lat, lon, alt) =
                        Self::ned_to_latlon(&sample.position_ned, sample.ref_lat, sample.ref_lon);

                    // Generate realistic HDOP and satellite count
                    let hdop = 0.8 + self.rng.gen::<f32>() * 0.4; // 0.8-1.2
                    let satellites = 8 + (self.rng.gen::<u8>() % 6); // 8-13

                    return Some(GpsReading {
                        lat,
                        lon,
                        alt: alt as f32,
                        vel_n: sample.velocity_ned[0] as f32,
                        vel_e: sample.velocity_ned[1] as f32,
                        vel_d: sample.velocity_ned[2] as f32,
                        hdop,
                        satellites,
                    });
                }
            }
        }

        None
    }

    /// Reset the sensor state.
    pub fn reset(&mut self) {
        for d in &mut self.drift {
            d.reset();
        }
        self.delay_buffer.clear();
        self.last_update_time = -1000.0;
        self.last_output_time = -1000.0;
    }

    /// Get the current configuration.
    pub fn config(&self) -> &GpsConfig {
        &self.config
    }

    /// Get the delay in seconds.
    pub fn delay_s(&self) -> f64 {
        self.config.delay_ms / 1000.0
    }
}

impl Default for GpsSensor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gps_delay() {
        let config = GpsConfig {
            delay_ms: 120.0,
            update_rate_hz: 5.0,
            ..Default::default()
        };
        let mut sensor = GpsSensor::with_config_and_seed(config, 42);

        let ref_lat = 40.0;
        let ref_lon = -105.0;
        let velocity = [0.0, 0.0, 0.0];

        // Record when we set position and when GPS reports it
        let mut position_times: Vec<(f64, [f64; 3])> = Vec::new();
        let mut gps_times: Vec<(f64, GpsReading)> = Vec::new();

        let dt = 0.001; // 1ms simulation step
        let mut time = 0.0;

        // Move north at 10 m/s for 1 second
        while time < 2.0 {
            let position = [time * 10.0, 0.0, 0.0]; // Moving north
            position_times.push((time, position));

            if let Some(reading) = sensor.sample(&position, &velocity, ref_lat, ref_lon, time) {
                gps_times.push((time, reading));
            }

            time += dt;
        }

        // Verify we got GPS readings
        assert!(
            !gps_times.is_empty(),
            "Should have received GPS readings"
        );

        // Check that GPS readings are delayed by ~120ms
        // The GPS position at time T should correspond to true position at T - 0.12s
        for (gps_time, reading) in &gps_times {
            if *gps_time < 0.2 {
                continue; // Skip initial transient
            }

            // Convert GPS lat back to north position
            let gps_north = (reading.lat - ref_lat).to_radians() * EARTH_RADIUS_M;

            // Expected position ~120ms earlier
            let expected_time = gps_time - 0.12;
            let expected_north = expected_time * 10.0;

            // Allow for noise and discretization (within 5m)
            let error = (gps_north - expected_north).abs();
            assert!(
                error < 5.0,
                "GPS at t={:.3}s reported north={:.1}m, expected ~{:.1}m (delay error {:.1}m)",
                gps_time,
                gps_north,
                expected_north,
                error
            );
        }
    }

    #[test]
    fn test_gps_update_rate() {
        let config = GpsConfig {
            update_rate_hz: 5.0,
            ..Default::default()
        };
        let mut sensor = GpsSensor::with_config_and_seed(config, 42);

        let position = [0.0, 0.0, 0.0];
        let velocity = [0.0, 0.0, 0.0];
        let ref_lat = 40.0;
        let ref_lon = -105.0;

        let dt = 0.001;
        let mut time = 0.0;
        let mut reading_count = 0;

        while time < 1.0 {
            if sensor
                .sample(&position, &velocity, ref_lat, ref_lon, time)
                .is_some()
            {
                reading_count += 1;
            }
            time += dt;
        }

        // At 5 Hz, should get ~5 readings per second (allowing for delay startup)
        assert!(
            reading_count >= 4 && reading_count <= 6,
            "Expected ~5 readings at 5Hz, got {}",
            reading_count
        );
    }

    #[test]
    fn test_gps_coordinate_conversion() {
        // Test NED to lat/lon conversion
        let ref_lat = 40.0;
        let ref_lon = -105.0;

        // 1000m north should increase latitude
        let (lat, lon, _) = GpsSensor::ned_to_latlon(&[1000.0, 0.0, 0.0], ref_lat, ref_lon);
        assert!(lat > ref_lat, "Moving north should increase latitude");
        assert!(
            (lon - ref_lon).abs() < 0.0001,
            "Moving north should not change longitude"
        );

        // 1000m east should increase longitude
        let (lat, lon, _) = GpsSensor::ned_to_latlon(&[0.0, 1000.0, 0.0], ref_lat, ref_lon);
        assert!(
            (lat - ref_lat).abs() < 0.0001,
            "Moving east should not change latitude"
        );
        assert!(lon > ref_lon, "Moving east should increase longitude");

        // Down should decrease altitude
        let (_, _, alt) = GpsSensor::ned_to_latlon(&[0.0, 0.0, 100.0], ref_lat, ref_lon);
        assert!(alt < 0.0, "Positive down should give negative altitude");
    }

    #[test]
    fn test_gps_noise_bounds() {
        let mut sensor = GpsSensor::with_config_and_seed(GpsConfig::default(), 42);

        let position = [0.0, 0.0, 0.0];
        let velocity = [0.0, 0.0, 0.0];
        let ref_lat = 40.0;
        let ref_lon = -105.0;

        let dt = 0.001;
        let mut time = 0.0;
        let mut readings = Vec::new();

        while time < 10.0 {
            if let Some(reading) = sensor.sample(&position, &velocity, ref_lat, ref_lon, time) {
                readings.push(reading);
            }
            time += dt;
        }

        // Check that position noise is reasonable (within 10m horizontally)
        for reading in &readings {
            let north_error = (reading.lat - ref_lat).to_radians() * EARTH_RADIUS_M;
            let east_error =
                (reading.lon - ref_lon).to_radians() * EARTH_RADIUS_M * ref_lat.to_radians().cos();

            assert!(
                north_error.abs() < 10.0,
                "North error {} should be < 10m",
                north_error
            );
            assert!(
                east_error.abs() < 10.0,
                "East error {} should be < 10m",
                east_error
            );
            assert!(
                reading.alt.abs() < 15.0,
                "Altitude error {} should be < 15m",
                reading.alt
            );
        }
    }
}
