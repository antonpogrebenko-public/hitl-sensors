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
    /// Correlation time of the dominant position error, in seconds.
    ///
    /// `horizontal_noise_sigma` / `altitude_noise_sigma` state a module's
    /// *total* accuracy, and in a real receiver that total is dominated by
    /// ionospheric, ephemeris and multipath terms that vary over minutes. This
    /// is the time constant of that slow component.
    pub noise_correlation_tau: f64,
    /// Portion of the total position sigma that is uncorrelated receiver
    /// jitter, as a fraction in `[0, 1]`. The remainder is carried by the
    /// correlated term, so the two together still sum to the stated accuracy.
    pub white_noise_fraction: f64,
}

impl Default for GpsConfig {
    fn default() -> Self {
        Self {
            position_drift_tau: 30.0,    // seconds
            position_drift_sigma: 0.06,  // m/s equivalent drift
            horizontal_noise_sigma: 1.5, // meters
            altitude_noise_sigma: 3.0,   // meters
            velocity_noise_sigma: 0.1,   // m/s
            delay_ms: 120.0,             // milliseconds
            update_rate_hz: 5.0,         // Hz
            // A stationary consumer receiver scatters a few tens of
            // centimetres between fixes while its absolute error wanders by
            // metres over minutes. 20% white against a 2-minute correlation
            // time reproduces that; both are model parameters, not datasheet
            // figures, which is why they are configurable rather than baked in.
            noise_correlation_tau: 120.0, // seconds
            white_noise_fraction: 0.2,    // of the total sigma
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
    /// The slow component of the module's stated accuracy, one per NED axis.
    correlated: [GaussMarkov; 3],
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
            // Altitude previously got an ad-hoc `altitude_noise_sigma * 0.1`
            // here, which made the vertical axis follow a different model from
            // the horizontal one *and* pushed total error past the configured
            // sigma. The accuracy decomposition below covers both axes on the
            // same terms, so this knob now means the same thing on all three.
            GaussMarkov::new(config.position_drift_tau, config.position_drift_sigma),
        ];

        // Split the stated accuracy into a small uncorrelated part and a large
        // slowly-varying one, preserving the total: sigma_w^2 + sigma_c^2 =
        // sigma^2. Feeding the whole figure in as white noise made every fix an
        // independent draw metres from the last, and an estimator differencing
        // those fixes sees vertical velocity that no aircraft could have — PX4
        // rejected arming with "vertical velocity unstable" on any build whose
        // GPS component carried real datasheet numbers.
        let fraction = config.white_noise_fraction.clamp(0.0, 1.0);
        let correlated_scale = (1.0 - fraction * fraction).sqrt();
        let totals = [
            config.horizontal_noise_sigma,
            config.horizontal_noise_sigma,
            config.altitude_noise_sigma,
        ];
        let correlated = std::array::from_fn(|i| {
            GaussMarkov::new(config.noise_correlation_tau, totals[i] * correlated_scale)
        });

        Self {
            config,
            drift,
            correlated,
            delay_buffer: VecDeque::new(),
            last_update_time: -1000.0, // Force first update
            last_output_time: -1000.0,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Convert NED position to lat/lon given reference point.
    fn ned_to_latlon(position_ned: &[f64; 3], ref_lat: f64, ref_lon: f64) -> (f64, f64, f64) {
        let ref_lat_rad = ref_lat.to_radians();

        // North offset to latitude change
        let lat = ref_lat + (position_ned[0] / EARTH_RADIUS_M).to_degrees();

        // East offset to longitude change (accounting for latitude)
        let lon = ref_lon + (position_ned[1] / (EARTH_RADIUS_M * ref_lat_rad.cos())).to_degrees();

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

            // Update both correlated processes: the explicit drift knob and
            // the slow component of the module's stated accuracy.
            let dt = update_period;
            for d in &mut self.drift {
                d.step(dt, &mut self.rng);
            }
            for c in &mut self.correlated {
                c.step(dt, &mut self.rng);
            }

            let fraction = self.config.white_noise_fraction.clamp(0.0, 1.0);
            let white_sigma = [
                self.config.horizontal_noise_sigma * fraction,
                self.config.horizontal_noise_sigma * fraction,
                self.config.altitude_noise_sigma * fraction,
            ];

            let mut noisy_position = *position_ned;
            for i in 0..3 {
                noisy_position[i] += self.drift[i].state() + self.correlated[i].state();

                let u1: f64 = self.rng.gen_range(0.0001..1.0);
                let u2: f64 = self.rng.gen();
                let (z, _) = box_muller(u1, u2);
                noisy_position[i] += white_sigma[i] * z;
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

            // Discard samples that a newer one has superseded, keeping the most
            // recent sample that is still old enough to output.
            //
            // The front is dropped only when the sample behind it has also aged
            // past the delay. Popping every sample at or below the target
            // instead leaves the front *newer* than the target, and the emit
            // check below can then never pass — so a reading only ever escaped
            // when the buffer happened to hold exactly one sample. That holds
            // while the delay is shorter than one update period, and fails
            // permanently once it is not: at 18 Hz with 120 ms of delay the
            // buffer never drops below three, GPS went silent, and PX4 refused
            // to arm with "ekf2 missing data".
            while self.delay_buffer.len() > 1 {
                let next_is_ready = self
                    .delay_buffer
                    .get(1)
                    .is_some_and(|next| next.time_s <= target_time);
                if next_is_ready {
                    self.delay_buffer.pop_front();
                } else {
                    break;
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
        for c in &mut self.correlated {
            c.reset();
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
        assert!(!gps_times.is_empty(), "Should have received GPS readings");

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

#[cfg(test)]
mod delay_buffer_tests {
    use super::*;

    /// Drive the model at 400 Hz for `secs` and count how many readings it emits.
    fn emitted_over(rate_hz: f64, delay_ms: f64, secs: f64) -> usize {
        let mut gps = GpsSensor::with_config(GpsConfig {
            update_rate_hz: rate_hz,
            delay_ms,
            horizontal_noise_sigma: 0.0,
            altitude_noise_sigma: 0.0,
            velocity_noise_sigma: 0.0,
            position_drift_sigma: 0.0,
            ..GpsConfig::default()
        });

        let mut count = 0;
        let ticks = (secs * 400.0) as usize;
        for i in 1..=ticks {
            let t = i as f64 / 400.0;
            if gps
                .sample(&[0.0, 0.0, 0.0], &[0.0, 0.0, 0.0], 40.0, -105.0, t)
                .is_some()
            {
                count += 1;
            }
        }
        count
    }

    #[test]
    fn emits_at_its_configured_rate_when_the_delay_is_shorter_than_a_period() {
        // The default shape: 10 Hz with 80 ms of delay.
        let n = emitted_over(10.0, 80.0, 3.0);
        assert!(
            (25..=32).contains(&n),
            "expected ~30 readings over 3 s at 10 Hz, got {n}"
        );
    }

    #[test]
    fn keeps_emitting_when_the_delay_exceeds_one_update_period() {
        // A real profile from the component database: 18 Hz with 120 ms delay,
        // so the delay spans more than two update periods and the buffer never
        // drains to a single sample.
        //
        // The trim loop used to pop every sample at or below the target, then
        // ask whether the remaining front was at or below the target — which it
        // could not be. Emission was only possible when the buffer happened to
        // hold exactly one sample, so this configuration went permanently
        // silent, PX4 saw no GPS at all, and the vehicle would not arm.
        let n = emitted_over(18.0, 120.0, 3.0);
        assert!(
            n > 0,
            "GPS went permanently silent at 18 Hz with 120 ms delay"
        );
        assert!(
            (45..=58).contains(&n),
            "expected ~54 readings over 3 s at 18 Hz, got {n}"
        );
    }

    #[test]
    fn keeps_emitting_across_a_range_of_real_profiles() {
        for (rate, delay) in [(5.0, 200.0), (10.0, 100.0), (18.0, 120.0), (25.0, 250.0)] {
            let n = emitted_over(rate, delay, 3.0);
            assert!(
                n > 0,
                "GPS silent at {rate} Hz with {delay} ms delay -- delay >= period must still emit"
            );
        }
    }

    #[test]
    fn the_emitted_sample_is_actually_delayed() {
        // The delay is the point of the buffer: a reading emitted now must
        // describe where the vehicle was `delay_ms` ago, not where it is.
        let mut gps = GpsSensor::with_config(GpsConfig {
            update_rate_hz: 10.0,
            delay_ms: 200.0,
            horizontal_noise_sigma: 0.0,
            altitude_noise_sigma: 0.0,
            velocity_noise_sigma: 0.0,
            position_drift_sigma: 0.0,
            ..GpsConfig::default()
        });

        // Climb steadily, so altitude encodes the time a sample was taken.
        let mut last_alt = None;
        for i in 1..=1200 {
            let t = i as f64 / 400.0;
            let down = -t; // 1 m/s climb
            if let Some(r) = gps.sample(&[0.0, 0.0, down], &[0.0, 0.0, -1.0], 40.0, -105.0, t) {
                last_alt = Some((t, r.alt as f64));
            }
        }
        let (t, alt) = last_alt.expect("some reading was emitted");
        // Altitude equals the time of the sample it came from, so `t - alt` is
        // the age of that sample: at least the configured delay.
        assert!(
            t - alt >= 0.19,
            "sample was {:.3} s old, expected at least the 0.2 s delay",
            t - alt
        );
    }
}

#[cfg(test)]
mod noise_model_tests {
    use super::*;

    /// The GPS profile this failed on in the field: an 18 Hz module whose
    /// datasheet accuracy is 1.5 m horizontal / 3.0 m vertical.
    fn field_profile() -> GpsConfig {
        GpsConfig {
            horizontal_noise_sigma: 1.5,
            altitude_noise_sigma: 3.0,
            velocity_noise_sigma: 0.1,
            delay_ms: 120.0,
            update_rate_hz: 18.0,
            position_drift_sigma: 0.0,
            position_drift_tau: 1000.0,
            ..Default::default()
        }
    }

    /// Altitude error of each emitted reading, holding the vehicle still.
    fn altitude_errors(config: GpsConfig, seconds: f64) -> Vec<f64> {
        let mut sensor = GpsSensor::with_config_and_seed(config, 7);
        let truth = [0.0, 0.0, -100.0]; // 100 m up, stationary
        let mut errors = Vec::new();
        let step = 1.0 / 400.0;
        let mut t = 0.0;
        while t < seconds {
            if let Some(r) = sensor.sample(&truth, &[0.0; 3], 40.0, -105.0, t) {
                errors.push(r.alt as f64 - 100.0);
            }
            t += step;
        }
        errors
    }

    fn std_dev(xs: &[f64]) -> f64 {
        let mean = xs.iter().sum::<f64>() / xs.len() as f64;
        (xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / xs.len() as f64).sqrt()
    }

    /// Standard deviation of the change between consecutive readings.
    fn jitter(xs: &[f64]) -> f64 {
        let diffs: Vec<f64> = xs.windows(2).map(|w| w[1] - w[0]).collect();
        std_dev(&diffs)
    }

    /// Real GNSS position error is dominated by slowly-varying atmospheric,
    /// ephemeris and multipath terms: a receiver held still wanders over
    /// minutes, it does not hop metres between fixes 55 ms apart. Applying the
    /// datasheet sigma as per-sample white noise is what the EKF sees as
    /// enormous vertical velocity, and it is why PX4 refused to arm with
    /// "vertical velocity unstable" on any build with a GPS component
    /// selected.
    ///
    /// Sample-to-sample jitter must therefore be a small fraction of the total
    /// error, not sqrt(2) times it.
    #[test]
    fn consecutive_fixes_do_not_jump_the_full_datasheet_sigma() {
        let errors = altitude_errors(field_profile(), 400.0);
        assert!(errors.len() > 1000, "expected a long run, got {}", errors.len());

        let jitter = jitter(&errors);
        assert!(
            jitter < 1.2,
            "consecutive altitude fixes differ by sigma {jitter:.3} m; a 3.0 m module \
             must not hop that far between fixes 55 ms apart — the error has to be \
             correlated, not white"
        );
    }

    /// The decomposition must not quietly shrink the module's stated accuracy:
    /// the total steady-state error still has to match what the datasheet says.
    #[test]
    fn total_error_still_matches_the_configured_sigma() {
        let errors = altitude_errors(field_profile(), 2000.0);
        let total = std_dev(&errors);
        assert!(
            (total - 3.0).abs() < 1.0,
            "total altitude error sigma {total:.3} m should stay near the configured 3.0 m"
        );
    }

    /// Horizontal used the configured sigma as pure white noise while altitude
    /// got an ad-hoc extra 10% correlated term. Both axes must follow the same
    /// model, or a build's horizontal behaviour cannot be reasoned about from
    /// its vertical behaviour.
    #[test]
    fn horizontal_is_correlated_on_the_same_terms_as_altitude() {
        let config = field_profile();
        let mut sensor = GpsSensor::with_config_and_seed(config, 11);
        let truth = [0.0, 0.0, -100.0];
        let mut norths = Vec::new();
        let step = 1.0 / 400.0;
        let mut t = 0.0;
        while t < 400.0 {
            if let Some(r) = sensor.sample(&truth, &[0.0; 3], 40.0, -105.0, t) {
                norths.push((r.lat - 40.0).to_radians() * EARTH_RADIUS_M);
            }
            t += step;
        }
        let jitter = jitter(&norths);
        assert!(
            jitter < 0.6,
            "consecutive north fixes differ by sigma {jitter:.3} m; a 1.5 m module \
             must not hop that far between fixes"
        );
    }
}
