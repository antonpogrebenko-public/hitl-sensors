# hitl-sensors

Rust sensor noise models for HITL simulation. No I/O — pure math, WASM-compatible.

## Overview

Provides realistic sensor models with configurable noise:
- **IMU** — Accelerometer and gyroscope with bias drift
- **Barometer** — Pressure/altitude with temperature effects
- **GPS** — Position, velocity with delay and drift
- **Magnetometer** — Field vector with hard/soft iron distortion

## Usage

```rust
use hitl_sensors::{ImuSensor, GpsSensor, BaroSensor, MagSensor};
use hitl_sensors::{SensorsConfig, ImuConfig, GpsConfig};

// Create with defaults
let mut imu = ImuSensor::new();
let mut gps = GpsSensor::new();

// Or with custom config
let config = ImuConfig {
    gyro_noise_density: 0.0008,
    accel_noise_density: 0.006,
    gyro_bias_sigma: 0.0,  // Disable drift for HITL
    gyro_bias_tau: 1000.0,
};
let mut imu = ImuSensor::with_config(config);

// Sample sensors
let imu_reading = imu.sample(true_accel, true_gyro, dt);
let gps_reading = gps.sample(position_ned, velocity_ned, sim_time);
```

## Sensor Configs

### IMU (ImuConfig)
| Parameter | Default | HITL Recommended |
|-----------|---------|------------------|
| gyro_noise_density | 0.0008 | 0.0008 |
| accel_noise_density | 0.006 | 0.006 |
| gyro_bias_sigma | 0.0001 | **0.0** |
| gyro_bias_tau | 100.0 | 1000.0 |
| accel_bias_sigma | 0.003 | **0.0** |
| accel_bias_tau | 300.0 | 1000.0 |

### GPS (GpsConfig)
| Parameter | Default | HITL Recommended |
|-----------|---------|------------------|
| horizontal_noise_sigma | 1.5 | **0.1** |
| altitude_noise_sigma | 3.0 | **0.3** |
| velocity_noise_sigma | 0.1 | 0.05 |
| position_drift_sigma | 0.5 | **0.0** |
| delay_ms | 100.0 | 80.0 |
| update_rate_hz | 5.0 | **10.0** |

## Key Files

- `src/imu.rs` — IMU sensor with Gauss-Markov bias drift
- `src/gps.rs` — GPS with position noise, drift, and delay buffer
- `src/baro.rs` — Barometer with pressure noise
- `src/mag.rs` — Magnetometer with hard/soft iron effects
- `src/noise.rs` — GaussMarkov process, WhiteNoise utilities

## Building

```bash
cargo build
cargo test
```

## Gotchas

### Gyro bias drift causes "High Gyro Bias" in PX4
Set `gyro_bias_sigma: 0.0` to disable Gauss-Markov drift. PX4's EKF2 detects accumulated bias as a sensor fault.

### GPS noise causes position zig-zag
Set `horizontal_noise_sigma: 0.1` (10cm) instead of 1.5m for HITL. High noise causes EKF position jumps.

### PX4 rejects sensors with zero noise
Don't set noise to 0.0 — PX4's sensor validators detect "stuck" sensors. Use small realistic values.

### GPS position drift disabled for HITL
Set `position_drift_sigma: 0.0` to prevent slow position walk that triggers failsafes.
