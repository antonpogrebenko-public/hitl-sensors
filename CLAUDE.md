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
| accel_bias_sigma | 0.0 | **0.0** |
| accel_bias_tau | 1000.0 | 1000.0 |

`ImuSensor` holds an internal `accel_bias: [GaussMarkov; 3]` state (one per axis). Set `accel_bias_sigma: 0.0` to disable drift for HITL — same rationale as gyro bias.

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

### Accel bias drift can corrupt EKF attitude
Set `accel_bias_sigma: 0.0` for HITL. The per-axis `[GaussMarkov; 3]` accel bias is added to each accelerometer sample; accumulated drift causes EKF attitude divergence in long flights.

### GPS noise causes position zig-zag
Set `horizontal_noise_sigma: 0.1` (10cm) instead of 1.5m for HITL. High noise causes EKF position jumps.

### PX4 rejects sensors with zero noise
Don't set noise to 0.0 — PX4's sensor validators detect "stuck" sensors. Use small realistic values.

### GPS position drift disabled for HITL
Set `position_drift_sigma: 0.0` to prevent slow position walk that triggers failsafes.

## Approach
- Read existing files before writing. Don't re-read unless changed.
- Thorough in reasoning, concise in output.
- No sycophantic openers or closing fluff.
- No emojis or em-dashes.
- Do not guess APIs, versions, flags, commit SHAs, or package names. Verify by reading code or docs before asserting.

### GPS delay is active for the entire flight
The `|| time_s > delay_s` bypass was removed — the delay model (ring buffer, capped at 20 samples) is enforced from t=0 through the full flight. Do not expect instantaneous GPS fix on startup; the first valid reading arrives after `delay_ms` has elapsed.

### GPS `GpsReading.alt` is AGL, not MSL
`alt` is altitude above the launch point (above ground level). Callers that need MSL altitude must add `reference_alt` themselves. This matches what PX4 expects for local-NED-origin alt.

### Baro altitude must be in [-500, 11000] m
`isa_pressure_pa` and `isa_temperature_k` clamp altitude to the [-500 m, 11000 m] range before ISA calculations. Values outside this range (e.g. from a corrupt physics state) previously produced NaN pressure/temperature via negative-base power. The clamp prevents NaN propagation into the EKF.

### Sensors::reset() resets all four sensors
`reset()` now calls `imu.reset()`, `gps.reset()`, `baro.reset()`, and `mag.reset()`. Previously baro and mag were silently skipped, leaving stale state after a simulation restart.
