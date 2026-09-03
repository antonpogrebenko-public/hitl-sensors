//! Sensor sampling benchmarks.
//!
//! The daemon samples the IMU on every 400 Hz tick and the remaining
//! sensors on dividers below it. These figures are what the sensor models
//! cost out of the 2500 µs tick budget.
//!
//! Seeded throughout. An unseeded constructor calls `rand::random()`, which
//! would make each run start from a different point in the noise process —
//! measurable as variance that has nothing to do with the code.
//!
//! Run: cargo bench -p hitl-sensors

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hitl_sensors::{ImuConfig, ImuSensor, Sensors, SensorsConfig, UnitQuaternion};

/// Level hover: 1g in body -Z, no rotation.
const TRUE_ACCEL: [f64; 3] = [0.0, 0.0, -9.80665];
const TRUE_GYRO: [f64; 3] = [0.0, 0.0, 0.0];
const DT: f64 = 1.0 / 400.0;

fn bench_imu_sample(c: &mut Criterion) {
    let mut imu = ImuSensor::with_config_and_seed(ImuConfig::default(), 0xB1A5);
    c.bench_function("imu_sample", |b| {
        b.iter(|| {
            imu.sample(
                black_box(&TRUE_ACCEL),
                black_box(&TRUE_GYRO),
                black_box(DT),
            )
        })
    });
}

fn bench_sample_all(c: &mut Criterion) {
    let mut sensors = Sensors::with_config_and_seed(SensorsConfig::default(), 0xB1A5);
    let position = [0.0, 0.0, -10.0];
    let velocity = [0.0, 0.0, 0.0];
    let attitude = UnitQuaternion::identity();
    let mut time_s = 0.0;

    c.bench_function("sample_all", |b| {
        b.iter(|| {
            time_s += DT;
            sensors.sample_all(
                black_box(&TRUE_ACCEL),
                black_box(&TRUE_GYRO),
                black_box(10.0),
                black_box(&position),
                black_box(&velocity),
                black_box(47.3977),
                black_box(8.5456),
                black_box(&attitude),
                black_box(time_s),
                black_box(DT),
            )
        })
    });
}

/// One second of sampling at the daemon's rate.
///
/// `sample_all` advances `time_s`, and the GPS and barometer models gate
/// their own output on it, so a fixed timestamp would bench a path that
/// never emits. This walks real time forward the way the loop does.
fn bench_sample_all_x400(c: &mut Criterion) {
    let position = [0.0, 0.0, -10.0];
    let velocity = [0.0, 0.0, 0.0];
    let attitude = UnitQuaternion::identity();

    c.bench_function("sample_all_x400", |b| {
        b.iter(|| {
            let mut sensors = Sensors::with_config_and_seed(SensorsConfig::default(), 0xB1A5);
            let mut time_s = 0.0;
            for _ in 0..400 {
                time_s += DT;
                black_box(sensors.sample_all(
                    &TRUE_ACCEL,
                    &TRUE_GYRO,
                    10.0,
                    &position,
                    &velocity,
                    47.3977,
                    8.5456,
                    &attitude,
                    time_s,
                    DT,
                ));
            }
        })
    });
}

criterion_group!(
    benches,
    bench_imu_sample,
    bench_sample_all,
    bench_sample_all_x400
);
criterion_main!(benches);
