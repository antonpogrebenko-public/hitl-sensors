# Changelog

## 0.3.0 — 2026-09-04

Sensor sampling costs half what it did. Measured against a same-session control
arm (stash, measure, restore, measure) rather than a stored baseline:

| bench | before | after |
|---|---|---|
| `imu_sample` | 401.45 ns | 206.42 ns |
| `sample_all` | 612.25 ns | 306.86 ns |
| `sample_all_x400` | 223.79 us | 109.29 us |

- **Box-Muller was discarding half its output.** The transform produces two
  independent normal variates per pair of uniforms; the previous code returned
  one and threw the other away, so every sample paid for a `sqrt`, a `ln` and
  two trig calls. `NormalSource` now keeps the spare and returns it on the next
  call.

- **`GaussMarkov` recomputed its coefficients every step.** `alpha` and
  `noise_sigma` are functions of `dt`, and `dt` does not change between ticks.
  They are cached and recomputed only when `dt` actually differs, with
  `cached_dt` initialised to `f64::NAN` so the first call always computes.

This is a minor rather than a patch bump because the *order* in which random
draws are consumed changes: a seeded run no longer reproduces the previous
sequence value-for-value. The distributions are identical.

Criterion benches added under `benches/sample.rs`.
