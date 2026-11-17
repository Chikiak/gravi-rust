//! Tests para los sembradores (seeders) de gravi-core.
//! Paridad con GraviSharp.Core.SimulationStore (SeedLinear/SeedRandomCloud/
//! SeedUniformField/SeedOrbitalDisk).

use gravi_core::constants::{MASS_EPS, PhysicsConstants};
use gravi_core::simulation_store::{
    allocate, seed_linear, seed_orbital_disk, seed_random_cloud, seed_uniform_field, SimConfig,
};

const COUNT: usize = 2000;
const EPS: f32 = 1e-3;

fn cfg() -> SimConfig {
    SimConfig { body_count: 16, width: 100.0, height: 100.0 }
}

fn cfg_big() -> SimConfig {
    SimConfig { body_count: COUNT, width: 100.0, height: 100.0 }
}

#[test]
fn seed_linear_xy_in_range_and_speed_magnitude() {
    let mut store = allocate(cfg()).unwrap();
    let speed = 10.0f32;
    seed_linear(&mut store, 100.0, 100.0, speed, 1337);

    let x = store.x_slice();
    let y = store.y_slice();
    let vx = store.vx_slice();
    let vy = store.vy_slice();
    let mass = store.mass_slice();
    let radius = store.radius_slice();

    for i in 0..store.count {
        assert!(x[i] >= 0.0 && x[i] < 100.0, "x[{}]={}", i, x[i]);
        assert!(y[i] >= 0.0 && y[i] < 100.0, "y[{}]={}", i, y[i]);
        let mag = (vx[i] * vx[i] + vy[i] * vy[i]).sqrt();
        assert!((mag - speed).abs() < 1e-4, "mag[{}]={}", i, mag);
        assert!((mass[i] - 1.0).abs() < MASS_EPS, "mass[{}]={}", i, mass[i]);
        assert!((radius[i] - 0.5).abs() < MASS_EPS, "radius[{}]={}", i, radius[i]);
    }
    // Fuerzas cero tras el seeder y tail consistente.
    let fx = store.fx_slice();
    let fy = store.fy_slice();
    for i in 0..store.count {
        assert!((fx[i] - 0.0).abs() < MASS_EPS, "fx[{}]={}", i, fx[i]);
        assert!((fy[i] - 0.0).abs() < MASS_EPS, "fy[{}]={}", i, fy[i]);
    }
    assert_eq!(store.live_count, store.count);
    assert_eq!(store.free_head, -1);
}

#[test]
fn seed_random_cloud_xy_in_range_and_centered() {
    let mut store = allocate(cfg_big()).unwrap();
    seed_random_cloud(&mut store, 100.0, 100.0, 40.0, 5.0, 7);

    let x = store.x_slice();
    let y = store.y_slice();
    for i in 0..store.count {
        assert!(x[i] >= 0.0 && x[i] <= 99.0, "x[{}]={}", i, x[i]);
        assert!(y[i] >= 0.0 && y[i] <= 99.0, "y[{}]={}", i, y[i]);
    }
    let mean_x: f32 = x.iter().copied().sum::<f32>() / COUNT as f32;
    let mean_y: f32 = y.iter().copied().sum::<f32>() / COUNT as f32;
    assert!((mean_x - 50.0).abs() < 3.0, "mean_x={}", mean_x);
    assert!((mean_y - 50.0).abs() < 3.0, "mean_y={}", mean_y);
}

#[test]
fn seed_uniform_field_xy_in_range_and_speed_bounded() {
    let mut store = allocate(cfg_big()).unwrap();
    let max_speed = 20.0f32;
    seed_uniform_field(&mut store, 100.0, 100.0, max_speed, 9);

    let x = store.x_slice();
    let y = store.y_slice();
    let vx = store.vx_slice();
    let vy = store.vy_slice();
    for i in 0..store.count {
        assert!(x[i] >= 0.0 && x[i] < 100.0, "x[{}]={}", i, x[i]);
        assert!(y[i] >= 0.0 && y[i] < 100.0, "y[{}]={}", i, y[i]);
        let mag = (vx[i] * vx[i] + vy[i] * vy[i]).sqrt();
        assert!(mag <= max_speed + EPS, "mag[{}]={}", i, mag);
    }
}

#[test]
fn seed_orbital_disk_radii_and_kepler_speed() {
    let mut store = allocate(cfg_big()).unwrap();
    let inner = 10.0f32;
    let outer = 50.0f32;
    let central_mass = 1.0e6f32;
    let constants = PhysicsConstants::default();
    let softening = constants.softening;
    let g = constants.g;
    seed_orbital_disk(
        &mut store,
        100.0,
        100.0,
        inner,
        outer,
        central_mass,
        softening,
        42,
    );

    let x = store.x_slice();
    let y = store.y_slice();
    let vx = store.vx_slice();
    let vy = store.vy_slice();
    let cx = 50.0f32;
    let cy = 50.0f32;

    for i in 0..store.count {
        let dx = x[i] - cx;
        let dy = y[i] - cy;
        let r = (dx * dx + dy * dy).sqrt();
        assert!(
            r >= inner - EPS && r <= outer + EPS,
            "r[{}]={} inner={} outer={}",
            i,
            r,
            inner,
            outer
        );
        let r_sq_soft = r * r + softening * softening;
        let v_kepler = r * (g * central_mass / (r_sq_soft * r_sq_soft.sqrt())).sqrt();
        let mag = (vx[i] * vx[i] + vy[i] * vy[i]).sqrt();
        assert!(
            (mag - v_kepler).abs() < 1e-2 * v_kepler,
            "mag[{}]={} v_kepler={}",
            i,
            mag,
            v_kepler
        );
    }
}

#[test]
fn seeders_deterministic_per_seed() {
    let mut a = allocate(cfg()).unwrap();
    let mut b = allocate(cfg()).unwrap();
    let mut c = allocate(cfg()).unwrap();
    seed_linear(&mut a, 100.0, 100.0, 10.0, 1337);
    seed_linear(&mut b, 100.0, 100.0, 10.0, 1337);
    seed_linear(&mut c, 100.0, 100.0, 10.0, 4242);

    let xa = a.x_slice();
    let xb = b.x_slice();
    let xc = c.x_slice();
    let ya = a.y_slice();
    let yb = b.y_slice();
    let yc = c.y_slice();
    let va = a.vx_slice();
    let vb = b.vx_slice();
    for i in 0..a.count {
        assert_eq!(xa[i], xb[i], "x[{}] determinismo fallo", i);
        assert_eq!(ya[i], yb[i], "y[{}] determinismo fallo", i);
        assert_eq!(va[i], vb[i], "vx[{}] determinismo fallo", i);
    }
    // Diferente seed debe producir una siembra diferente (casi seguro).
    let mut differ = false;
    for i in 0..a.count {
        if xa[i] != xc[i] || ya[i] != yc[i] {
            differ = true;
            break;
        }
    }
    assert!(differ, "semillas distintas produjeron la misma siembra");
}
