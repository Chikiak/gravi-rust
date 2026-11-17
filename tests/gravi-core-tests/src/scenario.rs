//! Pruebas de integración extraídas de crates/gravi-core/src/scenario.rs
//! (serialización `.grav` v1, Issue 12).

use gravi_core::constants::MASS_EPS;
use gravi_core::scenario::{save_scenario, load_scenario, HEADER_SIZE};
use gravi_core::simulation_store::{allocate, seed_linear, SimConfig};
use std::fs;

fn config() -> SimConfig {
    SimConfig { body_count: 8, width: 100.0, height: 100.0 }
}

/// Aserción de tolerancia absoluta para f32 en pruebas de round-trip.
fn assert_abs_diff_f32(a: f32, b: f32, eps: f32, canal: &str, i: usize) {
    assert!(
        (a - b).abs() < eps,
        "round-trip {canal}[{i}] excede tolerancia: a={a} b={b} diff={}",
        (a - b).abs()
    );
}

#[test]
fn ac1_round_trip_save_load_recovers_xyvxvy() {
    let mut store = allocate(config()).unwrap();
    let speed = 10.0f32;
    seed_linear(&mut store, 100.0, 100.0, speed, 1337);

    let path = std::env::temp_dir().join("gravi_scenario_roundtrip.grav");
    let _ = save_scenario(&path, &store, 2);
    let loaded = load_scenario(&path).unwrap();

    assert_eq!(loaded.count, store.count);
    assert_eq!(loaded.live_count, store.count);
    let x = store.x_slice();
    let y = store.y_slice();
    let vx = store.vx_slice();
    let vy = store.vy_slice();
    let lx = loaded.x_slice();
    let ly = loaded.y_slice();
    let lvx = loaded.vx_slice();
    let lvy = loaded.vy_slice();
    for i in 0..store.count {
        assert_abs_diff_f32(lx[i], x[i], MASS_EPS, "x", i);
        assert_abs_diff_f32(ly[i], y[i], MASS_EPS, "y", i);
        assert_abs_diff_f32(lvx[i], vx[i], MASS_EPS, "vx", i);
        assert_abs_diff_f32(lvy[i], vy[i], MASS_EPS, "vy", i);
    }
    let _ = fs::remove_file(&path);
}

#[test]
fn ac2_attractor_count_gt_4_errors() {
    let store = allocate(config()).unwrap();
    let path = std::env::temp_dir().join("gravi_scenario_attractors.grav");
    assert!(save_scenario(&path, &store, 5).is_err());
    assert!(save_scenario(&path, &store, 4).is_ok());
    let _ = fs::remove_file(&path);
}

#[test]
fn ac3_header_validated_before_allocate_bad_magic() {
    let path = std::env::temp_dir().join("gravi_scenario_bad.grav");
    let mut data = vec![0u8; HEADER_SIZE + 10];
    data[0..4].copy_from_slice(b"XXXX"); // magic inválido
    fs::write(&path, &data).unwrap();
    assert!(load_scenario(&path).is_err());
    let _ = fs::remove_file(&path);
}

#[test]
fn ac4_header_validated_before_allocate_bad_version() {
    let path = std::env::temp_dir().join("gravi_scenario_badver.grav");
    let mut data = vec![0u8; HEADER_SIZE + 10];
    data[0..4].copy_from_slice(b"GRV1");
    data[4..8].copy_from_slice(&2u32.to_le_bytes()); // versión inválida
    fs::write(&path, &data).unwrap();
    assert!(load_scenario(&path).is_err());
    let _ = fs::remove_file(&path);
}

#[test]
fn ac5_load_scenario_rechaza_x_nan() {
    let mut store = allocate(config()).unwrap();
    seed_linear(&mut store, 100.0, 100.0, 10.0, 1337);
    store.x_slice()[0] = f32::NAN;

    let path = std::env::temp_dir().join("gravi_scenario_ac5_x_nan.grav");
    let _ = save_scenario(&path, &store, 2);
    let err = match load_scenario(&path) {
        Err(e) => e,
        Ok(_) => panic!("load_scenario aceptó x[0] = NaN"),
    };
    assert!(err.contains("x[0]"), "error inesperado: {err}");
    let _ = fs::remove_file(&path);
}

#[test]
fn ac5_load_scenario_rechaza_y_inf() {
    let mut store = allocate(config()).unwrap();
    seed_linear(&mut store, 100.0, 100.0, 10.0, 1337);
    store.y_slice()[0] = f32::INFINITY;

    let path = std::env::temp_dir().join("gravi_scenario_ac5_y_inf.grav");
    let _ = save_scenario(&path, &store, 2);
    let err = match load_scenario(&path) {
        Err(e) => e,
        Ok(_) => panic!("load_scenario aceptó y[0] = Inf"),
    };
    assert!(err.contains("y[0]"), "error inesperado: {err}");
    let _ = fs::remove_file(&path);
}

#[test]
fn ac5_load_scenario_rechaza_vx_neg_inf() {
    let mut store = allocate(config()).unwrap();
    seed_linear(&mut store, 100.0, 100.0, 10.0, 1337);
    store.vx_slice()[0] = f32::NEG_INFINITY;

    let path = std::env::temp_dir().join("gravi_scenario_ac5_vx_neg_inf.grav");
    let _ = save_scenario(&path, &store, 2);
    let err = match load_scenario(&path) {
        Err(e) => e,
        Ok(_) => panic!("load_scenario aceptó vx[0] = -Inf"),
    };
    assert!(err.contains("vx[0]"), "error inesperado: {err}");
    let _ = fs::remove_file(&path);
}

#[test]
fn ac5_load_scenario_rechaza_vy_nan() {
    let mut store = allocate(config()).unwrap();
    seed_linear(&mut store, 100.0, 100.0, 10.0, 1337);
    store.vy_slice()[0] = f32::NAN;

    let path = std::env::temp_dir().join("gravi_scenario_ac5_vy_nan.grav");
    let _ = save_scenario(&path, &store, 2);
    let err = match load_scenario(&path) {
        Err(e) => e,
        Ok(_) => panic!("load_scenario aceptó vy[0] = NaN"),
    };
    assert!(err.contains("vy[0]"), "error inesperado: {err}");
    let _ = fs::remove_file(&path);
}
