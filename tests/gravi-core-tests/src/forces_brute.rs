//! Tests extraídos de crates/gravi-core/src/simulation_store.rs (Issue 3 — criterios de aceptación).
//! Integration test: solo API pública de gravi_core.

use gravi_core::simulation_store::{allocate, dispose, reflect_bounds, seed_linear, SimConfig};

fn config() -> SimConfig {
    SimConfig {
        body_count: 16,
        width: 100.0,
        height: 100.0,
    }
}

#[test]
fn ac1_allocate_validates_body_count_and_dims() {
    assert!(allocate(SimConfig { body_count: 0, width: 10.0, height: 10.0 }).is_err());
    assert!(allocate(SimConfig { body_count: 5, width: 0.0, height: 10.0 }).is_err());
    assert!(allocate(SimConfig { body_count: 5, width: 10.0, height: 0.0 }).is_err());
    assert!(allocate(SimConfig { body_count: 5, width: -1.0, height: 10.0 }).is_err());
    assert!(allocate(SimConfig { body_count: 5, width: 10.0, height: -1.0 }).is_err());
    assert!(allocate(SimConfig { body_count: 5, width: 10.0, height: 10.0 }).is_ok());
}

#[test]
fn ac2_after_allocate_live_count_equals_count_free_head_minus_one_gen_zero() {
    let store = allocate(config()).unwrap();
    assert_eq!(store.live_count, store.count);
    assert_eq!(store.free_head, -1);
    assert_eq!(store.count, 16);
    for g in store.generation_slice().iter() {
        assert_eq!(*g, 0);
    }
    drop(store);
}

#[test]
fn ac3_dispose_is_idempotent_and_nulls() {
    let mut store = allocate(config()).unwrap();
    let count = store.count;
    dispose(&mut store);
    assert_eq!(store.live_count, 0);
    assert_eq!(store.free_head, -1);
    // Segundo dispose: no-op, sin panic.
    dispose(&mut store);
    assert_eq!(store.live_count, 0);
    assert_eq!(store.free_head, -1);
    // count aún válido para que Drop libere una sola vez al salir del scope.
    assert_eq!(store.count, count);
}

#[test]
fn ac4_seed_linear_fills_xy_in_range_and_speed_magnitude() {
    let mut store = allocate(config()).unwrap();
    let speed = 10.0f32;
    seed_linear(&mut store, 100.0, 100.0, speed, 1337);

    let x = store.x_slice();
    let y = store.y_slice();
    let vx = store.vx_slice();
    let vy = store.vy_slice();

    for i in 0..store.count {
        assert!(x[i] >= 0.0 && x[i] <= 100.0, "x[{}]={}", i, x[i]);
        assert!(y[i] >= 0.0 && y[i] <= 100.0, "y[{}]={}", i, y[i]);
        let mag = (vx[i] * vx[i] + vy[i] * vy[i]).sqrt();
        assert!((mag - speed).abs() < 1e-4, "mag[{}]={} speed={}", i, mag, speed);
    }
    assert_eq!(store.live_count, store.count);
    assert_eq!(store.free_head, -1);
}

#[test]
fn ac5_reflect_bounds_clamps_and_inverts_velocity() {
    let mut store = allocate(config()).unwrap();
    // Escribir valores fuera de los límites (sobre slices en un scope
    // limitado para liberar el borrow inmutable antes de reflect_bounds).
    {
        let x = store.x_slice();
        let y = store.y_slice();
        let vx = store.vx_slice();
        let vy = store.vy_slice();
        x[0] = -5.0;
        vx[0] = 3.0;
        x[1] = 105.0;
        vx[1] = -2.0;
        y[2] = -1.0;
        vy[2] = 4.0;
        y[3] = 101.0;
        vy[3] = -1.0;
    }

    reflect_bounds(&mut store, 100.0, 100.0);

    let x = store.x_slice();
    let y = store.y_slice();
    let vx = store.vx_slice();
    let vy = store.vy_slice();
    assert_eq!(x[0], 0.0);
    assert_eq!(vx[0], -3.0);
    assert_eq!(x[1], 100.0);
    assert_eq!(vx[1], 2.0);
    assert_eq!(y[2], 0.0);
    assert_eq!(vy[2], -4.0);
    assert_eq!(y[3], 100.0);
    assert_eq!(vy[3], 1.0);
}

#[test]
fn dispose_then_drop_no_double_free() {
    let mut store = allocate(config()).unwrap();
    dispose(&mut store);
    // Drop ejecuta al salir del scope sin doble liberación.
}
