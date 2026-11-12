//! Tests extraídos de crates/gravi-core/src/diagnostics.rs (Issue 8 — criterios de aceptación).
//! Integration test: solo API pública de gravi_core.

use gravi_core::diagnostics::{
    calculate_flops_per_frame, center_of_mass, compute_diagnostics, kinetic_energy,
    linear_momentum, potential_energy,
};
use gravi_core::store::NativeStore;

/// Store de 2 cuerpos con masa unitaria (canales de masa a 1.0).
fn store_two_bodies() -> NativeStore {
    let store = NativeStore::new(2).unwrap();
    let mass = store.mass_slice();
    mass[0] = 1.0;
    mass[1] = 1.0;
    store
}

#[test]
fn ac_kinetic_energy_two_bodies_unit_speed() {
    let store = store_two_bodies();
    let vx = store.vx_slice();
    let vy = store.vy_slice();
    vx[0] = 1.0;
    vy[0] = 0.0;
    vx[1] = 1.0;
    vy[1] = 0.0;
    let ke = kinetic_energy(&store).unwrap();
    assert!((ke - 1.0).abs() < 1e-4, "ke={}", ke);
}

#[test]
fn ac_potential_energy_is_negative() {
    let store = store_two_bodies();
    let x = store.x_slice();
    let y = store.y_slice();
    x[0] = 0.0;
    y[0] = 0.0;
    x[1] = 1.0;
    y[1] = 0.0;
    let pe = potential_energy(&store).unwrap();
    assert!(pe < 0.0, "pe={} (esperada negativa)", pe);
}

#[test]
fn ac_linear_momentum_two_body_opposing_approx_zero() {
    let store = store_two_bodies();
    let vx = store.vx_slice();
    let vy = store.vy_slice();
    vx[0] = 1.0;
    vy[0] = 0.0;
    vx[1] = -1.0;
    vy[1] = 0.0;
    let (px, py) = linear_momentum(&store).unwrap();
    assert!(px.abs() < 1e-4, "px={}", px);
    assert_eq!(py, 0.0);
}

#[test]
fn ac_center_of_mass_two_equal_bodies() {
    let store = store_two_bodies();
    let x = store.x_slice();
    let y = store.y_slice();
    x[0] = 0.0;
    y[0] = 0.0;
    x[1] = 2.0;
    y[1] = 0.0;
    let (cmx, cmy) = center_of_mass(&store).unwrap();
    assert!((cmx - 1.0).abs() < 1e-4, "cmx={}", cmx);
    assert_eq!(cmy, 0.0);
}

#[test]
fn ac_calculate_flops_per_frame_n_squared_times_12() {
    assert_eq!(calculate_flops_per_frame(0), 0);
    assert_eq!(calculate_flops_per_frame(1), 12);
    assert_eq!(calculate_flops_per_frame(4), 192);
    assert_eq!(calculate_flops_per_frame(16), 3072);
}

#[test]
fn ac_compute_diagnostics_aggregates_all_fields() {
    let store = store_two_bodies();
    let vx = store.vx_slice();
    vx[0] = 1.0;
    vx[1] = -1.0;
    let d = compute_diagnostics(&store).unwrap();
    assert!((d.kinetic_energy - 1.0).abs() < 1e-4);
    assert!(d.potential_energy < 0.0);
    assert!(d.linear_momentum_x.abs() < 1e-4);
    assert_eq!(d.linear_momentum_y, 0.0);
    assert_eq!(d.flops_per_frame, 2 * 2 * 12);
    // total_energy = ke + pe por construcción.
    assert_eq!(d.total_energy, d.kinetic_energy + d.potential_energy);
}
