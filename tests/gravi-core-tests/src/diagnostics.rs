//! Tests extraídos de crates/gravi-core/src/diagnostics.rs (Issue 8 — criterios de aceptación).
//! Integration test: solo API pública de gravi_core.

use gravi_core::diagnostics::{
    calculate_flops_per_frame, center_of_mass, compute_diagnostics, kinetic_energy,
    linear_momentum, potential_energy,
};
use gravi_core::store::NativeStore;

/// Store de 2 cuerpos con masa unitaria (canales de masa a 1.0).
fn store_two_bodies() -> NativeStore {
    let mut store = NativeStore::new(2).unwrap();
    store.init_lifecycle();
    store.mass_slice_mut()[0] = 1.0;
    store.mass_slice_mut()[1] = 1.0;
    store
}

#[test]
fn ac_kinetic_energy_two_bodies_unit_speed() {
    let mut store = store_two_bodies();
    store.vx_slice_mut()[0] = 1.0;
    store.vy_slice_mut()[0] = 0.0;
    store.vx_slice_mut()[1] = 1.0;
    store.vy_slice_mut()[1] = 0.0;
    let ke = kinetic_energy(&store).unwrap();
    assert!((ke - 1.0).abs() < 1e-4, "ke={}", ke);
}

#[test]
fn ac_potential_energy_is_negative() {
    let mut store = store_two_bodies();
    store.x_slice_mut()[0] = 0.0;
    store.y_slice_mut()[0] = 0.0;
    store.x_slice_mut()[1] = 1.0;
    store.y_slice_mut()[1] = 0.0;
    let pe = potential_energy(&store).unwrap();
    assert!(pe < 0.0, "pe={} (esperada negativa)", pe);
}

#[test]
fn ac_linear_momentum_two_body_opposing_approx_zero() {
    let mut store = store_two_bodies();
    store.vx_slice_mut()[0] = 1.0;
    store.vy_slice_mut()[0] = 0.0;
    store.vx_slice_mut()[1] = -1.0;
    store.vy_slice_mut()[1] = 0.0;
    let (px, py) = linear_momentum(&store).unwrap();
    assert!(px.abs() < 1e-4, "px={}", px);
    assert_eq!(py, 0.0);
}

#[test]
fn ac_center_of_mass_two_equal_bodies() {
    let mut store = store_two_bodies();
    store.x_slice_mut()[0] = 0.0;
    store.y_slice_mut()[0] = 0.0;
    store.x_slice_mut()[1] = 2.0;
    store.y_slice_mut()[1] = 0.0;
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
    let mut store = store_two_bodies();
    store.vx_slice_mut()[0] = 1.0;
    store.vx_slice_mut()[1] = -1.0;
    let d = compute_diagnostics(&store).unwrap();
    assert!((d.kinetic_energy - 1.0).abs() < 1e-4);
    assert!(d.potential_energy < 0.0);
    assert!(d.linear_momentum_x.abs() < 1e-4);
    assert_eq!(d.linear_momentum_y, 0.0);
    assert_eq!(d.flops_per_frame, 2 * 2 * 12);
    // total_energy = ke + pe por construcción.
    assert_eq!(d.total_energy, d.kinetic_energy + d.potential_energy);
}

/// Construye un store de 4 cuerpos: 3 vivos (masa 1.0) + 1 muerto (masa -1.0).
fn store_three_live_one_dead() -> NativeStore {
    let mut store = NativeStore::new(4).unwrap();
    store.init_lifecycle();
    store.mass_slice_mut()[0] = 1.0;
    store.mass_slice_mut()[1] = 1.0;
    store.mass_slice_mut()[2] = 1.0;
    store.mass_slice_mut()[3] = -1.0; // cuerpo muerto
    store
}

/// Construye un store de 3 cuerpos vivos (masa 1.0), control sin muertos.
fn store_three_live() -> NativeStore {
    let mut store = NativeStore::new(3).unwrap();
    store.init_lifecycle();
    store.mass_slice_mut()[0] = 1.0;
    store.mass_slice_mut()[1] = 1.0;
    store.mass_slice_mut()[2] = 1.0;
    store
}

#[test]
fn ke_excluye_cuerpo_muerto() {
    let mut store_dead = store_three_live_one_dead();
    store_dead.vx_slice_mut()[0] = 1.0;
    store_dead.vy_slice_mut()[0] = 0.0;
    store_dead.vx_slice_mut()[1] = 1.0;
    store_dead.vy_slice_mut()[1] = 0.0;
    store_dead.vx_slice_mut()[2] = 1.0;
    store_dead.vy_slice_mut()[2] = 0.0;
    store_dead.vx_slice_mut()[3] = 99.0;
    store_dead.vy_slice_mut()[3] = 99.0; // muerto, no debe contar

    let mut store_control = store_three_live();
    store_control.vx_slice_mut()[0] = 1.0;
    store_control.vy_slice_mut()[0] = 0.0;
    store_control.vx_slice_mut()[1] = 1.0;
    store_control.vy_slice_mut()[1] = 0.0;
    store_control.vx_slice_mut()[2] = 1.0;
    store_control.vy_slice_mut()[2] = 0.0;

    let ke_dead = kinetic_energy(&store_dead).unwrap();
    let ke_control = kinetic_energy(&store_control).unwrap();
    assert!((ke_dead - ke_control).abs() < 1e-4, "ke_dead={} ke_control={}", ke_dead, ke_control);
    assert!((ke_dead - 1.5).abs() < 1e-4, "ke_dead={}", ke_dead);
}

#[test]
fn pe_aplica_softening_y_excluye_muerto() {
    let mut store_dead = store_three_live_one_dead();
    store_dead.x_slice_mut()[0] = 0.0;
    store_dead.y_slice_mut()[0] = 0.0;
    store_dead.x_slice_mut()[1] = 1.0;
    store_dead.y_slice_mut()[1] = 0.0;
    store_dead.x_slice_mut()[2] = 2.0;
    store_dead.y_slice_mut()[2] = 0.0;
    store_dead.x_slice_mut()[3] = 999.0;
    store_dead.y_slice_mut()[3] = 999.0; // muerto

    // Referencia f64 con r2 + 64.0, G = 1500.0, masas = 1.0
    let g: f64 = 1500.0;
    let softening_sq: f64 = 64.0;
    let mut expected: f64 = 0.0;
    let xs = [0.0f64, 1.0, 2.0];
    for i in 0..3 {
        for j in (i + 1)..3 {
            let dx = xs[j] - xs[i];
            let dy = 0.0;
            let dist2 = dx * dx + dy * dy + softening_sq;
            expected -= g * dist2.sqrt().recip();
        }
    }

    let pe_dead = potential_energy(&store_dead).unwrap();
    assert!((pe_dead as f64 - expected).abs() < 1e-3, "pe_dead={} expected={}", pe_dead, expected);

    // Insertar cuerpo muerto entre vivos no cambia el valor.
    let mut store_control = store_three_live();
    store_control.x_slice_mut()[0] = 0.0;
    store_control.y_slice_mut()[0] = 0.0;
    store_control.x_slice_mut()[1] = 1.0;
    store_control.y_slice_mut()[1] = 0.0;
    store_control.x_slice_mut()[2] = 2.0;
    store_control.y_slice_mut()[2] = 0.0;
    let pe_control = potential_energy(&store_control).unwrap();
    assert!((pe_dead - pe_control).abs() < 1e-4, "pe_dead={} pe_control={}", pe_dead, pe_control);
}

#[test]
fn momento_excluye_cuerpo_muerto() {
    let mut store_dead = store_three_live_one_dead();
    store_dead.vx_slice_mut()[0] = 1.0;
    store_dead.vy_slice_mut()[0] = 0.0;
    store_dead.vx_slice_mut()[1] = -1.0;
    store_dead.vy_slice_mut()[1] = 0.0;
    store_dead.vx_slice_mut()[2] = 0.0;
    store_dead.vy_slice_mut()[2] = 1.0;
    store_dead.vx_slice_mut()[3] = 50.0;
    store_dead.vy_slice_mut()[3] = -50.0; // muerto

    let mut store_control = store_three_live();
    store_control.vx_slice_mut()[0] = 1.0;
    store_control.vy_slice_mut()[0] = 0.0;
    store_control.vx_slice_mut()[1] = -1.0;
    store_control.vy_slice_mut()[1] = 0.0;
    store_control.vx_slice_mut()[2] = 0.0;
    store_control.vy_slice_mut()[2] = 1.0;

    let (px_d, py_d) = linear_momentum(&store_dead).unwrap();
    let (px_c, py_c) = linear_momentum(&store_control).unwrap();
    assert!((px_d - px_c).abs() < 1e-4, "px_d={} px_c={}", px_d, px_c);
    assert!((py_d - py_c).abs() < 1e-4, "py_d={} py_c={}", py_d, py_c);
    assert!((px_d - 0.0).abs() < 1e-4, "px_d={}", px_d);
    assert!((py_d - 1.0).abs() < 1e-4, "py_d={}", py_d);
}

#[test]
fn com_excluye_muerto_y_umbral_epsilon() {
    // Solo muertos: devuelve (0.0, 0.0) sin panic.
    let mut store_all_dead = NativeStore::new(3).unwrap();
    store_all_dead.init_lifecycle();
    store_all_dead.mass_slice_mut()[0] = -1.0;
    store_all_dead.mass_slice_mut()[1] = -1.0;
    store_all_dead.mass_slice_mut()[2] = -1.0;
    let com_dead = center_of_mass(&store_all_dead).unwrap();
    assert_eq!(com_dead, (0.0, 0.0));

    // 3 vivos + 1 muerto: igual al control sin muertos.
    let mut store_dead = store_three_live_one_dead();
    store_dead.x_slice_mut()[0] = 0.0;
    store_dead.y_slice_mut()[0] = 0.0;
    store_dead.x_slice_mut()[1] = 2.0;
    store_dead.y_slice_mut()[1] = 0.0;
    store_dead.x_slice_mut()[2] = 4.0;
    store_dead.y_slice_mut()[2] = 0.0;
    store_dead.x_slice_mut()[3] = 999.0;
    store_dead.y_slice_mut()[3] = 999.0; // muerto

    let mut store_control = store_three_live();
    store_control.x_slice_mut()[0] = 0.0;
    store_control.y_slice_mut()[0] = 0.0;
    store_control.x_slice_mut()[1] = 2.0;
    store_control.y_slice_mut()[1] = 0.0;
    store_control.x_slice_mut()[2] = 4.0;
    store_control.y_slice_mut()[2] = 0.0;

    let (cmx_d, cmy_d) = center_of_mass(&store_dead).unwrap();
    let (cmx_c, cmy_c) = center_of_mass(&store_control).unwrap();
    assert!((cmx_d - cmx_c).abs() < 1e-4, "cmx_d={} cmx_c={}", cmx_d, cmx_c);
    assert!((cmy_d - cmy_c).abs() < 1e-4, "cmy_d={} cmy_c={}", cmy_d, cmy_c);
    assert!((cmx_d - 2.0).abs() < 1e-4, "cmx_d={}", cmx_d);
    assert_eq!(cmy_d, 0.0);
}

#[test]
fn diagnosticos_equivales_control_con_muerto() {
    let mut store_dead = store_three_live_one_dead();
    store_dead.vx_slice_mut()[0] = 1.0;
    store_dead.vy_slice_mut()[0] = 0.0;
    store_dead.x_slice_mut()[0] = 0.0;
    store_dead.y_slice_mut()[0] = 0.0;
    store_dead.vx_slice_mut()[1] = -1.0;
    store_dead.vy_slice_mut()[1] = 0.0;
    store_dead.x_slice_mut()[1] = 2.0;
    store_dead.y_slice_mut()[1] = 0.0;
    store_dead.vx_slice_mut()[2] = 0.0;
    store_dead.vy_slice_mut()[2] = 1.0;
    store_dead.x_slice_mut()[2] = 4.0;
    store_dead.y_slice_mut()[2] = 0.0;
    store_dead.vx_slice_mut()[3] = 99.0;
    store_dead.vy_slice_mut()[3] = -99.0;
    store_dead.x_slice_mut()[3] = 999.0;
    store_dead.y_slice_mut()[3] = 999.0; // muerto

    let mut store_control = store_three_live();
    store_control.vx_slice_mut()[0] = 1.0;
    store_control.vy_slice_mut()[0] = 0.0;
    store_control.x_slice_mut()[0] = 0.0;
    store_control.y_slice_mut()[0] = 0.0;
    store_control.vx_slice_mut()[1] = -1.0;
    store_control.vy_slice_mut()[1] = 0.0;
    store_control.x_slice_mut()[1] = 2.0;
    store_control.y_slice_mut()[1] = 0.0;
    store_control.vx_slice_mut()[2] = 0.0;
    store_control.vy_slice_mut()[2] = 1.0;
    store_control.x_slice_mut()[2] = 4.0;
    store_control.y_slice_mut()[2] = 0.0;

    let d_dead = compute_diagnostics(&store_dead).unwrap();
    let d_control = compute_diagnostics(&store_control).unwrap();

    assert!((d_dead.kinetic_energy - d_control.kinetic_energy).abs() < 1e-4);
    assert!((d_dead.potential_energy - d_control.potential_energy).abs() < 1e-4);
    assert!((d_dead.linear_momentum_x - d_control.linear_momentum_x).abs() < 1e-4);
    assert!((d_dead.linear_momentum_y - d_control.linear_momentum_y).abs() < 1e-4);
    assert!((d_dead.center_of_mass_x - d_control.center_of_mass_x).abs() < 1e-4);
    assert!((d_dead.center_of_mass_y - d_control.center_of_mass_y).abs() < 1e-4);
    assert_eq!(d_dead.total_energy, d_dead.kinetic_energy + d_dead.potential_energy);
}
