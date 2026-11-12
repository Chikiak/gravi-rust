//! Pruebas de integración extraídas de crates/gravi-core/src/verifier.rs
//! (físicas F2/F3/F5, Issue 11).

use gravi_core::simulation_store::{allocate, SimConfig};
use gravi_core::verifier::{
    verify, VerifierPreset, COM_DRIFT_MAX_PX, ENERGY_DRIFT_MAX_PCT,
    MOMENTUM_DRIFT_MAX_PCT, seed_two_body_symmetric_for_verification,
};

#[test]
fn ac_seed_two_body_symmetric_positions_and_velocities() {
    let config = SimConfig {
        body_count: 2,
        width: 200.0,
        height: 200.0,
    };
    let mut store = allocate(config).unwrap();
    gravi_core::store::NativeStore::init_lifecycle(&mut store);

    seed_two_body_symmetric_for_verification(&mut store, 100.0, 5.0);

    let x = store.x_slice();
    let y = store.y_slice();
    let vx = store.vx_slice();
    let vy = store.vy_slice();
    let mass = store.mass_slice();

    // Cuerpo 0: (-50, 0), v = (0, 5).
    assert!((x[0] - (-50.0)).abs() < 1e-5, "x[0]={}", x[0]);
    assert_eq!(y[0], 0.0);
    assert_eq!(vx[0], 0.0);
    assert_eq!(vy[0], 5.0);
    assert_eq!(mass[0], 1.0);

    // Cuerpo 1: (+50, 0), v = (0, -5).
    assert!((x[1] - 50.0).abs() < 1e-5, "x[1]={}", x[1]);
    assert_eq!(y[1], 0.0);
    assert_eq!(vx[1], 0.0);
    assert_eq!(vy[1], -5.0);
    assert_eq!(mass[1], 1.0);

    assert_eq!(store.live_count, 2);
    assert_eq!(store.free_head, -1);
}

#[test]
fn ac_presets_exist_f2_f3_f5() {
    let f2 = VerifierPreset::F2;
    let f3 = VerifierPreset::F3;
    let f5 = VerifierPreset::F5;
    assert_eq!(f2.body_count(), 2);
    assert_eq!(f3.body_count(), 3);
    assert_eq!(f5.body_count(), 5);
    assert_eq!(f2.ticks(), 600);
    assert_ne!(f2, f3);
    assert_ne!(f3, f5);
}

#[test]
fn ac_max_heap_delta_is_zero() {
    let result = verify(VerifierPreset::F2);
    assert_eq!(result.max_heap_delta, 0, "max_heap_delta={}", result.max_heap_delta);
}

#[test]
fn ac_two_body_symmetric_600_ticks_energy_momentum_com() {
    let result = verify(VerifierPreset::F2);
    assert!(
        result.passed,
        "verification failed: energy_drift_pct={:.4} (max 0.5), momentum_drift_pct={:.4} (max 1.0), com_drift_px={:.4} (max 5.0)",
        result.energy_drift_pct, result.momentum_drift_pct, result.com_drift_px
    );
    assert!(result.energy_drift_pct < ENERGY_DRIFT_MAX_PCT);
    assert!(result.momentum_drift_pct < MOMENTUM_DRIFT_MAX_PCT);
    assert!(result.com_drift_px < COM_DRIFT_MAX_PX);
    assert_eq!(result.ticks, 600);
}

#[test]
fn ac_all_presets_run_and_report() {
    for preset in [VerifierPreset::F2, VerifierPreset::F3, VerifierPreset::F5] {
        let result = verify(preset);
        assert_eq!(result.max_heap_delta, 0);
        assert_eq!(result.ticks, preset.ticks());
    }
}
