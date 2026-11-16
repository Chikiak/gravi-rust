//! Pruebas de integración extraídas de crates/gravi-core/src/verifier.rs
//! (físicas F2/F3/F5, Issue 11).

use gravi_core::constants::PhysicsConstants;
use gravi_core::diagnostics::compute_diagnostics;
use gravi_core::heap_verifier::HeapVerifier;
use gravi_core::physics_step::PhysicsStep;
use gravi_core::simulation_store::{
    allocate, clear_forces, compute_forces_brute, integrate_symplectic_euler, SimConfig,
};
use gravi_core::store::NativeStore;
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

#[test]
fn ac_seeder_solo_dos_vivos() {
    let config = SimConfig {
        body_count: 5,
        width: 200.0,
        height: 200.0,
    };
    let mut store = allocate(config).unwrap();
    NativeStore::init_lifecycle(&mut store);

    seed_two_body_symmetric_for_verification(&mut store, 100.0, 5.0);

    let x = store.x_slice();
    let y = store.y_slice();
    let vx = store.vx_slice();
    let vy = store.vy_slice();
    let mass = store.mass_slice();

    assert_eq!(store.live_count, 2, "solo los cuerpos 0 y 1 deben estar vivos");
    assert_eq!(mass[0], 1.0, "mass[0] debe ser unitaria");
    assert_eq!(mass[1], 1.0, "mass[1] debe ser unitaria");
    assert_eq!(mass[2], 0.0, "relleno inerte: mass[2] == 0");
    assert_eq!(mass[3], 0.0, "relleno inerte: mass[3] == 0");
    assert_eq!(mass[4], 0.0, "relleno inerte: mass[4] == 0");
    assert!(x[2..5].iter().all(|&v| v == 0.0), "relleno en origen x");
    assert!(y[2..5].iter().all(|&v| v == 0.0), "relleno en origen y");
    assert!(vx[2..5].iter().all(|&v| v == 0.0), "relleno velocidad nula vx");
    assert!(vy[2..5].iter().all(|&v| v == 0.0), "relleno velocidad nula vy");
}

#[test]
fn ac_verify_insensible_a_relleno() {
    let r2 = verify(VerifierPreset::F2);
    let (e5, m5, c5) = run_verify_with_count(5);
    assert!(
        (r2.energy_drift_pct - e5).abs() < 1e-6,
        "energy_drift_pct debe ser idéntico con relleno inerte: count=2 -> {}, count=5 -> {}",
        r2.energy_drift_pct,
        e5
    );
    assert!(
        (r2.momentum_drift_pct - m5).abs() < 1e-6,
        "momentum_drift_pct debe ser idéntico con relleno inerte: count=2 -> {}, count=5 -> {}",
        r2.momentum_drift_pct,
        m5
    );
    assert!(
        (r2.com_drift_px - c5).abs() < 1e-6,
        "com_drift_px debe ser idéntico con relleno inerte: count=2 -> {}, count=5 -> {}",
        r2.com_drift_px,
        c5
    );
}

/// Replica el hot path de `verify` pero con `body_count = count` y seeder
/// con ese mismo count, para aislar el efecto del relleno del seeder.
fn run_verify_with_count(count: usize) -> (f32, f32, f32) {
    let ticks = 600u64;
    let dt = PhysicsStep::default().dt;
    let c = PhysicsConstants::default();

    let config = SimConfig {
        body_count: count,
        width: 200.0,
        height: 200.0,
    };
    let mut store = allocate(config).expect("verifier test: allocate falló");
    NativeStore::init_lifecycle(&mut store);

    seed_two_body_symmetric_for_verification(&mut store, 100.0, 5.0);

    let d0 = compute_diagnostics(&store).expect("verifier test: diagnóstico inicial falló");

    let mut heap = HeapVerifier::new();
    let mut _max_heap_delta: usize = 0;
    let mut _tick: u64 = 0;
    while _tick < ticks {
        heap.reset_heap_bytes();
        heap.begin_frame();
        clear_forces(&mut store);
        compute_forces_brute(&store, c.g);
        integrate_symplectic_euler(&mut store, dt, c.damping);
        let report = heap.end_frame();
        if report.delta_bytes > 0 {
            let delta = report.delta_bytes as usize;
            if delta > _max_heap_delta {
                _max_heap_delta = delta;
            }
        }
        _tick += 1;
    }

    let d1 = compute_diagnostics(&store).expect("verifier test: diagnóstico final falló");

    let e0 = d0.total_energy;
    let e1 = d1.total_energy;
    let energy_drift_pct = if e0.abs() > 1e-7 {
        ((e1 - e0).abs() / e0.abs()) * 100.0
    } else {
        0.0
    };

    let p0x = d0.linear_momentum_x;
    let p0y = d0.linear_momentum_y;
    let p1x = d1.linear_momentum_x;
    let p1y = d1.linear_momentum_y;
    let m0 = (p0x * p0x + p0y * p0y).sqrt();
    let m1 = (p1x * p1x + p1y * p1y).sqrt();
    let momentum_drift_pct = if m0 > 1e-7 {
        ((m1 - m0).abs() / m0) * 100.0
    } else {
        m1 * 100.0
    };

    let cm0x = d0.center_of_mass_x;
    let cm0y = d0.center_of_mass_y;
    let cm1x = d1.center_of_mass_x;
    let cm1y = d1.center_of_mass_y;
    let com_drift_px = ((cm1x - cm0x) * (cm1x - cm0x) + (cm1y - cm0y) * (cm1y - cm0y)).sqrt();

    (energy_drift_pct, momentum_drift_pct, com_drift_px)
}
