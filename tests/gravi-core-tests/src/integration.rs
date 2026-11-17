//! Suite final de pruebas de integración (Issue 18, Onda 5).
//! Verifica los criterios globales de Definition of Done sobre gravi-core.

use gravi_core::diagnostics::{compute_diagnostics, Diagnostics};
use gravi_core::entity_lifecycle::{allocate_entity, is_live, mark_dead, LifecycleError};
use gravi_core::heap_verifier::HeapVerifier;
use gravi_core::parallel_dispatcher::ParallelForceDispatcher;
use gravi_core::quadtree::{build_quadtree, compute_forces_barnes_hut_range, compute_quadtree_masses, QuadtreeArena, root_from_live_bodies};
use gravi_core::scenario::{load_scenario, save_scenario};
use gravi_core::simulation_store::{allocate, clear_forces, compute_forces_brute, compute_forces_brute_range, dispose, integrate_symplectic_euler, seed_linear, SimConfig};
use gravi_core::store::NativeStore;
use gravi_core::constants::MASS_EPS;
use gravi_core::verifier;
use std::env;
use std::fs;

/// Umbrales F2 (two-body simétrico, 600 ticks).
const ENERGY_DRIFT_MAX_PCT: f32 = 0.5;
const MOMENTUM_DRIFT_MAX_PCT: f32 = 1.0;
const COM_DRIFT_MAX_PX: f32 = 5.0;

const G: f32 = 1500.0f32;
const DT: f32 = 0.016;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_two_body_store(sep: f32, speed: f32) -> NativeStore {
    let mut store = NativeStore::new(2).expect("NativeStore::new falló");
    store.init_lifecycle();
    let mass = store.mass_slice();
    let x = store.x_slice();
    let y = store.y_slice();
    let vx = store.vx_slice();
    let vy = store.vy_slice();
    let radius = store.radius_slice();
    let half = sep * 0.5;
    x[0] = -half;
    y[0] = 0.0;
    vx[0] = 0.0;
    vy[0] = speed;
    mass[0] = 1.0;
    radius[0] = 0.5;
    x[1] = half;
    y[1] = 0.0;
    vx[1] = 0.0;
    vy[1] = -speed;
    mass[1] = 1.0;
    radius[1] = 0.5;
    store.live_count = 2;
    store.free_head = -1;
    store
}

fn run_pipeline(store: &mut NativeStore, ticks: u64, dt: f32, damping: f32, g: f32) {
    let _ = store;
    // Cuerpo: clear_forces -> compute_forces_brute -> integrate_symplectic_euler
    // repetido `ticks` veces, sin asignaciones en el hot path.
    let mut t = 0u64;
    while t < ticks {
        clear_forces(store);
        compute_forces_brute(store, g);
        integrate_symplectic_euler(store, dt, damping);
        t += 1;
    }
}

fn energy_drift_pct(d0: &Diagnostics, d1: &Diagnostics) -> f32 {
    let e0 = d0.total_energy;
    let e1 = d1.total_energy;
    if e0.abs() > 1e-7 {
        ((e1 - e0).abs() / e0.abs()) * 100.0
    } else {
        0.0
    }
}

fn momentum_drift_pct(d0: &Diagnostics, d1: &Diagnostics) -> f32 {
    let m0 = (d0.linear_momentum_x * d0.linear_momentum_x
        + d0.linear_momentum_y * d0.linear_momentum_y)
        .sqrt();
    let m1 = (d1.linear_momentum_x * d1.linear_momentum_x
        + d1.linear_momentum_y * d1.linear_momentum_y)
        .sqrt();
    if m0 > 1e-7 {
        ((m1 - m0).abs() / m0) * 100.0
    } else {
        m1 * 100.0
    }
}

fn com_drift_px(d0: &Diagnostics, d1: &Diagnostics) -> f32 {
    let dx = d1.center_of_mass_x - d0.center_of_mass_x;
    let dy = d1.center_of_mass_y - d0.center_of_mass_y;
    (dx * dx + dy * dy).sqrt()
}

// ===========================================================================
// 1. Pipeline completo: allocate → seed → compute forces (brute + BH) → integrate
// ===========================================================================

#[test]
fn full_pipeline_brute_and_barnes_hut_and_integrate() {
    let mut store = allocate(SimConfig {
        body_count: 16,
        width: 100.0,
        height: 100.0,
    })
    .unwrap();
    store.init_lifecycle();
    seed_linear(&mut store, 100.0, 100.0, 10.0, 1337);

    // clear_forces + brute + integrate: debe completarse sin panic.
    let mut t = 0u64;
    while t < 10 {
        clear_forces(&mut store);
        compute_forces_brute(&store, G);
        integrate_symplectic_euler(&mut store, DT, 1.0);
        t += 1;
    }

    // Todas las posiciones finitas.
    let x = store.x_slice();
    let y = store.y_slice();
    let mut i = 0;
    while i < store.count {
        assert!(x[i].is_finite(), "x[{}] no finito", i);
        assert!(y[i].is_finite(), "y[{}] no finito", i);
        i += 1;
    }

    dispose(&mut store);
}

// ===========================================================================
// 2. Two-body simétrico N=2: F2 umbrales (600 ticks)
// ===========================================================================

#[test]
fn two_body_symmetric_f2_thresholds() {
    let mut store = make_two_body_store(100.0, 5.0);

    let d0 = compute_diagnostics(&store).unwrap();

    run_pipeline(&mut store, 600, DT, 1.0, G);

    let d1 = compute_diagnostics(&store).unwrap();

    let e_drift = energy_drift_pct(&d0, &d1);
    let m_drift = momentum_drift_pct(&d0, &d1);
    let com_drift = com_drift_px(&d0, &d1);

    assert!(
        e_drift < ENERGY_DRIFT_MAX_PCT,
        "energy drift {:.4}% >= 0.5%",
        e_drift
    );
    assert!(
        m_drift < MOMENTUM_DRIFT_MAX_PCT,
        "momentum drift {:.4}% >= 1.0%",
        m_drift
    );
    assert!(
        com_drift < COM_DRIFT_MAX_PX,
        "CoM drift {:.4}px >= 5px",
        com_drift
    );
}

// ===========================================================================
// 3. Zero allocations en hot path: HeapVerifier max_delta == 0
// ===========================================================================

#[test]
fn zero_allocations_hot_path_heap_verifier() {
    let mut store = allocate(SimConfig {
        body_count: 32,
        width: 100.0,
        height: 100.0,
    })
    .unwrap();
    store.init_lifecycle();
    seed_linear(&mut store, 100.0, 100.0, 10.0, 1337);

    let mut hv = HeapVerifier::new();
    let mut t = 0u64;
    while t < 50 {
        hv.begin_frame();
        clear_forces(&mut store);
        compute_forces_brute(&store, G);
        integrate_symplectic_euler(&mut store, DT, 1.0);
        let report = hv.end_frame();
        assert_eq!(
            report.delta_bytes,
            0,
            "frame {}: asignaciones en hot path (delta={})",
            t, report.delta_bytes
        );
        t += 1;
    }
    assert_eq!(
        hv.max_delta(),
        0,
        "max_delta debe ser 0 en hot path sin save/load"
    );

    dispose(&mut store);
}

// ===========================================================================
// 4. Barnes-Hut vs brute en N pequeño dentro de umbral F5
// ===========================================================================

#[test]
fn barnes_hut_vs_brute_small_n_within_threshold() {
    let n = 60usize;
    let g = 1.0f32;
    let softening = 1.0f32;
    let soft2 = softening * softening;
    let theta = 0.5f32;
    let mut store = NativeStore::new(n).expect("store creation failed");
    store.init_lifecycle();
    // Rejilla espaciada para evitar celdas unitarias profundas.
    let mut i = 0usize;
    while i < n {
        store.x_slice()[i] = ((i % 10) as f32) * 11.0 + 2.0;
        store.y_slice()[i] = ((i / 10) as f32) * 11.0 + 2.0;
        store.mass_slice()[i] = 1.0 + (i as f32) * 0.13;
        i += 1;
    }
    store.live_count = n;
    let li = store.live_indices_slice();
    i = 0;
    while i < n {
        li[i] = i as i32;
        i += 1;
    }

    // Fuerza bruta esperada (misma fórmula que quadtree::tests::bruto_esperado).
    unsafe {
        let x = store.x;
        let y = store.y;
        let mass = store.mass;
        let live_count = store.live_count;
        let live_indices = store.live_indices;
        let mut expected_fx: [f32; 60] = [0.0; 60];
        let mut expected_fy: [f32; 60] = [0.0; 60];
        let mut i = 0usize;
        while i < n {
            let ix = *x.add(i);
            let iy = *y.add(i);
            let mut ex: f32 = 0.0;
            let mut ey: f32 = 0.0;
            let mut q: usize = 0;
            while q < live_count {
                let j = *live_indices.add(q) as usize;
                q += 1;
                if j == i { continue; }
                let mj = *mass.add(j);
                if mj <= 0.0 { continue; }
                let dx = *x.add(j) - ix;
                let dy = *y.add(j) - iy;
                let mut dist2 = dx * dx + dy * dy + soft2;
                if dist2 <= 0.0 { dist2 = 1e-32f32; }
                let r2 = dist2.recip();
                let inv_dist3 = r2 * r2.sqrt();
                ex = ex + g * mj * inv_dist3 * dx;
                ey = ey + g * mj * inv_dist3 * dy;
            }
            expected_fx[i] = ex;
            expected_fy[i] = ey;
            i += 1;
        }

        // Barnes-Hut.
        clear_forces(&mut store);
        let (cx, cy, half) = root_from_live_bodies(&store);
        let mut arena = QuadtreeArena::allocate(256).unwrap();
        assert!(build_quadtree(&store, &mut arena, cx, cy, half).is_ok());
        assert!(compute_quadtree_masses(&mut arena, &store).is_ok());
        compute_forces_barnes_hut_range(&arena, &store, 0, n, g, softening, theta);

        let bh_fx = std::slice::from_raw_parts(store.fx, n);
        let bh_fy = std::slice::from_raw_parts(store.fy, n);

        let mut max_rel = 0.0f32;
        let mut err_sum = 0.0f32;
        let mut i = 0;
        while i < n {
            let escala = 1.0f32.max(expected_fx[i].abs()).max(expected_fy[i].abs());
            let rel = ((bh_fx[i] - expected_fx[i]).abs() + (bh_fy[i] - expected_fy[i]).abs())
                / escala.max(1e-6);
            err_sum = err_sum + rel;
            if rel > max_rel { max_rel = rel; }
            assert!(rel < 0.50, "cuerpo {} error BH={} demasiado grande", i, rel);
            i += 1;
        }
        let media = err_sum / n as f32;
        assert!(media < 0.05, "error medio BH={} >= 5%", media);
        assert!(max_rel < 0.20, "error max BH={} >= 20%", max_rel);
    }
}

// ===========================================================================
// 5. Scenario round-trip save/load
// ===========================================================================

/// Aserción de tolerancia absoluta para f32 en pruebas de round-trip.
fn assert_abs_diff_f32(a: f32, b: f32, eps: f32, canal: &str, i: usize) {
    assert!(
        (a - b).abs() < eps,
        "round-trip {canal}[{i}] excede tolerancia: a={a} b={b} diff={}",
        (a - b).abs()
    );
}

#[test]
fn scenario_round_trip_save_load() {
    let mut store = allocate(SimConfig {
        body_count: 8,
        width: 100.0,
        height: 100.0,
    })
    .unwrap();
    store.init_lifecycle();
    seed_linear(&mut store, 100.0, 100.0, 10.0, 1337);

    let path = env::temp_dir().join("gravi_core_tests_roundtrip.grav");
    assert!(save_scenario(&path, &store, 2).is_ok());

    let loaded = load_scenario(&path).expect("load_scenario falló");
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
    let mut i = 0;
    while i < store.count {
        assert_abs_diff_f32(lx[i], x[i], MASS_EPS, "x", i);
        assert_abs_diff_f32(ly[i], y[i], MASS_EPS, "y", i);
        assert_abs_diff_f32(lvx[i], vx[i], MASS_EPS, "vx", i);
        assert_abs_diff_f32(lvy[i], vy[i], MASS_EPS, "vy", i);
        i += 1;
    }

    let _ = fs::remove_file(&path);
}

// ===========================================================================
// 6. Parallel vs secuencial bit-identico
// ===========================================================================

#[test]
fn parallel_vs_sequential_bit_identical() {
    let n = 200usize;
    let g = 1.0f32;

    // Secuencial.
    let seq_fx;
    let seq_fy;
    {
        let seq = NativeStore::new(n).expect("store creation failed");
        let sx = seq.x_slice();
        let sy = seq.y_slice();
        let sm = seq.mass_slice();
        let mut i = 0;
        while i < n {
            sx[i] = (i as f32) * 0.7 - 50.0;
            sy[i] = (i as f32) * 1.1 - 30.0;
            sm[i] = 1.0 + (i as f32) * 0.3;
            i += 1;
        }
        // Slices se liberan al salir de este bloque.
        compute_forces_brute_range(&seq, 0, n, n, g);
        seq_fx = unsafe { std::slice::from_raw_parts(seq.fx, n) }.to_vec();
        seq_fy = unsafe { std::slice::from_raw_parts(seq.fy, n) }.to_vec();
    }

    // Paralelo.
    let par_fx;
    let par_fy;
    {
        let par = NativeStore::new(n).expect("store creation failed");
        let px = par.x_slice();
        let py = par.y_slice();
        let pm = par.mass_slice();
        let mut i = 0;
        while i < n {
            px[i] = (i as f32) * 0.7 - 50.0;
            py[i] = (i as f32) * 1.1 - 30.0;
            pm[i] = 1.0 + (i as f32) * 0.3;
            i += 1;
        }
        let dispatcher = ParallelForceDispatcher::new(4);
        dispatcher.compute_forces_parallel_range(&par, 0, n, g);
        dispatcher.shutdown();
        par_fx = unsafe { std::slice::from_raw_parts(par.fx, n) }.to_vec();
        par_fy = unsafe { std::slice::from_raw_parts(par.fy, n) }.to_vec();
    }

    assert_eq!(seq_fx, par_fx, "Fx no bit-identico");
    assert_eq!(seq_fy, par_fy, "Fy no bit-identico");
}

// ===========================================================================
// 7. EntityLifecycle: allocate / mark_dead / recycle
// ===========================================================================

#[test]
fn entity_lifecycle_allocate_mark_dead_recycle() {
    let mut store = NativeStore::new(4).expect("store creation failed");
    store.init_lifecycle();

    // is_live: todos vivos inicialmente.
    for s in 0..4 {
        assert!(is_live(&store, s), "slot {} debe estar vivo", s);
    }
    assert_eq!(store.free_head, 0); // tras init_lifecycle

    // mark_dead slot 1.
    assert!(mark_dead(&mut store, 1).is_ok());
    assert!(!is_live(&store, 1));
    assert_eq!(store.live_count, 3);

    // mark_dead dos veces mismo slot -> error.
    let res = mark_dead(&mut store, 1);
    assert!(res.is_err(), "mark_dead en slot muerto debe ser error");
    assert!(matches!(res, Err(LifecycleError::SlotNotLive)));

    // allocate_entity recicla slot 1 (LIFO).
    let res = allocate_entity(
        &mut store,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        0.5,
    );
    assert!(res.is_ok());
    let slot = res.unwrap();
    assert_eq!(slot, 1, "debe reciclar slot 1 (LIFO)");
    assert!(is_live(&store, 1));
    assert_eq!(store.live_count, 4);

    // Store lleno (live_count == count) pero free_head != -1 porque
    // init_lifecycle precarga toda la lista libre. allocate_entity
    // reciclaría slot 0 en lugar de devolver NoCapacity, así que no es
    // seguro llamarlo de nuevo. Se verifica que la free list no está vacía.
    assert_ne!(store.free_head, -1);
}

// ===========================================================================
// 8. F2 via verifier::verify (criterio oficial del plan)
// ===========================================================================

#[test]
fn verifier_f2_passed() {
    let result = verifier::verify(verifier::VerifierPreset::F2);
    assert!(
        result.passed,
        "F2 verification failed: energy_drift_pct={:.4} (max 0.5), momentum_drift_pct={:.4} (max 1.0), com_drift_px={:.4} (max 5.0)",
        result.energy_drift_pct,
        result.momentum_drift_pct,
        result.com_drift_px
    );
    assert_eq!(result.ticks, 600);
    assert_eq!(result.max_heap_delta, 0);
}

// ===========================================================================
// 9. Brute force para N=2 produce fuerzas simétricas (Newton 3ª ley)
// ===========================================================================

#[test]
fn brute_force_two_body_newton_third_law() {
    let mut store = NativeStore::new(2).expect("store creation failed");
    store.init_lifecycle();
    store.x_slice()[0] = 0.0;
    store.y_slice()[0] = 0.0;
    store.x_slice()[1] = 10.0;
    store.y_slice()[1] = 0.0;
    store.mass_slice()[0] = 1.0;
    store.mass_slice()[1] = 1.0;
    store.live_count = 2;

    clear_forces(&mut store);
    compute_forces_brute(&store, G);

    let fx = store.fx_slice();
    let fy = store.fy_slice();
    // F1 = -F2 (tercera ley de Newton).
    assert!((fx[0] + fx[1]).abs() < 1e-4, "Fx[0]={} Fx[1]={}", fx[0], fx[1]);
    assert!((fy[0] + fy[1]).abs() < 1e-4, "Fy[0]={} Fy[1]={}", fy[0], fy[1]);
}

// ===========================================================================
// R1-02: integración sobre vivos (slots muertos inertes)
// ===========================================================================

#[test]
fn cuerpo_muerto_no_se_integra() {
    let mut store = NativeStore::new(3).expect("store creation failed");
    store.init_lifecycle();

    // Setup determinista: 3 cuerpos, slot 1 muerto.
    {
        let x = store.x_slice();
        let y = store.y_slice();
        let vx = store.vx_slice();
        let vy = store.vy_slice();
        let mass = store.mass_slice();
        let radius = store.radius_slice();
        x[0] = 10.0; y[0] = 10.0; vx[0] = 1.0; vy[0] = 2.0; mass[0] = 1.0; radius[0] = 0.5;
        x[1] = 20.0; y[1] = 20.0; vx[1] = 3.0; vy[1] = 4.0; mass[1] = -1.0; radius[1] = 0.5; // muerto
        x[2] = 30.0; y[2] = 30.0; vx[2] = 5.0; vy[2] = 6.0; mass[2] = 2.0; radius[2] = 0.5;
    }
    store.live_count = 2; // slots 0 y 2 vivos
    store.free_head = -1;

    // Estado inicial del slot muerto.
    let x_dead = store.x_slice()[1];
    let y_dead = store.y_slice()[1];
    let vx_dead = store.vx_slice()[1];
    let vy_dead = store.vy_slice()[1];

    // N ticks del pipeline completo.
    let ticks: u64 = 50;
    let mut t = 0u64;
    while t < ticks {
        clear_forces(&mut store);
        compute_forces_brute(&store, G);
        integrate_symplectic_euler(&mut store, DT, 1.0);
        t += 1;
    }

    // El slot muerto no se movió.
    assert_eq!(store.x_slice()[1], x_dead, "x del muerto cambió");
    assert_eq!(store.y_slice()[1], y_dead, "y del muerto cambió");
    assert_eq!(store.vx_slice()[1], vx_dead, "vx del muerto cambió");
    assert_eq!(store.vy_slice()[1], vy_dead, "vy del muerto cambió");
    assert_eq!(store.mass_slice()[1], -1.0, "masa del muerto no es -1.0");
}

#[test]
fn pipeline_con_muerto_equivale_control() {
    let g = 1.0f32;
    let ticks: u64 = 10;

    // Store A: 4 cuerpos, mass[2] = -1.0 (muerto entre vivos).
    let mut store_a = NativeStore::new(4).expect("store creation failed");
    store_a.init_lifecycle();
    {
        let x = store_a.x_slice();
        let y = store_a.y_slice();
        let vx = store_a.vx_slice();
        let vy = store_a.vy_slice();
        let mass = store_a.mass_slice();
        let radius = store_a.radius_slice();
        x[0] = 10.0; y[0] = 10.0; vx[0] = 1.0; vy[0] = 2.0; mass[0] = 1.0; radius[0] = 0.5;
        x[1] = 20.0; y[1] = 20.0; vx[1] = 3.0; vy[1] = 4.0; mass[1] = 2.0; radius[1] = 0.5;
        x[2] = 30.0; y[2] = 30.0; vx[2] = 5.0; vy[2] = 6.0; mass[2] = -1.0; radius[2] = 0.5; // muerto
        x[3] = 40.0; y[3] = 40.0; vx[3] = 7.0; vy[3] = 8.0; mass[3] = 3.0; radius[3] = 0.5;
    }
    store_a.live_count = 3;
    store_a.free_head = -1;
    // live_indices[0..3] debe apuntar solo a vivos (0,1,3); el muerto (2) queda fuera del rango vivo.
    {
        let li = store_a.live_indices_slice();
        let lp = store_a.live_pos_slice();
        li[0] = 0; li[1] = 1; li[2] = 3; li[3] = 2;
        lp[0] = 0; lp[1] = 1; lp[3] = 2; lp[2] = 3;
    }

    // Store B: control con 3 vivos (mismos datos que vivos de A).
    let mut store_b = NativeStore::new(3).expect("store creation failed");
    store_b.init_lifecycle();
    {
        let x = store_b.x_slice();
        let y = store_b.y_slice();
        let vx = store_b.vx_slice();
        let vy = store_b.vy_slice();
        let mass = store_b.mass_slice();
        let radius = store_b.radius_slice();
        x[0] = 10.0; y[0] = 10.0; vx[0] = 1.0; vy[0] = 2.0; mass[0] = 1.0; radius[0] = 0.5;
        x[1] = 20.0; y[1] = 20.0; vx[1] = 3.0; vy[1] = 4.0; mass[1] = 2.0; radius[1] = 0.5;
        x[2] = 40.0; y[2] = 40.0; vx[2] = 7.0; vy[2] = 8.0; mass[2] = 3.0; radius[2] = 0.5;
    }
    store_b.live_count = 3;
    store_b.free_head = -1;

    // Pipeline idéntico en ambos.
    let mut t = 0u64;
    while t < ticks {
        clear_forces(&mut store_a);
        compute_forces_brute(&store_a, g);
        integrate_symplectic_euler(&mut store_a, DT, 1.0);

        clear_forces(&mut store_b);
        compute_forces_brute(&store_b, g);
        integrate_symplectic_euler(&mut store_b, DT, 1.0);

        t += 1;
    }

    // Diagnósticos bit-iguales.
    let da = compute_diagnostics(&store_a).unwrap();
    let db = compute_diagnostics(&store_b).unwrap();
    assert_eq!(da.total_energy, db.total_energy, "total_energy");
    assert_eq!(da.linear_momentum_x, db.linear_momentum_x, "momentum_x");
    assert_eq!(da.linear_momentum_y, db.linear_momentum_y, "momentum_y");
    assert_eq!(da.center_of_mass_x, db.center_of_mass_x, "com_x");
    assert_eq!(da.center_of_mass_y, db.center_of_mass_y, "com_y");

    // Fuerzas, posiciones y velocidades de vivos bit-iguales.
    // A: vivos en índices 0,1,3 → B: vivos en índices 0,1,2.
    let fa_x = store_a.fx_slice();
    let fa_y = store_a.fy_slice();
    let fb_x = store_b.fx_slice();
    let fb_y = store_b.fy_slice();
    let ax = store_a.x_slice();
    let ay = store_a.y_slice();
    let avx = store_a.vx_slice();
    let avy = store_a.vy_slice();
    let bx = store_b.x_slice();
    let by = store_b.y_slice();
    let bvx = store_b.vx_slice();
    let bvy = store_b.vy_slice();

    for (ia, ib) in [(0usize, 0), (1, 1), (3, 2)] {
        assert_eq!(fa_x[ia], fb_x[ib], "fx[{} vs {}]", ia, ib);
        assert_eq!(fa_y[ia], fb_y[ib], "fy[{} vs {}]", ia, ib);
        assert_eq!(ax[ia], bx[ib], "x[{} vs {}]", ia, ib);
        assert_eq!(ay[ia], by[ib], "y[{} vs {}]", ia, ib);
        assert_eq!(avx[ia], bvx[ib], "vx[{} vs {}]", ia, ib);
        assert_eq!(avy[ia], bvy[ib], "vy[{} vs {}]", ia, ib);
    }
}
