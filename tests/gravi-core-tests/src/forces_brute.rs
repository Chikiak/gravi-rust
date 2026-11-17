//! Tests extraídos de crates/gravi-core/src/simulation_store.rs (Issue 3 — criterios de aceptación).
//! Integration test: solo API pública de gravi_core.

use gravi_core::constants::MASS_EPS;
use gravi_core::simulation_store::{allocate, clear_forces, compute_forces_brute, dispose, reflect_bounds, seed_linear, SimConfig};

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
    // Estado post-dispose: todo liberado; el Drop al salir del scope no hace doble liberación.
    assert_eq!(store.live_count, 0);
    assert_eq!(store.free_head, -1);
    assert_eq!(store.count, 16);
}

// ===========================================================================
// R1-02: fuerzas escalares con filtro de muertos
// ===========================================================================

#[test]
fn fuerza_cero_con_muerto_entre_vivos() {
    let g = 1.0f32;

    // Store A: 4 cuerpos, mass[2] = -1.0 (muerto entre vivos).
    let mut store_a = allocate(SimConfig {
        body_count: 4,
        width: 100.0,
        height: 100.0,
    })
    .unwrap();
    store_a.init_lifecycle();
    {
        let x = store_a.x_slice();
        let y = store_a.y_slice();
        let vx = store_a.vx_slice();
        let vy = store_a.vy_slice();
        let mass = store_a.mass_slice();
        let radius = store_a.radius_slice();
        x[0] = 10.0;
        y[0] = 10.0;
        vx[0] = 1.0;
        vy[0] = 2.0;
        mass[0] = 1.0;
        radius[0] = 0.5;
        x[1] = 20.0;
        y[1] = 20.0;
        vx[1] = 3.0;
        vy[1] = 4.0;
        mass[1] = 2.0;
        radius[1] = 0.5;
        x[2] = 30.0;
        y[2] = 30.0;
        vx[2] = 5.0;
        vy[2] = 6.0;
        mass[2] = -1.0; // muerto
        radius[2] = 0.5;
        x[3] = 40.0;
        y[3] = 40.0;
        vx[3] = 7.0;
        vy[3] = 8.0;
        mass[3] = 3.0;
        radius[3] = 0.5;
    }
    store_a.live_count = 3;
    store_a.free_head = -1;

    // Store B: control con 3 vivos (mismos datos que vivos de A).
    let mut store_b = allocate(SimConfig {
        body_count: 3,
        width: 100.0,
        height: 100.0,
    })
    .unwrap();
    store_b.init_lifecycle();
    {
        let x = store_b.x_slice();
        let y = store_b.y_slice();
        let vx = store_b.vx_slice();
        let vy = store_b.vy_slice();
        let mass = store_b.mass_slice();
        let radius = store_b.radius_slice();
        x[0] = 10.0;
        y[0] = 10.0;
        vx[0] = 1.0;
        vy[0] = 2.0;
        mass[0] = 1.0;
        radius[0] = 0.5;
        x[1] = 20.0;
        y[1] = 20.0;
        vx[1] = 3.0;
        vy[1] = 4.0;
        mass[1] = 2.0;
        radius[1] = 0.5;
        x[2] = 40.0;
        y[2] = 40.0;
        vx[2] = 7.0;
        vy[2] = 8.0;
        mass[2] = 3.0;
        radius[2] = 0.5;
    }
    store_b.live_count = 3;
    store_b.free_head = -1;

    clear_forces(&mut store_a);
    compute_forces_brute(&store_a, g);
    clear_forces(&mut store_b);
    compute_forces_brute(&store_b, g);

    let fa_x = store_a.fx_slice();
    let fa_y = store_a.fy_slice();
    let fb_x = store_b.fx_slice();
    let fb_y = store_b.fy_slice();

    // Vivos de A (índices 0,1,3) bit-iguales a control B (índices 0,1,2).
    assert_eq!(fa_x[0], fb_x[0], "fx[0]");
    assert_eq!(fa_y[0], fb_y[0], "fy[0]");
    assert_eq!(fa_x[1], fb_x[1], "fx[1]");
    assert_eq!(fa_y[1], fb_y[1], "fy[1]");
    assert_eq!(fa_x[3], fb_x[2], "fx[3] vs control[2]");
    assert_eq!(fa_y[3], fb_y[2], "fy[3] vs control[2]");

    // Muerto en A: fuerza cero (con tolerancia epsilon), masa inalterada.
    assert!((fa_x[2] - 0.0).abs() < MASS_EPS, "fx del muerto={}", fa_x[2]);
    assert!((fa_y[2] - 0.0).abs() < MASS_EPS, "fy del muerto={}", fa_y[2]);
    assert_eq!(store_a.mass_slice()[2], -1.0, "masa del muerto inalterada");

    dispose(&mut store_a);
    dispose(&mut store_b);
}
