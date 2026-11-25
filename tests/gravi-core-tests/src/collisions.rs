//! Pruebas de integración extraídas de crates/gravi-core/src/collisions.rs
//! (colisiones elásticas e inelásticas, Issue 10 partes A y B).

use gravi_core::collisions::{
    resolve_collisions, resolve_elastic_collision, resolve_inelastic_collision, CollisionConfig,
};
use gravi_core::heap_verifier::{HeapVerifier, probe_allocate_usize};
use gravi_core::spatial_grid::{rebuild_grid, SpatialGrid};
use gravi_core::store::NativeStore;

fn make_store(count: usize) -> NativeStore {
    let mut store = NativeStore::new(count).expect("store creation failed");
    store.init_lifecycle();
    store
}

/// Dos cuerpos iguales de masa 1, radio 1, en (0,0) y (1.5, 0),
/// velocidades opuestas (i: +1, j: -1) -> colisión frontal elástica.
/// Tras el rebote con e=1: i queda con vx=-1, j con vx=+1 (intercambian).
#[test]
fn ac1_elastic_collision_swaps_velocities_equal_mass() {
    let mut store = make_store(2);
    unsafe {
        *store.x.add(0) = 0.0;
        *store.y.add(0) = 0.0;
        *store.vx.add(0) = 1.0;
        *store.vy.add(0) = 0.0;
        *store.mass.add(0) = 1.0;
        *store.radius.add(0) = 1.0;

        *store.x.add(1) = 1.5;
        *store.y.add(1) = 0.0;
        *store.vx.add(1) = -1.0;
        *store.vy.add(1) = 0.0;
        *store.mass.add(1) = 1.0;
        *store.radius.add(1) = 1.0;
    }

    unsafe { resolve_elastic_collision(&mut store, 0, 1, 1.0); }

    unsafe {
        assert!((*store.vx.add(0) - (-1.0)).abs() < 1e-5, "vx[0]={}", *store.vx.add(0));
        assert!((*store.vx.add(1) - 1.0).abs() < 1e-5, "vx[1]={}", *store.vx.add(1));
        assert!((*store.vy.add(0) - 0.0).abs() < 1e-5);
        assert!((*store.vy.add(1) - 0.0).abs() < 1e-5);
    }
}

/// Separación posicional: tras la colisión los cuerpos no se superponen.
#[test]
fn ac2_elastic_collision_positional_separation() {
    let mut store = make_store(2);
    unsafe {
        *store.x.add(0) = 0.0;
        *store.y.add(0) = 0.0;
        *store.vx.add(0) = 0.0;
        *store.vy.add(0) = 0.0;
        *store.mass.add(0) = 1.0;
        *store.radius.add(0) = 1.0;

        *store.x.add(1) = 1.0;
        *store.y.add(1) = 0.0;
        *store.vx.add(1) = 0.0;
        *store.vy.add(1) = 0.0;
        *store.mass.add(1) = 1.0;
        *store.radius.add(1) = 1.0;
    }

    unsafe { resolve_elastic_collision(&mut store, 0, 1, 1.0); }

    unsafe {
        let dx = *store.x.add(1) - *store.x.add(0);
        let dy = *store.y.add(1) - *store.y.add(0);
        let dist = (dx * dx + dy * dy).sqrt();
        assert!(dist >= 2.0 - 1e-5, "dist={} (debe ser >= r_sum=2.0)", dist);
    }
}

/// Cuerpos alejándose (v_rel_n >= 0) no reciben impulso.
#[test]
fn ac3_elastic_collision_no_impulse_when_separating() {
    let mut store = make_store(2);
    unsafe {
        *store.x.add(0) = 0.0;
        *store.y.add(0) = 0.0;
        *store.vx.add(0) = 1.0;
        *store.vy.add(0) = 0.0;
        *store.mass.add(0) = 1.0;
        *store.radius.add(0) = 1.0;

        *store.x.add(1) = 1.0;
        *store.y.add(1) = 0.0;
        *store.vx.add(1) = 2.0;
        *store.vy.add(1) = 0.0;
        *store.mass.add(1) = 1.0;
        *store.radius.add(1) = 1.0;
    }

    let vx0_before = unsafe { *store.vx.add(0) };
    let vx1_before = unsafe { *store.vx.add(1) };
    unsafe { resolve_elastic_collision(&mut store, 0, 1, 1.0); }
    unsafe {
        assert_eq!(*store.vx.add(0), vx0_before, "vx[0] no debe cambiar al alejarse");
        assert_eq!(*store.vx.add(1), vx1_before, "vx[1] no debe cambiar al alejarse");
    }
}

/// Sin overlap (dist >= r_sum) -> sin colisión.
#[test]
fn ac4_elastic_collision_no_overlap_no_impulse() {
    let mut store = make_store(2);
    unsafe {
        *store.x.add(0) = 0.0;
        *store.y.add(0) = 0.0;
        *store.vx.add(0) = -1.0;
        *store.vy.add(0) = 0.0;
        *store.mass.add(0) = 1.0;
        *store.radius.add(0) = 1.0;

        *store.x.add(1) = 5.0;
        *store.y.add(1) = 0.0;
        *store.vx.add(1) = -2.0;
        *store.vy.add(1) = 0.0;
        *store.mass.add(1) = 1.0;
        *store.radius.add(1) = 1.0;
    }

    let vx0_before = unsafe { *store.vx.add(0) };
    let vx1_before = unsafe { *store.vx.add(1) };
    unsafe { resolve_elastic_collision(&mut store, 0, 1, 1.0); }
    unsafe {
        assert_eq!(*store.vx.add(0), vx0_before);
        assert_eq!(*store.vx.add(1), vx1_before);
    }
}

/// Fusión inelástica: cuerpos lentos con overlap se fusionan, masa y
/// momento conservados, j marcado muerto (mass = -1.0).
#[test]
fn ac5_inelastic_fusion_conserves_mass_and_momentum() {
    let mut store = make_store(2);
    unsafe {
        *store.x.add(0) = 0.0;
        *store.y.add(0) = 0.0;
        *store.vx.add(0) = 2.0;
        *store.vy.add(0) = 0.0;
        *store.mass.add(0) = 1.0;
        *store.radius.add(0) = 1.0;

        *store.x.add(1) = 1.0;
        *store.y.add(1) = 0.0;
        *store.vx.add(1) = 0.0;
        *store.vy.add(1) = 0.0;
        *store.mass.add(1) = 1.0;
        *store.radius.add(1) = 1.0;
    }

    store.live_count = 2;

    let mut grid = SpatialGrid::allocate(4, 4, 2).unwrap();
    grid.set_cell_size(64.0);
    rebuild_grid(&mut grid, &store);

    let cfg = CollisionConfig {
        elastic_restitution: 0.8,
        inelastic_threshold_sq: 100.0, // fuerza fusión (v_rel^2 < 100)
        enable_fusion: true,
        radius_per_sqrt_mass: 0.5,
    };
    let n = unsafe { resolve_collisions(&mut store, &grid, cfg) };
    assert_eq!(n, 1, "se esperaba 1 fusión");

    unsafe {
        // Momento inicial: 1*2 + 1*0 = 2. Tras fusión: masa 2 -> vx = 1.
        assert!((*store.vx.add(0) - 1.0).abs() < 1e-5, "vx[0]={}", *store.vx.add(0));
        assert!((*store.mass.add(0) - 2.0).abs() < 1e-5, "mass[0]={}", *store.mass.add(0));
        assert!(*store.mass.add(1) < 0.0, "j debe estar muerto (mass<0)");
    }
}

/// resolve_collisions sobre grid real: detecta y resuelve el par.
#[test]
fn ac6_resolve_collisions_detects_pair_on_grid() {
    let mut store = make_store(2);
    unsafe {
        *store.x.add(0) = 10.0;
        *store.y.add(0) = 10.0;
        *store.vx.add(0) = 1.0;
        *store.vy.add(0) = 0.0;
        *store.mass.add(0) = 1.0;
        *store.radius.add(0) = 1.0;

        *store.x.add(1) = 11.0;
        *store.y.add(1) = 10.0;
        *store.vx.add(1) = -1.0;
        *store.vy.add(1) = 0.0;
        *store.mass.add(1) = 1.0;
        *store.radius.add(1) = 1.0;
    }
    store.live_count = 2;

    let mut grid = SpatialGrid::allocate(4, 4, 2).unwrap();
    grid.set_cell_size(64.0);
    rebuild_grid(&mut grid, &store);

    let cfg = CollisionConfig::default(); // e=0.8, umbral 0.25 -> elástico (v_rel²=4 > 0.25)
    let n = unsafe { resolve_collisions(&mut store, &grid, cfg) };
    assert_eq!(n, 1, "se esperaba 1 colisión");
    // Rebote frontal igual masa con e=0.8: impulso=1.8, vx[0]=1-1.8=-0.8, vx[1]=-1+1.8=0.8.
    unsafe {
        assert!((*store.vx.add(0) - (-0.8)).abs() < 1e-5, "vx[0]={}", *store.vx.add(0));
        assert!((*store.vx.add(1) - 0.8).abs() < 1e-5, "vx[1]={}", *store.vx.add(1));
    }
    // Separación posicional: dist >= r_sum = 2.0.
    unsafe {
        let dx = *store.x.add(1) - *store.x.add(0);
        let dy = *store.y.add(1) - *store.y.add(0);
        let dist = (dx * dx + dy * dy).sqrt();
        assert!(dist >= 2.0 - 1e-5, "dist={} (debe ser >= 2.0)", dist);
    }
}

/// resolve_collisions ignora slots muertos (mass < 0).
#[test]
fn ac7_resolve_collisions_skips_dead_slots() {
    let mut store = make_store(3);
    unsafe {
        // i=0 y j=1 vivos y colisionando; i=2 muerto.
        *store.x.add(0) = 10.0;
        *store.y.add(0) = 10.0;
        *store.vx.add(0) = 1.0;
        *store.vy.add(0) = 0.0;
        *store.mass.add(0) = 1.0;
        *store.radius.add(0) = 1.0;

        *store.x.add(1) = 11.0;
        *store.y.add(1) = 10.0;
        *store.vx.add(1) = -1.0;
        *store.vy.add(1) = 0.0;
        *store.mass.add(1) = 1.0;
        *store.radius.add(1) = 1.0;

        *store.x.add(2) = 10.5;
        *store.y.add(2) = 10.0;
        *store.mass.add(2) = -1.0; // muerto
    }
    store.live_count = 3;

    // body_count_max debe ser >= store.count (3) porque next indexa por
    // índice de entidad (0..store.count).
    let mut grid = SpatialGrid::allocate(4, 4, 3).unwrap();
    grid.set_cell_size(64.0);
    rebuild_grid(&mut grid, &store);

    let cfg = CollisionConfig::default();
    let n = unsafe { resolve_collisions(&mut store, &grid, cfg) };
    assert_eq!(n, 1, "solo el par vivo (0,1) colisiona");
}

/// `rebuild_grid` panica si `body_count_max < store.count`: `next` se
/// dimensiona por entidad (0..store.count) y desbordaría.
#[test]
#[should_panic(expected = "body_count_max")]
fn rebuild_grid_panics_si_body_count_max_menor_que_store_count() {
    let store = make_store(3); // count = 3
    let mut grid = SpatialGrid::allocate(4, 4, 2).unwrap(); // body_count_max = 2 < 3
    grid.set_cell_size(64.0);
    rebuild_grid(&mut grid, &store);
}

/// Zero-allocation en hot path: comprobación por inspección de código
/// (no hay Vec/String/format!/Box/closures en resolve_elastic_collision ni
/// resolve_collisions). Salvaguarda: llamada repetida sin panic.
#[test]
fn ac8_no_allocation_hot_path_repeated_calls() {
    let mut store = make_store(4);
    unsafe {
        for s in 0..4 {
            *store.x.add(s) = (s as f32) * 20.0;
            *store.y.add(s) = 10.0;
            *store.vx.add(s) = 1.0;
            *store.vy.add(s) = 0.0;
            *store.mass.add(s) = 1.0;
            *store.radius.add(s) = 1.0;
        }
    }
    store.live_count = 4;

    let mut grid = SpatialGrid::allocate(4, 4, 4).unwrap();
    grid.set_cell_size(64.0);
    rebuild_grid(&mut grid, &store);

    let cfg = CollisionConfig::default();

    // Hot path: 1000 llamadas repetidas sin asignaciones de heap.
    let mut hv = HeapVerifier::new();
    hv.begin_frame();
    for _ in 0..1000 {
        let _ = unsafe { resolve_collisions(&mut store, &grid, cfg) };
    }
    let report = hv.end_frame();
    assert_eq!(
        report.delta_bytes,
        0,
        "resolve_collisions asignó heap: delta_bytes={}",
        report.delta_bytes
    );

    // Sin panic y sin corromper invariantes básicas.
    assert!(store.live_count >= 1);

    // Control positivo: una asignación real debe ser detectada por el medidor.
    let mut hv_ctrl = HeapVerifier::new();
    hv_ctrl.begin_frame();
    let _control = probe_allocate_usize();
    let report_ctrl = hv_ctrl.end_frame();
    assert!(
        report_ctrl.delta_bytes > 0,
        "control positivo no detectó asignación: delta_bytes={}",
        report_ctrl.delta_bytes
    );
}

/// `resolve_inelastic_collision`: conserva masa y momento, centro de masa,
/// radio = sqrt(M) * radius_per_sqrt_mass, y `j` marcado muerto
/// (mass = -1.0, generación++, LIFO free list, live_count--).
#[test]
fn ac9_inelastic_collision_full_contract() {
    let mut store = make_store(3);
    unsafe {
        // i: m=1, x=0, v=+4 ; j: m=3, x=0.5, v=-2.
        *store.x.add(0) = 0.0;
        *store.y.add(0) = 0.0;
        *store.vx.add(0) = 4.0;
        *store.vy.add(0) = 0.0;
        *store.mass.add(0) = 1.0;
        *store.radius.add(0) = 1.0;
        *store.generation.add(0) = 0;

        *store.x.add(1) = 0.5;
        *store.y.add(1) = 0.0;
        *store.vx.add(1) = -2.0;
        *store.vy.add(1) = 0.0;
        *store.mass.add(1) = 3.0;
        *store.radius.add(1) = 1.0;
        *store.generation.add(1) = 0;
    }
    store.live_count = 2;

    let gen_j_before = unsafe { *store.generation.add(1) };
    let Ok(()) = (unsafe { resolve_inelastic_collision(&mut store, 0, 1, 0.5) })
        else { panic!("resolve_inelastic_collision falló inesperadamente en ac9") };

    unsafe {
        let new_m = 4.0f32;
        // Momento: (1*4 + 3*(-2)) / 4 = -0.5.
        assert!((*store.vx.add(0) - (-0.5)).abs() < 1e-5, "vx[0]={}", *store.vx.add(0));
        assert!((*store.vy.add(0) - 0.0).abs() < 1e-5);
        // Centro de masa: (1*0 + 3*0.5) / 4 = 0.375.
        assert!((*store.x.add(0) - 0.375).abs() < 1e-5, "x[0]={}", *store.x.add(0));
        assert!((*store.y.add(0) - 0.0).abs() < 1e-5);
        assert!((*store.mass.add(0) - new_m).abs() < 1e-5, "mass[0]={}", *store.mass.add(0));
        // Radio = sqrt(4) * 0.5 = 1.0.
        assert!((*store.radius.add(0) - 1.0).abs() < 1e-5, "radius[0]={}", *store.radius.add(0));
        // j muerto: centinela, generación++, cabeza LIFO, live_count--.
        assert!(*store.mass.add(1) < 0.0, "j debe estar muerto (mass<0)");
        assert_eq!(
            *store.generation.add(1),
            gen_j_before + 1,
            "generación[j] debe incrementarse"
        );
        assert_eq!(store.free_head, 1, "free_head debe apuntar a j (LIFO)");
        assert_eq!(store.live_count, 1, "live_count--");
    }
}

/// `resolve_inelastic_collision` con masa nula en el par: sin NaN/Inf
/// en ningún canal (`inv_new_m = 0` protege la división).
#[test]
fn ac10_inelastic_collision_zero_mass_no_nan() {
    let mut store = make_store(2);
    unsafe {
        *store.x.add(0) = 0.0;
        *store.y.add(0) = 0.0;
        *store.vx.add(0) = 1.0;
        *store.vy.add(0) = 0.0;
        *store.mass.add(0) = 0.0;
        *store.radius.add(0) = 0.5;

        *store.x.add(1) = 0.5;
        *store.y.add(1) = 0.0;
        *store.vx.add(1) = 0.0;
        *store.vy.add(1) = 0.0;
        *store.mass.add(1) = 0.0;
        *store.radius.add(1) = 0.5;
    }
    store.live_count = 2;

    let Ok(()) = (unsafe { resolve_inelastic_collision(&mut store, 0, 1, 0.5) })
        else { panic!("resolve_inelastic_collision falló inesperadamente en ac10") };

    unsafe {
        assert!(!store.vx.add(0).read().is_nan(), "vx[0] no debe ser NaN");
        assert!(!store.vy.add(0).read().is_nan());
        assert!(!store.mass.add(0).read().is_nan());
        assert!(!store.radius.add(0).read().is_nan());
        assert!(*store.mass.add(1) < 0.0, "j debe quedar muerto");
    }
}
