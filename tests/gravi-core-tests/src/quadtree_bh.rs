//! quadtree_bh.rs — Tests de integración: traversal Barnes-Hut
//! (compute_forces_barnes_hut_range).
//! Extraídos de crates/gravi-core/src/quadtree.rs (mod tests).

use gravi_core::constants::MASS_EPS;
use gravi_core::quadtree::{build_quadtree, compute_forces_barnes_hut_range, compute_quadtree_masses, QuadtreeArena, root_from_live_bodies};
use gravi_core::store::NativeStore;

fn make_store(count: usize) -> NativeStore {
    let mut store = NativeStore::new(count).expect("store creation failed");
    store.init_lifecycle();
    store
}

/// Aceleración esperada (fuerza bruta pairwise) para el cuerpo `i`.
fn bruto_esperado(
    store: &NativeStore,
    i: usize,
    g: f32,
    soft2: f32,
) -> (f32, f32) {
    unsafe {
        let x = store.x;
        let y = store.y;
        let mass = store.mass;
        let live_count = store.live_count;
        let live_indices = store.live_indices;
        let ix = *x.add(i);
        let iy = *y.add(i);
        let mut ex: f32 = 0.0;
        let mut ey: f32 = 0.0;
        let mut q: usize = 0;
        while q < live_count {
            let j = *live_indices.add(q) as usize;
            q += 1;
            if j == i {
                continue;
            }
            let mj = *mass.add(j);
            if mj <= 0.0 {
                continue;
            }
            let dx = *x.add(j) - ix;
            let dy = *y.add(j) - iy;
            let mut dist2 = dx * dx + dy * dy + soft2;
            if dist2 <= 0.0 {
                dist2 = 1e-32f32;
            }
            let r2 = dist2.recip();
            let inv_dist3 = r2 * r2.sqrt();
            ex = ex + g * mj * inv_dist3 * dx;
            ey = ey + g * mj * inv_dist3 * dy;
        }
        (ex, ey)
    }
}

#[test]
fn b7_theta_cero_coincide_con_fuerza_bruta() {
    let mut store = make_store(5);
    store.x_slice()[0] = 10.0; store.y_slice()[0] = 10.0; store.mass_slice()[0] = 1.0;
    store.x_slice()[1] = 90.0; store.y_slice()[1] = 90.0; store.mass_slice()[1] = 2.0;
    store.x_slice()[2] = 10.0; store.y_slice()[2] = 90.0; store.mass_slice()[2] = 3.0;
    store.x_slice()[3] = 90.0; store.y_slice()[3] = 10.0; store.mass_slice()[3] = 4.0;
    store.x_slice()[4] = 50.0; store.y_slice()[4] = 50.0; store.mass_slice()[4] = 5.0;
    store.live_count = 5;
    let li = store.live_indices_slice();
    for i in 0..5usize { li[i] = i as i32; }

    let mut arena = QuadtreeArena::allocate(64).unwrap();
    assert!(build_quadtree(&store, &mut arena, 50.0, 50.0, 100.0).is_ok());
    assert!(compute_quadtree_masses(&mut arena, &store).is_ok());

    let g: f32 = 1.0;
    let softening: f32 = 0.0;
    let theta: f32 = 0.0;
    compute_forces_barnes_hut_range(&arena, &store, 0, 5, g, softening, theta);

    unsafe {
        let fx = store.fx; let fy = store.fy;
        let mut max_rel: f32 = 0.0;
        let mut i = 0usize;
        while i < 5usize {
            let (ex, ey) = bruto_esperado(&store, i, g, softening * softening);
            let ax = *fx.add(i); let ay = *fy.add(i);
            let escala = 1.0f32.max(ex.abs()).max(ey.abs());
            let rel = ((ax - ex).abs() + (ay - ey).abs()) / escala.max(1e-6);
            if rel > max_rel { max_rel = rel; }
            assert!(rel < 1e-4, "cuerpo {} theta=0 no coincide: ax={} ay={} ex={} ey={}", i, ax, ay, ex, ey);
            i += 1;
        }
        assert!(max_rel < 1e-4, "max_rel={}", max_rel);
    }
}

#[test]
fn b7_theta_positivo_error_acotado() {
    let mut store = make_store(60);
    let mut i = 0usize;
    while i < 60usize {
        // Cuerpos separados en una rejilla para evitar celdas unitarias profundas.
        store.x_slice()[i] = ((i % 10) as f32) * 11.0 + 2.0;
        store.y_slice()[i] = ((i / 10) as f32) * 11.0 + 2.0;
        store.mass_slice()[i] = 1.0 + (i as f32) * 0.13;
        i += 1;
    }
    store.live_count = 60;
    let li = store.live_indices_slice();
    i = 0;
    while i < 60usize { li[i] = i as i32; i += 1; }

    let mut arena = QuadtreeArena::allocate(256).unwrap();
    assert!(build_quadtree(&store, &mut arena, 50.0, 50.0, 60.0).is_ok());
    assert!(compute_quadtree_masses(&mut arena, &store).is_ok());

    let g: f32 = 1.0;
    let softening: f32 = 1.0;
    let theta: f32 = 0.5;
    compute_forces_barnes_hut_range(&arena, &store, 0, 60, g, softening, theta);

    unsafe {
        let fx = store.fx; let fy = store.fy;
        let soft2 = softening * softening;
        let mut err_sum: f32 = 0.0;
        let mut max_rel: f32 = 0.0;
        let mut i = 0usize;
        while i < 60usize {
            let (ex, ey) = bruto_esperado(&store, i, g, soft2);
            let ax = *fx.add(i); let ay = *fy.add(i);
            let escala = 1.0f32.max(ex.abs()).max(ey.abs());
            let rel = ((ax - ex).abs() + (ay - ey).abs()) / escala;
            err_sum = err_sum + rel;
            if rel > max_rel { max_rel = rel; }
            assert!(rel < 0.50, "cuerpo {} theta=0.5 error={} (demasiado grande)", i, rel);
            i += 1;
        }
        let media = err_sum / 60.0;
        assert!(media < 0.05, "error medio={} >= 5%", media);
        assert!(max_rel < 0.20, "error max={} >= 20%", max_rel);
    }
}

#[test]
fn b7_cluster_degenerado_sin_nan() {
    // Cluster muy apretado (casi un mismo punto): no debe haber NaN ni Inf.
    // Las posiciones son distintas para que el arbol se construya (sin
    // MaxQuadtreeDepth/bucket aun en esta base; eso llega en F5-02).
    let mut store = make_store(12);
    let mut i = 0usize;
    while i < 12usize {
        store.x_slice()[i] = 5.0 + (i as f32) * 0.001;
        store.y_slice()[i] = 5.0 + (i as f32) * 0.001;
        store.mass_slice()[i] = 1.0;
        i += 1;
    }
    store.live_count = 12;
    let li = store.live_indices_slice();
    i = 0;
    while i < 12usize { li[i] = i as i32; i += 1; }

    let mut arena = QuadtreeArena::allocate(256).unwrap();
    let (cx, cy, half) = root_from_live_bodies(&store);
    assert!(build_quadtree(&store, &mut arena, cx, cy, half).is_ok());
    assert!(compute_quadtree_masses(&mut arena, &store).is_ok());

    let g: f32 = 1.0;
    let softening: f32 = 8.0;
    let theta: f32 = 0.5;
    compute_forces_barnes_hut_range(&arena, &store, 0, 12, g, softening, theta);

    unsafe {
        let fx = store.fx; let fy = store.fy;
        let mut i = 0usize;
        while i < 12usize {
            let ax = *fx.add(i);
            let ay = *fy.add(i);
            assert!(ax.is_finite(), "cuerpo {} Fx no finito: {}", i, ax);
            assert!(ay.is_finite(), "cuerpo {} Fy no finito: {}", i, ay);
            i += 1;
        }
    }
}

#[test]
fn b7_arbol_vacio_y_rango_vacio_no_hacen_nada() {
    let store = make_store(1);
    let mut arena = QuadtreeArena::allocate(4).unwrap();
    arena.root_index = -1;
    // Sin root: no escribe ni panica.
    compute_forces_barnes_hut_range(&arena, &store, 0, 1, 1.0, 1.0, 0.5);
    unsafe { assert!((*store.fx.add(0) - 0.0).abs() < MASS_EPS, "fx[0]={}", *store.fx.add(0)); }

    // Rango vacío: no escribe.
    let mut arena2 = QuadtreeArena::allocate(4).unwrap();
    let (cx, cy, half) = root_from_live_bodies(&store);
    assert!(build_quadtree(&store, &mut arena2, cx, cy, half).is_ok());
    assert!(compute_quadtree_masses(&mut arena2, &store).is_ok());
    compute_forces_barnes_hut_range(&arena2, &store, 0, 0, 1.0, 1.0, 0.5);
}

#[test]
fn quadtree_leaf_multi_fuerza_igual_bruto() {
    // AC3 fuerza: 3 cuerpos en la misma celda (LeafMulti).
    // compute_forces_barnes_hut_range aplica fuerza de los 3 cuerpos
    // individualmente a un cuerpo externo, coincidiendo con brute force.
    let n = 4usize;
    let mut store = make_store(n);
    // Cuerpos 0,1,2 en la misma posición (50,50); cuerpo 3 en (200,200) externo.
    store.x_slice()[0] = 50.0; store.y_slice()[0] = 50.0; store.mass_slice()[0] = 1.0;
    store.x_slice()[1] = 50.0; store.y_slice()[1] = 50.0; store.mass_slice()[1] = 2.0;
    store.x_slice()[2] = 50.0; store.y_slice()[2] = 50.0; store.mass_slice()[2] = 3.0;
    store.x_slice()[3] = 200.0; store.y_slice()[3] = 200.0; store.mass_slice()[3] = 5.0;
    store.live_count = n;
    let li = store.live_indices_slice();
    li[0] = 0; li[1] = 1; li[2] = 2; li[3] = 3;

    let mut arena = QuadtreeArena::allocate(64).unwrap();
    // Celda pequeña (half=1.0 < MIN_CELL_SIZE=2.0) -> no subdivide -> LeafMulti.
    assert!(
        build_quadtree(&store, &mut arena, 50.0, 50.0, 1.0).is_ok(),
        "debe construir con LeafMulti"
    );
    assert!(compute_quadtree_masses(&mut arena, &store).is_ok());

    let g: f32 = 1.0;
    let softening: f32 = 0.0;
    let theta: f32 = 0.0; // theta=0 -> fuerza bruta exacta
    compute_forces_barnes_hut_range(&arena, &store, 0, 4, g, softening, theta);

    unsafe {
        let fx = store.fx;
        let fy = store.fy;
        // Fuerza esperada sobre el cuerpo 3 por los cuerpos 0,1,2 (bruto).
        let ix = *store.x.add(3);
        let iy = *store.y.add(3);
        let mut ex: f32 = 0.0;
        let mut ey: f32 = 0.0;
        let mut j = 0usize;
        while j < 3usize {
            let mj = *store.mass.add(j);
            let dx = *store.x.add(j) - ix;
            let dy = *store.y.add(j) - iy;
            let mut dist2 = dx * dx + dy * dy;
            if dist2 <= 0.0 {
                dist2 = 1e-32f32;
            }
            let r2 = dist2.recip();
            let inv_dist3 = r2 * r2.sqrt();
            ex = ex + g * mj * inv_dist3 * dx;
            ey = ey + g * mj * inv_dist3 * dy;
            j += 1;
        }
        let escala = 1.0f32.max(ex.abs()).max(ey.abs());
        let rel = ((*fx.add(3) - ex).abs() + (*fy.add(3) - ey).abs()) / escala.max(1e-6);
        assert!(
            rel < 1e-4,
            "cuerpo 3 (externo) fuerza BH={},{} bruto={},{} rel={}",
            *fx.add(3),
            *fy.add(3),
            ex,
            ey,
            rel
        );
    }
}
