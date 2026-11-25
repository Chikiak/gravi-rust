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
    store.x_slice_mut()[0] = 10.0; store.y_slice_mut()[0] = 10.0; store.mass_slice_mut()[0] = 1.0;
    store.x_slice_mut()[1] = 90.0; store.y_slice_mut()[1] = 90.0; store.mass_slice_mut()[1] = 2.0;
    store.x_slice_mut()[2] = 10.0; store.y_slice_mut()[2] = 90.0; store.mass_slice_mut()[2] = 3.0;
    store.x_slice_mut()[3] = 90.0; store.y_slice_mut()[3] = 10.0; store.mass_slice_mut()[3] = 4.0;
    store.x_slice_mut()[4] = 50.0; store.y_slice_mut()[4] = 50.0; store.mass_slice_mut()[4] = 5.0;
    store.live_count = 5;
    let li = store.live_indices_slice_mut();
    for i in 0..5usize { li[i] = i as i32; }
    drop(li);

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
        store.x_slice_mut()[i] = ((i % 10) as f32) * 11.0 + 2.0;
        store.y_slice_mut()[i] = ((i / 10) as f32) * 11.0 + 2.0;
        store.mass_slice_mut()[i] = 1.0 + (i as f32) * 0.13;
        i += 1;
    }
    store.live_count = 60;
    let li = store.live_indices_slice_mut();
    i = 0;
    while i < 60usize { li[i] = i as i32; i += 1; }
    drop(li);

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
        store.x_slice_mut()[i] = 5.0 + (i as f32) * 0.001;
        store.y_slice_mut()[i] = 5.0 + (i as f32) * 0.001;
        store.mass_slice_mut()[i] = 1.0;
        i += 1;
    }
    store.live_count = 12;
    let li = store.live_indices_slice_mut();
    i = 0;
    while i < 12usize { li[i] = i as i32; i += 1; }
    drop(li);

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
    store.x_slice_mut()[0] = 50.0; store.y_slice_mut()[0] = 50.0; store.mass_slice_mut()[0] = 1.0;
    store.x_slice_mut()[1] = 50.0; store.y_slice_mut()[1] = 50.0; store.mass_slice_mut()[1] = 2.0;
    store.x_slice_mut()[2] = 50.0; store.y_slice_mut()[2] = 50.0; store.mass_slice_mut()[2] = 3.0;
    store.x_slice_mut()[3] = 200.0; store.y_slice_mut()[3] = 200.0; store.mass_slice_mut()[3] = 5.0;
    store.live_count = n;
    let li = store.live_indices_slice_mut();
    li[0] = 0; li[1] = 1; li[2] = 2; li[3] = 3;
    drop(li);

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

#[test]
fn leafbucket_masa_count_correctos() {
    // AC3: 3 cuerpos en la misma coordenada exacta (50,50), masas 1, 2, 3.
    // Con half=1.0 < MIN_CELL_SIZE=2.0 no se subdivide -> root es LeafMulti (kind=3).
    let n = 3usize;
    let mut store = make_store(n);
    store.x_slice_mut()[0] = 50.0;
    store.y_slice_mut()[0] = 50.0;
    store.mass_slice_mut()[0] = 1.0;
    store.x_slice_mut()[1] = 50.0;
    store.y_slice_mut()[1] = 50.0;
    store.mass_slice_mut()[1] = 2.0;
    store.x_slice_mut()[2] = 50.0;
    store.y_slice_mut()[2] = 50.0;
    store.mass_slice_mut()[2] = 3.0;
    store.live_count = n;
    let li = store.live_indices_slice_mut();
    li[0] = 0;
    li[1] = 1;
    li[2] = 2;
    drop(li);

    let mut arena = QuadtreeArena::allocate(64).unwrap();
    assert!(
        build_quadtree(&store, &mut arena, 50.0, 50.0, 1.0).is_ok(),
        "build no debe fallar con mitad menor a MIN_CELL_SIZE"
    );
    assert!(compute_quadtree_masses(&mut arena, &store).is_ok());

    let root = unsafe { arena.read_node(arena.root_index as usize) };
    assert_eq!(root.kind, 3, "root debe ser LeafMulti (kind=3)");
    assert_eq!(root.count, 3, "root.count debe ser 3 cuerpos");
    assert_eq!(root.mass, 6.0, "root.mass debe ser 1+2+3=6.0");
}

#[test]
fn leafbucket_fuerza_equivale_brute_con_cuerpos_coincidentes() {
    // AC1: 2 cuerpos coincidentes en (50,50) masas 1, 2; cuerpo 3 externo en (200,200) masa 5.
    // Fuerza sobre cuerpo 3 debe igualar fuerza bruta sobre cuerpos 0, 1 (theta=0 exacto).
    let n = 4usize;
    let mut store = make_store(n);
    store.x_slice_mut()[0] = 50.0;
    store.y_slice_mut()[0] = 50.0;
    store.mass_slice_mut()[0] = 1.0;
    store.x_slice_mut()[1] = 50.0;
    store.y_slice_mut()[1] = 50.0;
    store.mass_slice_mut()[1] = 2.0;
    store.x_slice_mut()[2] = 999.0;
    store.y_slice_mut()[2] = 999.0;
    store.mass_slice_mut()[2] = 0.0; // inerte, no aporta fuerza
    store.x_slice_mut()[3] = 200.0;
    store.y_slice_mut()[3] = 200.0;
    store.mass_slice_mut()[3] = 5.0;
    store.live_count = n;
    let li = store.live_indices_slice_mut();
    li[0] = 0;
    li[1] = 1;
    li[2] = 2;
    li[3] = 3;
    drop(li);

    let mut arena = QuadtreeArena::allocate(64).unwrap();
    assert!(
        build_quadtree(&store, &mut arena, 50.0, 50.0, 1.0).is_ok(),
        "debe construir con LeafMulti (sin subdividir)"
    );
    assert!(compute_quadtree_masses(&mut arena, &store).is_ok());

    let g: f32 = 1.0;
    let softening: f32 = 0.0;
    let theta: f32 = 0.0;
    compute_forces_barnes_hut_range(&arena, &store, 0, 4, g, softening, theta);

    unsafe {
        let fx = store.fx;
        let fy = store.fy;
        // Fuerza esperada sobre el cuerpo 3 por los cuerpos 0,1 (bruto, sin incluir al propio 3).
        let ix = *store.x.add(3);
        let iy = *store.y.add(3);
        let mut ex: f32 = 0.0;
        let mut ey: f32 = 0.0;
        let mut j = 0usize;
        while j < 2usize {
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
        let ax = *fx.add(3);
        let ay = *fy.add(3);
        let escala = 1.0f32.max(ex.abs()).max(ey.abs());
        let rel = ((ax - ex).abs() + (ay - ey).abs()) / escala.max(1e-6);
        assert!(
            rel < 1e-4,
            "cuerpo 3 fuerza BH={},{} bruto(0,1)={},{} rel={}",
            ax,
            ay,
            ex,
            ey,
            rel
        );
    }
}

#[test]
fn leafbucket_arena_no_desborda_con_cascada_coincidentes() {
    // AC2: 50 cuerpos coincidentes en (50,50) con arena pequeña (64) no debe
    // hacer overflow ni panicar; todas las fuerzas deben ser finitas.
    let n = 50usize;
    let mut store = make_store(n);
    let mut i = 0usize;
    while i < n {
        store.x_slice_mut()[i] = 50.0;
        store.y_slice_mut()[i] = 50.0;
        store.mass_slice_mut()[i] = 1.0;
        i += 1;
    }
    store.live_count = n;
    let li = store.live_indices_slice_mut();
    i = 0;
    while i < n {
        li[i] = i as i32;
        i += 1;
    }
    drop(li);

    let mut arena = QuadtreeArena::allocate(64).unwrap();
    assert!(
        build_quadtree(&store, &mut arena, 50.0, 50.0, 1.0).is_ok(),
        "50 cuerpos coincidentes no deben hacer overflow del arena"
    );
    assert!(compute_quadtree_masses(&mut arena, &store).is_ok());

    compute_forces_barnes_hut_range(&arena, &store, 0, 50, 1.0, 8.0, 0.5);

    unsafe {
        let fx = store.fx;
        let fy = store.fy;
        let mut i = 0usize;
        while i < n {
            let ax = *fx.add(i);
            let ay = *fy.add(i);
            assert!(ax.is_finite(), "cuerpo {} Fx no finito: {}", i, ax);
            assert!(ay.is_finite(), "cuerpo {} Fy no finito: {}", i, ay);
            i += 1;
        }
    }
}

#[test]
fn leafmulti_live_indices_no_consecutivos_masa_count() {
    // A1: live_indices es una permutación NO identidad (slots no consecutivos),
    // root LeafMulti con count y masa correctos, cadena body_next bien enlazada.
    let mut store = make_store(4);
    store.x_slice_mut()[0] = 50.0;
    store.y_slice_mut()[0] = 50.0;
    store.mass_slice_mut()[0] = 1.0;
    store.x_slice_mut()[1] = 50.0;
    store.y_slice_mut()[1] = 50.0;
    store.mass_slice_mut()[1] = 2.0;
    store.x_slice_mut()[2] = 50.0;
    store.y_slice_mut()[2] = 50.0;
    store.mass_slice_mut()[2] = 3.0;
    store.x_slice_mut()[3] = 200.0;
    store.y_slice_mut()[3] = 200.0;
    store.mass_slice_mut()[3] = 5.0;
    store.live_count = 4;
    let li = store.live_indices_slice_mut();
    li[0] = 2;
    li[1] = 0;
    li[2] = 1;
    li[3] = 3;
    drop(li);

    let mut arena = QuadtreeArena::allocate(64).unwrap();
    assert!(
        build_quadtree(&store, &mut arena, 50.0, 50.0, 1.0).is_ok(),
        "build no debe fallar con mitad menor a MIN_CELL_SIZE"
    );
    assert!(compute_quadtree_masses(&mut arena, &store).is_ok());

    let root = unsafe { arena.read_node(arena.root_index as usize) };
    assert_eq!(root.kind, 3, "root debe ser LeafMulti (kind=3)");
    assert_eq!(root.count, 4, "root.count debe ser 4 cuerpos");
    assert!(
        (root.mass - 11.0).abs() < 1e-6,
        "root.mass debe ser 11.0 (1+2+3+5), fue {}",
        root.mass
    );
    // Cadena vía puntero crudo: 2 -> 0 -> 1 -> 3 -> -1
    unsafe {
        assert_eq!(*store.body_next.add(2), 0, "body_next[2] debe ser 0");
        assert_eq!(*store.body_next.add(0), 1, "body_next[0] debe ser 1");
        assert_eq!(*store.body_next.add(1), 3, "body_next[1] debe ser 3");
        assert_eq!(*store.body_next.add(3), -1, "body_next[3] debe ser -1 (fin)");
    }
}

#[test]
fn leafmulti_live_indices_no_consecutivos_fuerza_igual_bruto() {
    // A2: misma permutación; fuerza BH (theta=0) == fuerza bruta para cada cuerpo vivo.
    let mut store = make_store(4);
    store.x_slice_mut()[0] = 50.0;
    store.y_slice_mut()[0] = 50.0;
    store.mass_slice_mut()[0] = 1.0;
    store.x_slice_mut()[1] = 50.0;
    store.y_slice_mut()[1] = 50.0;
    store.mass_slice_mut()[1] = 2.0;
    store.x_slice_mut()[2] = 50.0;
    store.y_slice_mut()[2] = 50.0;
    store.mass_slice_mut()[2] = 3.0;
    store.x_slice_mut()[3] = 200.0;
    store.y_slice_mut()[3] = 200.0;
    store.mass_slice_mut()[3] = 5.0;
    store.live_count = 4;
    let li = store.live_indices_slice_mut();
    li[0] = 2;
    li[1] = 0;
    li[2] = 1;
    li[3] = 3;
    drop(li);

    let mut arena = QuadtreeArena::allocate(64).unwrap();
    assert!(
        build_quadtree(&store, &mut arena, 50.0, 50.0, 1.0).is_ok(),
        "build no debe fallar"
    );
    assert!(compute_quadtree_masses(&mut arena, &store).is_ok());

    // g=1500.0 (DEFAULT_G de constants.rs): el suelo escala=1.0 convierte
    // `rel` en error absoluto, así que con g pequeño el error del bug de
    // membresía (~5e-5) quedaba bajo el umbral 1e-4 y el test pasaba también
    // con el bug. Con g=1500 el cuerpo 3 recibe del clúster masa 6 (correcto)
    // vs masa 3 (bug de rango): rel ≈ 7e-2 >> 1e-4. theta=0 mantiene la
    // comparación exacta contra bruto con el mismo g y soft2.
    let g: f32 = 1500.0;
    // softening=1.0 evita la singularidad: dx=dy=0 -> dist2 = 1, fuerzas
    // finitas, así que la comparación rel<1e-4 es real para los 4 cuerpos.
    let softening: f32 = 1.0;
    let theta: f32 = 0.0;
    compute_forces_barnes_hut_range(&arena, &store, 0, 4, g, softening, theta);

    unsafe {
        let fx = store.fx;
        let fy = store.fy;
        let mut i = 0usize;
        while i < 4usize {
            let (ex, ey) = bruto_esperado(&store, i, g, softening * softening);
            let ax = *fx.add(i);
            let ay = *fy.add(i);
            // La rama is_nan se mantiene como defensa y documentación
            // del cálculo: con softening > 0 y fuerzas finitas no altera la
            // verificación numérica de arriba.
            if ex.is_nan() || ey.is_nan() {
                assert!(
                    ax.is_nan() || ay.is_nan(),
                    "cuerpo {} BH={},{} no es NaN aunque bruto sí (singularidad cuerpos coincidentes)",
                    i,
                    ax,
                    ay
                );
            } else {
                let escala = 1.0f32.max(ex.abs()).max(ey.abs());
                let rel = ((ax - ex).abs() + (ay - ey).abs()) / escala.max(1e-6);
                assert!(
                    rel < 1e-4,
                    "cuerpo {} fuerza BH={},{} bruto={},{} rel={}",
                    i,
                    ax,
                    ay,
                    ex,
                    ey,
                    rel
                );
            }
            i += 1;
        }
    }
}

#[test]
fn leafmulti_no_incluye_cuerpo_ajeno_del_rango() {
    // B: slot 1 no vive (fuera de live_indices) y no debe aportar masa ni fuerza.
    let mut store = make_store(4);
    store.x_slice_mut()[0] = 50.0;
    store.y_slice_mut()[0] = 50.0;
    store.mass_slice_mut()[0] = 1.0;
    store.x_slice_mut()[1] = 9999.0;
    store.y_slice_mut()[1] = 9999.0;
    store.mass_slice_mut()[1] = 100.0;
    store.x_slice_mut()[2] = 50.0;
    store.y_slice_mut()[2] = 50.0;
    store.mass_slice_mut()[2] = 3.0;
    store.x_slice_mut()[3] = 200.0;
    store.y_slice_mut()[3] = 200.0;
    store.mass_slice_mut()[3] = 5.0;
    store.live_count = 3;
    let li = store.live_indices_slice_mut();
    li[0] = 0;
    li[1] = 2;
    li[2] = 3;
    drop(li);

    let mut arena = QuadtreeArena::allocate(64).unwrap();
    assert!(
        build_quadtree(&store, &mut arena, 50.0, 50.0, 1.0).is_ok(),
        "build no debe fallar"
    );
    assert!(compute_quadtree_masses(&mut arena, &store).is_ok());

    let root = unsafe { arena.read_node(arena.root_index as usize) };
    assert_eq!(root.kind, 3, "root debe ser LeafMulti (kind=3)");
    assert_eq!(
        root.count,
        3,
        "root.count debe ser 3 cuerpos (sin el ajeno slot 1)"
    );
    assert!(
        (root.mass - 9.0).abs() < 1e-6,
        "root.mass debe ser 9.0 (1+3+5, sin 100 del slot 1), fue {}",
        root.mass
    );

    // g=1500.0 (DEFAULT_G de constants.rs): el suelo escala=1.0 convierte `rel`
    // en error absoluto, así que con g pequeño el cuerpo ajeno (slot 1, masa 100
    // a (9999,9999)) aportaba ≈5e-5 < 1e-4 y el bug pasaba inadvertido. Con
    // g=1500 el error por incluirlo supera 1e-4 → el test distingue el bug;
    // theta=0 mantiene la comparación exacta contra bruto con el mismo g.
    let g: f32 = 1500.0;
    let softening: f32 = 0.0;
    let theta: f32 = 0.0;
    compute_forces_barnes_hut_range(&arena, &store, 0, 3, g, softening, theta);

    unsafe {
        let fx = store.fx;
        let fy = store.fy;
        // Cuerpo 3: fuerza bruta contra {0, 2} (no contra el ajeno slot 1).
        let (ex, ey) = bruto_esperado(&store, 3, g, softening * softening);
        let ax = *fx.add(3);
        let ay = *fy.add(3);
        let escala = 1.0f32.max(ex.abs()).max(ey.abs());
        let rel = ((ax - ex).abs() + (ay - ey).abs()) / escala.max(1e-6);
        assert!(
            rel < 1e-4,
            "cuerpo 3 fuerza BH={},{} bruto={},{} rel={}",
            ax,
            ay,
            ex,
            ey,
            rel
        );
    }
}

#[test]
fn leaf_multi_monopolo_lejano_theta_positivo() {
    // AC1: núcleo lejano de 200 cuerpos coincidentes -> monopolo analítico.
    const N: usize = 200;
    let mut store = make_store(N + 1);
    let mut i = 0usize;
    while i < N {
        store.x_slice_mut()[i] = 50.0;
        store.y_slice_mut()[i] = 50.0;
        store.mass_slice_mut()[i] = 1.0;
        i += 1;
    }
    // Cuerpo 200 alejado, masa 0 (inerte: no aporta masa al nodo hoja).
    store.x_slice_mut()[N] = 1000.0;
    store.y_slice_mut()[N] = 1000.0;
    store.mass_slice_mut()[N] = 0.0;
    store.live_count = N + 1;
    let li = store.live_indices_slice_mut();
    i = 0;
    while i < N + 1 {
        li[i] = i as i32;
        i += 1;
    }
    drop(li);

    let mut arena = QuadtreeArena::allocate(256).unwrap();
    assert!(
        build_quadtree(&store, &mut arena, 50.0, 50.0, 1.0).is_ok(),
        "root_half=1.0 < MIN_CELL_SIZE=2.0 -> LeafMulti sin subdividir"
    );
    assert!(compute_quadtree_masses(&mut arena, &store).is_ok());

    let root = unsafe { arena.read_node(arena.root_index as usize) };
    assert_eq!(root.kind, 3, "root debe ser LeafMulti (kind=3)");
    assert!(
        (root.mass - 200.0f32).abs() < 1e-6,
        "root.mass debe ser 200.0, fue {}",
        root.mass
    );

    let g: f32 = 1.0;
    let softening: f32 = 0.0;
    let theta: f32 = 0.3;
    compute_forces_barnes_hut_range(&arena, &store, 0, N + 1, g, softening, theta);

    unsafe {
        let fx = store.fx;
        let fy = store.fy;
        let ix = *store.x.add(N);
        let iy = *store.y.add(N);
        // Monopolo analítico: M=200 en (50,50) sobre el cuerpo en (1000,1000).
        let dx = 50.0f32 - ix;
        let dy = 50.0f32 - iy;
        let d2 = dx * dx + dy * dy;
        let inv_dist3 = d2.recip() * d2.recip().sqrt();
        let ex = g * 200.0f32 * inv_dist3 * dx;
        let ey = g * 200.0f32 * inv_dist3 * dy;
        let ax = *fx.add(N);
        let ay = *fy.add(N);
        let escala = 1.0f32.max(ex.abs()).max(ey.abs());
        let rel = ((ax - ex).abs() + (ay - ey).abs()) / escala.max(1e-6);
        assert!(
            rel < 1e-5,
            "cuerpo 200 monopolo lejano: BH={},{} analitico={},{} rel={}",
            ax,
            ay,
            ex,
            ey,
            rel
        );
    }
}

#[test]
fn leaf_multi_cerca_cadena_exacta() {
    // AC3: cuerpo cerca/dentro de la celda multi -> criterio no satisfecho -> cadena exacta.
    const N: usize = 200;
    let mut store = make_store(N + 1);
    let mut i = 0usize;
    while i < N {
        store.x_slice_mut()[i] = 50.0;
        store.y_slice_mut()[i] = 50.0;
        store.mass_slice_mut()[i] = 1.0;
        i += 1;
    }
    // Cuerpo 200 dentro/borde de la celda (50.5, 50.5).
    store.x_slice_mut()[N] = 50.5;
    store.y_slice_mut()[N] = 50.5;
    store.mass_slice_mut()[N] = 1.0;
    store.live_count = N + 1;
    let li = store.live_indices_slice_mut();
    i = 0;
    while i < N + 1 {
        li[i] = i as i32;
        i += 1;
    }
    drop(li);

    let mut arena = QuadtreeArena::allocate(256).unwrap();
    assert!(
        build_quadtree(&store, &mut arena, 50.0, 50.0, 1.0).is_ok(),
        "root_half=1.0 < MIN_CELL_SIZE=2.0 -> LeafMulti"
    );
    assert!(compute_quadtree_masses(&mut arena, &store).is_ok());

    let root = unsafe { arena.read_node(arena.root_index as usize) };
    assert_eq!(root.kind, 3, "root debe ser LeafMulti (kind=3)");

    let g: f32 = 1500.0;
    let softening: f32 = 1.0;
    let theta: f32 = 0.3;
    compute_forces_barnes_hut_range(&arena, &store, 0, N + 1, g, softening, theta);

    unsafe {
        let fx = store.fx;
        let fy = store.fy;
        let (ex, ey) = bruto_esperado(&store, N, g, softening * softening);
        let ax = *fx.add(N);
        let ay = *fy.add(N);
        let escala = 1.0f32.max(ex.abs()).max(ey.abs());
        let rel = ((ax - ex).abs() + (ay - ey).abs()) / escala.max(1e-6);
        assert!(
            rel < 1e-4,
            "cuerpo 200 cerca: BH={},{} bruto={},{} rel={}",
            ax,
            ay,
            ex,
            ey,
            rel
        );
    }
}
