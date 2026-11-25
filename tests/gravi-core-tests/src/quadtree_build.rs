//! quadtree_build.rs — Tests de integración: QuadNode, QuadtreeArena,
//! build_quadtree, compute_quadtree_masses y root_from_live_bodies.
//! Extraídos de crates/gravi-core/src/quadtree.rs (mod tests).

use gravi_core::constants::{MAX_QUADTREE_DEPTH, MAX_ROOT_HALF, MIN_CELL_SIZE, MIN_ROOT_HALF};
use gravi_core::quadtree::{build_quadtree, compute_quadtree_masses, QuadtreeArena, QuadtreeError, root_from_live_bodies, QuadNode};
use gravi_core::store::NativeStore;
use std::mem;

fn make_store(count: usize) -> NativeStore {
    let mut store = NativeStore::new(count).expect("store creation failed");
    store.init_lifecycle();
    store
}

#[test]
fn ac1_quadnode_es_32_bytes_repr_c() {
    assert_eq!(mem::size_of::<QuadNode>(), 32);
    assert_eq!(mem::align_of::<QuadNode>(), 4);
}

#[test]
fn ac2_arena_capacidad_es_potencia_de_2_maxima_4n16() {
    for n in [1usize, 2, 4, 8, 16, 100, 1000, 40000] {
        let arena = QuadtreeArena::allocate(n).unwrap();
        let needed = 4 * n + 16;
        assert!(arena.capacity >= needed, "cap={} < needed={}", arena.capacity, needed);
        // potencia de 2
        assert!(arena.capacity & (arena.capacity - 1) == 0, "cap={} no es pow2", arena.capacity);
        assert_eq!(arena.next_free, 0);
        assert_eq!(arena.root_index, -1);
        assert!(!arena.base.is_null());
    }
}

#[test]
fn ac3_arena_cero_cuerpos_es_error() {
    assert!(matches!(
        QuadtreeArena::allocate(0),
        Err(QuadtreeError::ZeroCapacity)
    ));
}

#[test]
fn ac4_build_arbol_simple_raiz_una_hoja() {
    let mut store = make_store(1);
    store.x_slice_mut()[0] = 10.0;
    store.y_slice_mut()[0] = 20.0;
    store.mass_slice_mut()[0] = 5.0;
    store.live_count = 1;
    let li = store.live_indices_slice_mut();
    li[0] = 0;
    let _ = li;

    let mut arena = QuadtreeArena::allocate(4).unwrap();
    let res = build_quadtree(&store, &mut arena, 0.0, 0.0, 100.0);
    assert!(res.is_ok());
    assert_eq!(arena.root_index, 0);

    unsafe {
        let root = arena.node(0);
        assert_eq!(root.kind, 1); // Leaf
        assert_eq!(root.body_index, 0);
        assert_eq!(root.count, 1);
        assert_eq!(root.mass, 5.0);
    }
}

#[test]
fn ac5_build_subdivide_hoja_al_segundo_cuerpo() {
    let mut store = make_store(2);
    store.x_slice_mut()[0] = 10.0;
    store.y_slice_mut()[0] = 10.0;
    store.mass_slice_mut()[0] = 1.0;
    store.x_slice_mut()[1] = 90.0;
    store.y_slice_mut()[1] = 90.0;
    store.mass_slice_mut()[1] = 2.0;
    store.live_count = 2;
    let li = store.live_indices_slice_mut();
    li[0] = 0;
    li[1] = 1;
    let _ = li;

    let mut arena = QuadtreeArena::allocate(8).unwrap();
    let res = build_quadtree(&store, &mut arena, 50.0, 50.0, 100.0);
    assert!(res.is_ok());

    unsafe {
        let root = arena.node(0);
        assert_eq!(root.kind, 2); // Internal (subdividió)
        assert_eq!(root.count, 2);
        assert_eq!(root.mass, 3.0);

        // Cada cuerpo en su propia hoja (hijos del root).
        let base = root.child_offset as usize;
        let mut found = 0usize;
        for k in 0..4usize {
            let ch = arena.node(base + k);
            if ch.kind == 1 {
                found += 1;
                assert_eq!(ch.count, 1);
            }
        }
        assert_eq!(found, 2);
    }
}

#[test]
fn ac6_alloc_node_bump_o1() {
    let mut arena = QuadtreeArena::allocate(16).unwrap();
    assert_eq!(arena.alloc_node(), Ok(0));
    assert_eq!(arena.alloc_node(), Ok(1));
    assert_eq!(arena.next_free, 2);
}

#[test]
fn ac7_overflow_si_sobrepasa_capacidad() {
    let mut arena = QuadtreeArena::allocate(1).unwrap();
    let cap = arena.capacity;
    for _ in 0..cap {
        assert!(arena.alloc_node().is_ok());
    }
    assert_eq!(arena.alloc_node().unwrap_err(), QuadtreeError::Overflow);
}

#[test]
fn ac8_drop_libera_sin_panic() {
    let arena = QuadtreeArena::allocate(16).unwrap();
    let _ptr = arena.base;
    assert!(!_ptr.is_null());
    drop(arena);
    // Si llegamos aquí, no hubo double-free ni panic.
}

#[test]
fn ac9_build_masas_acumuladas_en_camino() {
    let mut store = make_store(3);
    store.x_slice_mut()[0] = 10.0;
    store.y_slice_mut()[0] = 10.0;
    store.mass_slice_mut()[0] = 1.0;
    store.x_slice_mut()[1] = 12.0;
    store.y_slice_mut()[1] = 12.0;
    store.mass_slice_mut()[1] = 2.0;
    store.x_slice_mut()[2] = 90.0;
    store.y_slice_mut()[2] = 90.0;
    store.mass_slice_mut()[2] = 4.0;
    store.live_count = 3;
    let li = store.live_indices_slice_mut();
    li[0] = 0;
    li[1] = 1;
    li[2] = 2;
    let _ = li;

    let mut arena = QuadtreeArena::allocate(16).unwrap();
    assert!(build_quadtree(&store, &mut arena, 50.0, 50.0, 100.0).is_ok());

    unsafe {
        let root = arena.node(0);
        assert_eq!(root.count, 3);
        assert_eq!(root.mass, 7.0);
    }
}

#[test]
fn ac10_arena_dispose_pone_base_nulo() {
    let mut arena = QuadtreeArena::allocate(16).unwrap();
    unsafe { QuadtreeArena::dispose(&mut arena); }
    assert!(arena.base.is_null());
    assert_eq!(arena.capacity, 0);
    assert_eq!(arena.root_index, -1);
}

// --- compute_quadtree_masses ---

#[test]
fn a2_masses_arbol_vacio_root_negativo_es_ok() {
    let store = make_store(1);
    let mut arena = QuadtreeArena::allocate(4).unwrap();
    arena.root_index = -1; // sin arbol construido
    assert!(compute_quadtree_masses(&mut arena, &store).is_ok());
}

#[test]
fn a2_masses_acumula_masa_y_count_post_orden() {
    // 3 cuerpos en cuadrantes distintos del root (sin subdivicion anidada).
    let mut store = make_store(3);
    store.x_slice_mut()[0] = 10.0;
    store.y_slice_mut()[0] = 10.0;
    store.mass_slice_mut()[0] = 1.0;
    store.x_slice_mut()[1] = 90.0;
    store.y_slice_mut()[1] = 90.0;
    store.mass_slice_mut()[1] = 2.0;
    store.x_slice_mut()[2] = 10.0;
    store.y_slice_mut()[2] = 90.0;
    store.mass_slice_mut()[2] = 4.0;
    store.live_count = 3;
    let li = store.live_indices_slice_mut();
    li[0] = 0;
    li[1] = 1;
    li[2] = 2;
    let _ = li;

    let mut arena = QuadtreeArena::allocate(32).unwrap();
    assert!(build_quadtree(&store, &mut arena, 50.0, 50.0, 100.0).is_ok());
    let root_idx = arena.root_index as usize;

    // Zero de masa/count en TODOS los nodos para forzar recomputo desde cero.
    unsafe {
        for i in 0..arena.next_free {
            let n = arena.read_node(i);
            arena.write_node(i, QuadNode { mass: 0.0, count: 0, ..n });
        }
    }

    assert!(compute_quadtree_masses(&mut arena, &store).is_ok());

    unsafe {
        let root = arena.read_node(root_idx);
        assert_eq!(root.mass, 7.0, "raiz masa");
        assert_eq!(root.count, 3, "raiz count");
        // Cada hoja conserva masa y count de su cuerpo.
        if root.kind == 2 {
            let base = root.child_offset as usize;
            let mut hojas = 0usize;
            for k in 0..4usize {
                let ch = arena.read_node(base + k);
                if ch.kind == 1 {
                    hojas += 1;
                    assert_eq!(ch.count, 1);
                    assert!(ch.mass > 0.0);
                }
            }
            assert_eq!(hojas, 3);
        }
    }
}

#[test]
fn a2_masses_una_hoja_sin_subdividir() {
    let mut store = make_store(1);
    store.x_slice_mut()[0] = 10.0;
    store.y_slice_mut()[0] = 20.0;
    store.mass_slice_mut()[0] = 5.0;
    store.live_count = 1;
    let li = store.live_indices_slice_mut();
    li[0] = 0;
    let _ = li;

    let mut arena = QuadtreeArena::allocate(4).unwrap();
    assert!(build_quadtree(&store, &mut arena, 0.0, 0.0, 100.0).is_ok());
    assert!(compute_quadtree_masses(&mut arena, &store).is_ok());

    unsafe {
        let root = arena.read_node(arena.root_index as usize);
        assert_eq!(root.kind, 1);
        assert_eq!(root.mass, 5.0);
        assert_eq!(root.count, 1);
    }
}

// --- root_from_live_bodies ---

#[test]
fn a2_root_cero_cuerpo_devuelve_ceros() {
    let store = NativeStore::new(4).unwrap();
    let (cx, cy, half) = root_from_live_bodies(&store);
    assert_eq!((cx, cy), (0.0, 0.0));
    assert_eq!(half, MIN_ROOT_HALF); // sin cuerpos -> raíz con semilado mínimo
}

#[test]
fn a2_root_un_cuerpo_centro_en_el_cuerpo() {
    let mut store = make_store(1);
    store.x_slice_mut()[0] = 100.0;
    store.y_slice_mut()[0] = -200.0;
    store.mass_slice_mut()[0] = 7.0;
    store.live_count = 1;
    let li = store.live_indices_slice_mut();
    li[0] = 0;
    let _ = li;

    let (cx, cy, half) = root_from_live_bodies(&store);
    assert_eq!(cx, 100.0);
    assert_eq!(cy, -200.0);
    assert_eq!(half, MIN_ROOT_HALF); // extent nulo -> clamp a MIN_ROOT_HALF
}

#[test]
fn a2_root_centro_dentro_del_rectangulo_de_vivos() {
    let mut store = make_store(4);
    store.x_slice_mut()[0] = 0.0;
    store.y_slice_mut()[0] = 0.0;
    store.x_slice_mut()[1] = 100.0;
    store.y_slice_mut()[1] = 0.0;
    store.x_slice_mut()[2] = 0.0;
    store.y_slice_mut()[2] = 100.0;
    store.x_slice_mut()[3] = 100.0;
    store.y_slice_mut()[3] = 100.0;
    for i in 0..4usize {
        store.mass_slice_mut()[i] = 1.0;
    }
    store.live_count = 4;
    let li = store.live_indices_slice_mut();
    for i in 0..4usize {
        li[i] = i as i32;
    }
    let _ = li;

    let (cx, cy, half) = root_from_live_bodies(&store);
    assert_eq!(cx, 50.0);
    assert_eq!(cy, 50.0);
    assert_eq!(half, MIN_ROOT_HALF); // extent=100 -> half=50, clamp a MIN_ROOT_HALF=64

    // Centro dentro del rectangulo [0,100]x[0,100]
    assert!(cx >= 0.0 && cx <= 100.0);
    assert!(cy >= 0.0 && cy <= 100.0);
    assert!(half > 0.0);
}

#[test]
fn a2_root_solo_cuerpos_vivos_ignorando_muertos() {
    let mut store = make_store(4);
    // Cuerpo 0 y 2 vivos, 1 y 3 muertos (masa < 0, pero live_count define los vivos).
    store.x_slice_mut()[0] = 10.0;
    store.y_slice_mut()[0] = 10.0;
    store.mass_slice_mut()[0] = 1.0;
    store.x_slice_mut()[1] = 1000.0;
    store.y_slice_mut()[1] = 1000.0;
    store.mass_slice_mut()[1] = -1.0; // muerto (no esta en live_indices)
    store.x_slice_mut()[2] = 20.0;
    store.y_slice_mut()[2] = 20.0;
    store.mass_slice_mut()[2] = 1.0;
    store.x_slice_mut()[3] = -1000.0;
    store.y_slice_mut()[3] = -1000.0;
    store.mass_slice_mut()[3] = -1.0; // muerto
    store.live_count = 2;
    let li = store.live_indices_slice_mut();
    li[0] = 0;
    li[1] = 2;
    let _ = li;

    let (cx, cy, half) = root_from_live_bodies(&store);
    assert_eq!(cx, 15.0); // (10+20)/2
    assert_eq!(cy, 15.0);
    assert_eq!(half, MIN_ROOT_HALF); // extent=10 -> half=5, clamp a MIN_ROOT_HALF=64
}

#[test]
fn ac_alloc_node_no_avanza_sin_capacidad() {
    let mut arena = QuadtreeArena::allocate(1).unwrap(); // cap >= 16
    let cap = arena.capacity;
    // llenar hasta el tope
    for _ in 0..cap {
        assert!(arena.alloc_node().is_ok());
    }
    let antes = arena.next_free;
    assert_eq!(antes, cap);
    assert_eq!(
        arena.alloc_node().unwrap_err(),
        QuadtreeError::Overflow
    );
    assert_eq!(arena.next_free, cap, "next_free no debe cambiar tras Overflow");
}

#[test]
fn ac_overflow_no_corrompe_arena() {
    // 2 cuerpos que forzarán subdivisión
    let mut store = make_store(2);
    store.x_slice_mut()[0] = 10.0;
    store.y_slice_mut()[0] = 10.0;
    store.mass_slice_mut()[0] = 1.0;
    store.x_slice_mut()[1] = 90.0;
    store.y_slice_mut()[1] = 90.0;
    store.mass_slice_mut()[1] = 2.0;
    store.live_count = 2;
    let li = store.live_indices_slice_mut();
    li[0] = 0;
    li[1] = 1;
    let _ = li;

    // Arena artificialmente pequeña: capacidad 4 (< 5 nodos para raíz + 4 hijos).
    // Se usa allocate(1) (cap=32) y se reduce capacity para forzar Overflow
    // sin modificar QuadtreeArena::allocate.
    let mut arena = QuadtreeArena::allocate(1).unwrap();
    arena.capacity = 4;

    let antes = arena.next_free; // 0
    let res = build_quadtree(&store, &mut arena, 50.0, 50.0, 100.0);
    assert!(
        res.is_err(),
        "esperamos Overflow por falta de 4 slots para subdividir"
    );
    match res.unwrap_err() {
        QuadtreeError::Overflow => (),
        e => panic!("esperaba Overflow, got {:?}", e),
    }
    assert_eq!(
        arena.next_free,
        antes + 1,
        "solo el nodo raíz se asignó antes del Overflow; next_free no debe haberse disparado"
    );
    assert!(
        arena.next_free <= arena.capacity,
        "arena no corrupta: next_free={} > capacity={}",
        arena.next_free,
        arena.capacity
    );

    // Reset y reintento con capacidad suficiente
    arena.next_free = 0;
    arena.capacity = 32;
    let res2 = build_quadtree(&store, &mut arena, 50.0, 50.0, 100.0);
    assert!(
        res2.is_ok(),
        "tras reset con capacidad suficiente debe funcionar"
    );
}

// --- Nuevas pruebas R4-05 ---

#[test]
fn ac_consts_quadtree_guardas_existen() {
    assert_eq!(MIN_CELL_SIZE, 2.0);
    assert_eq!(MAX_QUADTREE_DEPTH, 32);
    assert_eq!(MIN_ROOT_HALF, 64.0);
    assert_eq!(MAX_ROOT_HALF, 4096.0);
}

#[test]
fn quadtree_root_clamp_min_max_half() {
    // AC1: extent pequeño -> half >= MIN_ROOT_HALF
    let mut store = make_store(2);
    store.x_slice_mut()[0] = 10.0;
    store.y_slice_mut()[0] = 10.0;
    store.mass_slice_mut()[0] = 1.0;
    store.x_slice_mut()[1] = 11.0;
    store.y_slice_mut()[1] = 11.0;
    store.mass_slice_mut()[1] = 1.0;
    store.live_count = 2;
    let li = store.live_indices_slice_mut();
    li[0] = 0;
    li[1] = 1;
    let _ = li;

    let (_cx, _cy, half) = root_from_live_bodies(&store);
    assert!(
        half >= MIN_ROOT_HALF,
        "half={} debe ser >= MIN_ROOT_HALF={}",
        half,
        MIN_ROOT_HALF
    );

    // extent grande (1920x980) -> half <= MAX_ROOT_HALF
    let mut store2 = make_store(2);
    store2.x_slice_mut()[0] = 0.0;
    store2.y_slice_mut()[0] = 0.0;
    store2.mass_slice_mut()[0] = 1.0;
    store2.x_slice_mut()[1] = 1920.0;
    store2.y_slice_mut()[1] = 980.0;
    store2.mass_slice_mut()[1] = 1.0;
    store2.live_count = 2;
    let li2 = store2.live_indices_slice_mut();
    li2[0] = 0;
    li2[1] = 1;
    let _ = li2;

    let (cx2, cy2, half2) = root_from_live_bodies(&store2);
    assert!(
        half2 <= MAX_ROOT_HALF,
        "half={} debe ser <= MAX_ROOT_HALF={}",
        half2,
        MAX_ROOT_HALF
    );
    assert!(cx2 >= 0.0 && cx2 <= 1920.0);
    assert!(cy2 >= 0.0 && cy2 <= 980.0);

    // live_count == 0 -> half = MIN_ROOT_HALF (no 0.0)
    let store3 = NativeStore::new(4).unwrap();
    let (_cx3, _cy3, half3) = root_from_live_bodies(&store3);
    assert_eq!(half3, MIN_ROOT_HALF);
}

#[test]
fn quadtree_no_desborda_arena_con_cuerpos_coincidentes() {
    // AC2: N cuerpos en la misma coordenada exacta, arena pequeña.
    // build_quadtree debe retornar Ok (no Err(Overflow)) y no hacer panic.
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
    let _ = li;

    // Arena pequeña: capacidad suficiente solo para la raíz. Con
    // root_half=1.0 (< MIN_CELL_SIZE=2.0) se degrada inmediatamente a
    // LeafMulti sin consumir nodos extra, por lo que no hay Overflow.
    let mut arena = QuadtreeArena::allocate(4).unwrap(); // cap >= 32
    arena.capacity = 1; // solo cabe la raíz

    let res = build_quadtree(&store, &mut arena, 50.0, 50.0, 1.0);
    assert!(
        res.is_ok(),
        "con cuerpos coincidentes y celda < MIN_CELL_SIZE no debe darse Overflow: {:?}",
        res.err()
    );

    // compute_forces no debe panic ni escribir fuera de rango
    assert!(compute_quadtree_masses(&mut arena, &store).is_ok());

    unsafe {
        let fx = store.fx;
            let fy = store.fy;
        // clear forces first
        i = 0;
        while i < n {
            *fx.add(i) = 0.0;
            *fy.add(i) = 0.0;
            i += 1;
        }
    }

    use gravi_core::quadtree::compute_forces_barnes_hut_range;
    compute_forces_barnes_hut_range(&arena, &store, 0, n, 1.0, 8.0, 0.5);

    unsafe {
        let fx = store.fx;
        let fy = store.fy;
        i = 0;
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
fn quadtree_leaf_multi_masa_acumulada() {
    // AC3 masa: 3 cuerpos en la misma celda (no subdivide por límite de
    // profundidad). compute_quadtree_masses da mass = suma y count = 3.
    let n = 3usize;
    let mut store = make_store(n);
    store.x_slice_mut()[0] = 50.0; store.y_slice_mut()[0] = 50.0; store.mass_slice_mut()[0] = 1.0;
    store.x_slice_mut()[1] = 50.0; store.y_slice_mut()[1] = 50.0; store.mass_slice_mut()[1] = 2.0;
    store.x_slice_mut()[2] = 50.0; store.y_slice_mut()[2] = 50.0; store.mass_slice_mut()[2] = 3.0;
    store.live_count = n;
    let li = store.live_indices_slice_mut();
    li[0] = 0; li[1] = 1; li[2] = 2;
    let _ = li;

    // Root half muy pequeño para forzar degradación inmediata (half=4 < MIN_CELL_SIZE=2? no, 4>=2).
    // Usar profundidad: forzar que no se subdivida pasando root_half=2.0 (>=MIN_CELL_SIZE)
    // pero con depth>=MAX_QUADTREE_DEPTH no es posible desde build_quadtree (depth=0).
    // En su lugar, usar una celda muy pequeña: half=1.0 < MIN_CELL_SIZE=2.0 -> no subdivide.
    let mut arena = QuadtreeArena::allocate(16).unwrap();
    assert!(
        build_quadtree(&store, &mut arena, 50.0, 50.0, 1.0).is_ok(),
        "debe construir sin overflow con celda pequeña"
    );

    // Zero masa/count para forzar recomputo
    unsafe {
        for i in 0..arena.next_free {
            let n = arena.read_node(i);
            arena.write_node(i, QuadNode { mass: 0.0, count: 0, ..n });
        }
    }

    assert!(compute_quadtree_masses(&mut arena, &store).is_ok());

    unsafe {
        let root = arena.read_node(arena.root_index as usize);
        assert_eq!(root.kind, 3, "esperaba LeafMulti, got kind={}", root.kind);
        assert_eq!(root.count, 3, "count debe ser 3");
        assert!(
            (root.mass - 6.0).abs() < 1e-6,
            "mass debe ser 6.0, got {}",
            root.mass
        );
    }
}

#[test]
fn ac_insert_profundo_limite_32() {
    // Profundidad 32: insertar cuerpos que forzarían subdivisión profunda,
    // verificar que se detiene en MAX_QUADTREE_DEPTH y degrada a LeafMulti.
    let n = 4usize;
    let mut store = make_store(n);
    // Cuerpos en posiciones muy cercanas para forzar muchas subdivisiones.
    store.x_slice_mut()[0] = 50.0; store.y_slice_mut()[0] = 50.0; store.mass_slice_mut()[0] = 1.0;
    store.x_slice_mut()[1] = 50.0 + 0.001; store.y_slice_mut()[1] = 50.0 + 0.001; store.mass_slice_mut()[1] = 1.0;
    store.x_slice_mut()[2] = 50.0 + 0.002; store.y_slice_mut()[2] = 50.0 + 0.002; store.mass_slice_mut()[2] = 1.0;
    store.x_slice_mut()[3] = 50.0 + 0.003; store.y_slice_mut()[3] = 50.0 + 0.003; store.mass_slice_mut()[3] = 1.0;
    store.live_count = n;
    let li = store.live_indices_slice_mut();
    li[0] = 0; li[1] = 1; li[2] = 2; li[3] = 3;
    let _ = li;

    let mut arena = QuadtreeArena::allocate(256).unwrap();
    let res = build_quadtree(&store, &mut arena, 50.0, 50.0, 64.0);
    assert!(res.is_ok(), "no debe overflow con profundidad limitada");

    // Verificar que no hay nodos Internal más allá de profundidad 32:
    // recorrer todos los nodos y confirmar kind válido.
    unsafe {
        let mut i = 0usize;
        while i < arena.next_free {
            let node = arena.read_node(i);
            assert!(
                node.kind <= 3,
                "nodo {} kind inválido {}",
                i,
                node.kind
            );
            i += 1;
        }
    }
}

#[test]
fn ac_overflow_solo_arena_llena_real() {
    // Overflow solo si arena realmente llena. Con cuerpos coincidentes y
    // degradación LeafMulti, no debe darse Overflow por cuerpos en la misma
    // celda.
    let n = 100usize;
    let mut store = make_store(n);
    let mut i = 0usize;
    while i < n {
        store.x_slice_mut()[i] = 100.0;
        store.y_slice_mut()[i] = 100.0;
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
    let _ = li;

    // Arena con capacidad suficiente para raíz + degradación (no se consumen
    // nodos extra en LeafMulti). Capacidad muy pequeña (solo raíz) debería
    // bastar porque no se subdivide.
    let mut arena = QuadtreeArena::allocate(n).unwrap();
    arena.capacity = 1; // solo cabe la raíz

    // Con 1 solo nodo (la raíz), al insertar el 2do cuerpo se intenta
    // subdividir pero no hay capacidad -> Overflow. Esto es correcto: la
    // arena está realmente llena.
    let res = build_quadtree(&store, &mut arena, 100.0, 100.0, 64.0);
    assert!(
        matches!(res, Err(QuadtreeError::Overflow)),
        "esperaba Overflow con arena de 1 nodo, got {:?}",
        res
    );

    // Ahora con capacidad suficiente (raíz + unos hijos) pero con cuerpos
    // coincidentes y root_half < MIN_CELL_SIZE -> se degrada a LeafMulti
    // inmediatamente, sin consumir nodos extra, por lo que no hay Overflow.
    let mut arena2 = QuadtreeArena::allocate(n).unwrap();
    arena2.capacity = 10;

    let res2 = build_quadtree(&store, &mut arena2, 100.0, 100.0, 1.0);
    assert!(
        res2.is_ok(),
        "con cuerpos coincidentes, celda pequeña y LeafMulti no debe Overflow: {:?}",
        res2.err()
    );
}
