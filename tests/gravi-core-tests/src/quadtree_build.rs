//! quadtree_build.rs — Tests de integración: QuadNode, QuadtreeArena,
//! build_quadtree, compute_quadtree_masses y root_from_live_bodies.
//! Extraídos de crates/gravi-core/src/quadtree.rs (mod tests).

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
    store.x_slice()[0] = 10.0;
    store.y_slice()[0] = 20.0;
    store.mass_slice()[0] = 5.0;
    store.live_count = 1;
    let li = store.live_indices_slice();
    li[0] = 0;

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
    store.x_slice()[0] = 10.0;
    store.y_slice()[0] = 10.0;
    store.mass_slice()[0] = 1.0;
    store.x_slice()[1] = 90.0;
    store.y_slice()[1] = 90.0;
    store.mass_slice()[1] = 2.0;
    store.live_count = 2;
    let li = store.live_indices_slice();
    li[0] = 0;
    li[1] = 1;

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
    store.x_slice()[0] = 10.0;
    store.y_slice()[0] = 10.0;
    store.mass_slice()[0] = 1.0;
    store.x_slice()[1] = 12.0;
    store.y_slice()[1] = 12.0;
    store.mass_slice()[1] = 2.0;
    store.x_slice()[2] = 90.0;
    store.y_slice()[2] = 90.0;
    store.mass_slice()[2] = 4.0;
    store.live_count = 3;
    let li = store.live_indices_slice();
    li[0] = 0;
    li[1] = 1;
    li[2] = 2;

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
    store.x_slice()[0] = 10.0;
    store.y_slice()[0] = 10.0;
    store.mass_slice()[0] = 1.0;
    store.x_slice()[1] = 90.0;
    store.y_slice()[1] = 90.0;
    store.mass_slice()[1] = 2.0;
    store.x_slice()[2] = 10.0;
    store.y_slice()[2] = 90.0;
    store.mass_slice()[2] = 4.0;
    store.live_count = 3;
    let li = store.live_indices_slice();
    li[0] = 0;
    li[1] = 1;
    li[2] = 2;

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
    store.x_slice()[0] = 10.0;
    store.y_slice()[0] = 20.0;
    store.mass_slice()[0] = 5.0;
    store.live_count = 1;
    let li = store.live_indices_slice();
    li[0] = 0;

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
    assert_eq!((cx, cy, half), (0.0, 0.0, 0.0));
}

#[test]
fn a2_root_un_cuerpo_centro_en_el_cuerpo() {
    let mut store = make_store(1);
    store.x_slice()[0] = 100.0;
    store.y_slice()[0] = -200.0;
    store.mass_slice()[0] = 7.0;
    store.live_count = 1;
    let li = store.live_indices_slice();
    li[0] = 0;

    let (cx, cy, half) = root_from_live_bodies(&store);
    assert_eq!(cx, 100.0);
    assert_eq!(cy, -200.0);
    assert_eq!(half, 0.0); // extent nulo
}

#[test]
fn a2_root_centro_dentro_del_rectangulo_de_vivos() {
    let mut store = make_store(4);
    store.x_slice()[0] = 0.0;
    store.y_slice()[0] = 0.0;
    store.x_slice()[1] = 100.0;
    store.y_slice()[1] = 0.0;
    store.x_slice()[2] = 0.0;
    store.y_slice()[2] = 100.0;
    store.x_slice()[3] = 100.0;
    store.y_slice()[3] = 100.0;
    for i in 0..4usize {
        store.mass_slice()[i] = 1.0;
    }
    store.live_count = 4;
    let li = store.live_indices_slice();
    for i in 0..4usize {
        li[i] = i as i32;
    }

    let (cx, cy, half) = root_from_live_bodies(&store);
    assert_eq!(cx, 50.0);
    assert_eq!(cy, 50.0);
    assert_eq!(half, 50.0); // max(extent_x, extent_y)/2 = 100/2

    // Centro dentro del rectangulo [0,100]x[0,100]
    assert!(cx >= 0.0 && cx <= 100.0);
    assert!(cy >= 0.0 && cy <= 100.0);
    assert!(half > 0.0);
}

#[test]
fn a2_root_solo_cuerpos_vivos_ignorando_muertos() {
    let mut store = make_store(4);
    // Cuerpo 0 y 2 vivos, 1 y 3 muertos (masa < 0, pero live_count define los vivos).
    store.x_slice()[0] = 10.0;
    store.y_slice()[0] = 10.0;
    store.mass_slice()[0] = 1.0;
    store.x_slice()[1] = 1000.0;
    store.y_slice()[1] = 1000.0;
    store.mass_slice()[1] = -1.0; // muerto (no esta en live_indices)
    store.x_slice()[2] = 20.0;
    store.y_slice()[2] = 20.0;
    store.mass_slice()[2] = 1.0;
    store.x_slice()[3] = -1000.0;
    store.y_slice()[3] = -1000.0;
    store.mass_slice()[3] = -1.0; // muerto
    store.live_count = 2;
    let li = store.live_indices_slice();
    li[0] = 0;
    li[1] = 2;

    let (cx, cy, half) = root_from_live_bodies(&store);
    assert_eq!(cx, 15.0); // (10+20)/2
    assert_eq!(cy, 15.0);
    assert_eq!(half, 5.0); // extent 10 / 2
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
    store.x_slice()[0] = 10.0;
    store.y_slice()[0] = 10.0;
    store.mass_slice()[0] = 1.0;
    store.x_slice()[1] = 90.0;
    store.y_slice()[1] = 90.0;
    store.mass_slice()[1] = 2.0;
    store.live_count = 2;
    let li = store.live_indices_slice();
    li[0] = 0;
    li[1] = 1;

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
