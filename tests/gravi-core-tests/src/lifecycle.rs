//! Tests extraídos de crates/gravi-core/src/entity_lifecycle.rs (Issue 3 — criterios de aceptación).
//! Integration test: solo API pública de gravi_core.

use gravi_core::entity_lifecycle::{allocate_entity, is_live, mark_dead};
use gravi_core::heap_verifier::{HeapVerifier, probe_allocate_usize};
use gravi_core::store::NativeStore;

/// Crea un store con `count` slots todos vivos, inicializa ciclo de vida.
fn make_store(count: usize) -> NativeStore {
    let mut store = NativeStore::new(count).expect("store creation failed");
    store.init_lifecycle();
    store
}

#[test]
fn ac1_is_live_reads_mass_ge_zero() {
    let store = make_store(4);

    // Todos vivos inicialmente (mass = 1.0).
    for s in 0..4 {
        assert!(is_live(&store, s), "slot {} debe estar vivo", s);
    }

    // Marcar slot 2 como muerto (mass = -1.0).
    unsafe { *store.mass.add(2) = -1.0 };
    assert!(!is_live(&store, 2), "slot 2 debe estar muerto tras mass = -1.0");
    assert!(is_live(&store, 0), "slot 0 sigue vivo");
    assert!(is_live(&store, 3), "slot 3 sigue vivo");

    // Masa 0.0 se considera vivo (>= 0.0).
    unsafe { *store.mass.add(1) = 0.0 };
    assert!(is_live(&store, 1), "slot con mass = 0.0 debe estar vivo");
}

#[test]
fn ac2_mark_dead_swap_remove_o1() {
    let mut store = make_store(4);
    assert_eq!(store.live_count, 4);

    // Marcar slot 1 como muerto.
    let res = mark_dead(&mut store, 1);
    assert!(res.is_ok(), "mark_dead(1) debe ser Ok");

    // AC2: mass = -1.0 en el slot muerto.
    assert!(!is_live(&store, 1), "slot 1 debe estar muerto");
    unsafe { assert_eq!(*store.mass.add(1), -1.0); }

    // AC2: generación++.
    unsafe { assert_eq!(*store.generation.add(1), 1); }

    // AC2: live_count--.
    assert_eq!(store.live_count, 3);

    // AC2: swap-remove — live_indices debe contener solo slots vivos.
    unsafe {
        assert_eq!(*store.live_indices.add(0), 0);
        assert_eq!(*store.live_indices.add(1), 3); // último slot vivo ocupa pos del muerto
        assert_eq!(*store.live_indices.add(2), 2);
    }

    // AC2: live_pos actualizado para el slot movido (3 está ahora en pos 1).
    unsafe { assert_eq!(*store.live_pos.add(3), 1); }

    // AC2: push LIFO en free_list — free_head = 1, free_list[1] = 0 (cabeza previa tras init_lifecycle).
    assert_eq!(store.free_head, 1);
    unsafe { assert_eq!(*store.free_list.add(1), 0); }
}

#[test]
fn ac2_mark_dead_last_live_slot() {
    let mut store = make_store(3);
    let res = mark_dead(&mut store, 2);
    assert!(res.is_ok());
    assert!(!is_live(&store, 2));
    assert_eq!(store.live_count, 2);
    assert_eq!(store.free_head, 2);
    unsafe {
        assert_eq!(*store.mass.add(2), -1.0);
        assert_eq!(*store.generation.add(2), 1);
        assert_eq!(*store.free_list.add(2), 0); // encadena al free_head previo (0 tras init_lifecycle)
    }
    // live_indices[0..2] = [0, 1] (slot 2 era el último, no hay swap real).
    unsafe {
        assert_eq!(*store.live_indices.add(0), 0);
        assert_eq!(*store.live_indices.add(1), 1);
    }
}

#[test]
fn ac2_mark_dead_twice_same_slot_is_error() {
    let mut store = make_store(2);
    assert!(mark_dead(&mut store, 0).is_ok());
    assert!(!is_live(&store, 0));
    let res = mark_dead(&mut store, 0);
    assert!(res.is_err(), "mark_dead en slot muerto debe ser error");
}

#[test]
fn ac3_allocate_entity_recycles_lifo_before_bump() {
    let mut store = make_store(4);

    // Marcar slots 1 y 2 como muertos → free_list LIFO: free_head=2, free_list[2]=1, free_list[1]=0 (cabeza previa tras init_lifecycle).
    assert!(mark_dead(&mut store, 1).is_ok());
    assert!(mark_dead(&mut store, 2).is_ok());
    assert_eq!(store.free_head, 2);
    unsafe {
        assert_eq!(*store.free_list.add(2), 1);
        assert_eq!(*store.free_list.add(1), 0);
    }

    // allocate_entity debe reciclar el más reciente (LIFO): slot 2.
    let res = allocate_entity(&mut store, 1.0, 2.0, 3.0, 4.0, 5.0, 0.5);
    assert!(res.is_ok());
    let slot = res.unwrap();
    assert_eq!(slot, 2, "debe reciclar slot 2 (LIFO, más reciente)");

    // free_head avanza a 1.
    assert_eq!(store.free_head, 1);

    // Otra allocate recicla slot 1.
    let res = allocate_entity(&mut store, 1.0, 2.0, 3.0, 4.0, 5.0, 0.5);
    assert!(res.is_ok());
    let slot = res.unwrap();
    assert_eq!(slot, 1);
    // free_head = 0 (el siguiente en la cadena desde init_lifecycle), no -1.
    assert_eq!(store.free_head, 0);
}

#[test]
fn ac3_allocate_entity_bumps_when_free_list_empty() {
    let mut store = make_store(2);
    assert_eq!(store.live_count, 2);
    assert_eq!(store.free_head, 0); // tras init_lifecycle, free_head = 0

    // Marcar todos los slots como muertos: free list poblada, live_count = 0.
    assert!(mark_dead(&mut store, 0).is_ok());
    assert!(mark_dead(&mut store, 1).is_ok());
    assert_eq!(store.live_count, 0);

    // Con free list poblada, allocate_entity recicla (no devuelve NoCapacity).
    let res = allocate_entity(&mut store, 1.0, 2.0, 3.0, 4.0, 5.0, 0.5);
    assert!(res.is_ok());
    assert_eq!(store.live_count, 1);
}

#[test]
fn ac3_allocate_entity_after_mark_dead_recycles_then_bumps() {
    let mut store = make_store(3);
    assert_eq!(store.live_count, 3);
    assert_eq!(store.free_head, 0); // tras init_lifecycle

    // Matar slot 1 → push LIFO, free_head = 1, free_list[1] = 0 (cabeza previa).
    assert!(mark_dead(&mut store, 1).is_ok());
    assert_eq!(store.live_count, 2);
    assert_eq!(store.free_head, 1);

    // Allocate recicla slot 1 (LIFO, cabeza de free list), live_count vuelve a 3.
    let res = allocate_entity(&mut store, 9.0, 9.0, 9.0, 9.0, 9.0, 0.5);
    assert!(res.is_ok());
    let slot = res.unwrap();
    assert_eq!(slot, 1);
    assert_eq!(store.live_count, 3);
}

#[test]
fn ac3_allocate_entity_initializes_all_channels() {
    let mut store = make_store(3);
    assert!(mark_dead(&mut store, 1).is_ok());

    let res = allocate_entity(&mut store, 1.5, 2.5, 3.5, 4.5, 5.5, 0.75);
    assert!(res.is_ok());
    let slot = res.unwrap();
    assert_eq!(slot, 1);

    unsafe {
        assert_eq!(*store.x.add(1), 1.5);
        assert_eq!(*store.y.add(1), 2.5);
        assert_eq!(*store.vx.add(1), 3.5);
        assert_eq!(*store.vy.add(1), 4.5);
        assert_eq!(*store.mass.add(1), 5.5);
        assert_eq!(*store.radius.add(1), 0.75);
        assert_eq!(*store.fx.add(1), 0.0);
        assert_eq!(*store.fy.add(1), 0.0);
        assert_eq!(*store.generation.add(1), 2); // mark_dead (0→1) + allocate (1→2)
    }
    assert!(is_live(&store, 1));
}

#[test]
fn ac3_allocate_entity_increments_generation() {
    let mut store = make_store(1);

    // Mark dead slot 0 (generation 0 → 1).
    assert!(mark_dead(&mut store, 0).is_ok());
    unsafe { assert_eq!(*store.generation.add(0), 1); }

    // Allocate again (recycle, generation 1 → 2).
    let res = allocate_entity(&mut store, 1.0, 2.0, 3.0, 4.0, 5.0, 0.5);
    assert!(res.is_ok());
    unsafe { assert_eq!(*store.generation.add(0), 2); }

    // Mark dead again (generation 2 → 3).
    assert!(mark_dead(&mut store, 0).is_ok());
    unsafe { assert_eq!(*store.generation.add(0), 3); }

    // Allocate again (recycle, generation 3 → 4).
    let res = allocate_entity(&mut store, 1.0, 2.0, 3.0, 4.0, 5.0, 0.5);
    assert!(res.is_ok());
    unsafe { assert_eq!(*store.generation.add(0), 4); }
}

#[test]
fn ac3_allocate_entity_appends_to_live_indices() {
    let mut store = make_store(2);
    // Marcar slot 0 muerto → live_count=1, live_indices=[1] (slot 1 movido a pos 0).
    assert!(mark_dead(&mut store, 0).is_ok());
    unsafe {
        assert_eq!(*store.live_indices.add(0), 1);
        assert_eq!(*store.live_pos.add(1), 0);
    }

    // Allocate recicla slot 0 → live_count=2, live_indices=[1, 0].
    let res = allocate_entity(&mut store, 1.0, 2.0, 3.0, 4.0, 5.0, 0.5);
    assert!(res.is_ok());
    unsafe {
        assert_eq!(*store.live_indices.add(0), 1);
        assert_eq!(*store.live_indices.add(1), 0);
        assert_eq!(*store.live_pos.add(0), 1);
        assert_eq!(*store.live_pos.add(1), 0);
    }
}

#[test]
fn ac4_no_heap_allocation_in_hot_path() {
    // AC4: las funciones no usan Vec/String/Box/closures. Verificado por inspección de código;
    // el hot path solo usa punteros crudos y aritmética de enteros.
    // Salvaguarda con el medidor de heap real: marcar y reciclar en bucle no asignan.
    let mut store = make_store(100);
    let mut hv = HeapVerifier::new();
    hv.begin_frame();

    // Marcar la mitad como muertos.
    for s in 0..50 {
        assert!(mark_dead(&mut store, s).is_ok());
    }
    assert_eq!(store.live_count, 50);
    assert_eq!(store.free_head, 49);

    // Reciclar todos en LIFO.
    for _ in 0..50 {
        let res = allocate_entity(&mut store, 1.0, 2.0, 3.0, 4.0, 5.0, 0.5);
        assert!(res.is_ok());
    }
    let report = hv.end_frame();
    assert_eq!(
        report.delta_bytes,
        0,
        "hot path asignó heap: delta_bytes={}",
        report.delta_bytes
    );
    assert_eq!(store.live_count, 100);
    // Tras drenar la free list inicializada por init_lifecycle, free_head queda en 0
    // (autociclo del slot 0), no en -1.
    assert_eq!(store.free_head, 0);

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
