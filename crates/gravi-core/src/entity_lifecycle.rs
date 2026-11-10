//! EntityLifecycle — Ciclo de vida O(1) de entidades sobre slots compactos.
//!
//! Zero-allocation en el hot path: solo punteros de canal y aritmética de enteros;
//! sin heap, sin LINQ, sin closures.
//!
//! Invariantes:
//! - `count` es capacidad fija; nunca cambia tras la asignación.
//! - Slot vivo si y solo si `mass[slot] >= 0.0` (centinela `-1.0` = muerto; `0.0` = vivo sin masa).
//! - `live_indices[0..live_count)` contiene los slots vivos; `live_pos[slot]`
//!   es su posición en `live_indices`.
//! - `free_head` es la cabeza de pila LIFO de slots libres (encadenados por
//!   `free_list[slot]`); `-1` = vacía.
//! - `generation[slot]` se incrementa en cada `mark_dead` y `allocate_entity`.

use crate::store::NativeStore;

/// Errores del ciclo de vida de entidades.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LifecycleError {
    /// El slot no está vivo (precondición de `mark_dead` violada).
    SlotNotLive,
    /// No hay capacidad disponible (free list vacía y `live_count == count`).
    NoCapacity,
}

/// Devuelve `true` si y solo si el slot está vivo (`mass >= 0.0`).
///
/// Zero-allocation: solo una lectura de puntero y comparación.
#[inline(always)]
pub fn is_live(store: &NativeStore, slot: i32) -> bool {
    debug_assert!(!store.base_address.is_null(), "store no inicializado");
    debug_assert!(
        slot >= 0 && slot < store.count as i32,
        "slot fuera de rango: {} >= {}", slot, store.count
    );
    unsafe { *store.mass.add(slot as usize) >= 0.0 }
}

/// Marca un slot como muerto: swap-remove O(1) en `live_indices`,
/// `mass = -1.0`, generación++, push LIFO en `free_list`, `live_count--`.
///
/// Precondición: `is_live(store, slot) == true`.
#[inline(always)]
pub fn mark_dead(store: &mut NativeStore, slot: i32) -> Result<(), LifecycleError> {
    debug_assert!(!store.base_address.is_null(), "store no inicializado");
    debug_assert!(
        slot >= 0 && slot < store.count as i32,
        "slot fuera de rango: {} >= {}", slot, store.count
    );
    if !is_live(store, slot) {
        return Err(LifecycleError::SlotNotLive);
    }

    unsafe {
        let last_live_pos = store.live_count - 1;
        let last_slot = *store.live_indices.add(last_live_pos);

        // Swap-remove: el último slot vivo ocupa la posición del slot muerto.
        let pos = *store.live_pos.add(slot as usize);
        *store.live_indices.add(pos as usize) = last_slot;
        *store.live_pos.add(last_slot as usize) = pos;

        // Centinela de slot muerto.
        *store.mass.add(slot as usize) = -1.0;
        *store.generation.add(slot as usize) += 1;

        // Push LIFO en la lista libre.
        *store.free_list.add(slot as usize) = store.free_head;
        store.free_head = slot;

        store.live_count -= 1;
    }

    Ok(())
}

/// Alloca un slot vivo: pop de FreeList si hay, o bump (`live_count`) si
/// `live_count < count`. Devuelve `Err(NoCapacity)` sin capacidad.
/// Inicializa todos los canales del slot, incrementa `generation` y hace
/// append a `live_indices`.
#[inline(always)]
pub fn allocate_entity(
    store: &mut NativeStore,
    x: f32,
    y: f32,
    vx: f32,
    vy: f32,
    mass: f32,
    radius: f32,
) -> Result<i32, LifecycleError> {
    debug_assert!(!store.base_address.is_null(), "store no inicializado");

    let slot: i32;
    if store.free_head != -1 {
        // Recicla slot libre (LIFO).
        slot = store.free_head;
        unsafe {
            store.free_head = *store.free_list.add(slot as usize);
        }
    } else if store.live_count < store.count {
        // Bump sobre el siguiente slot libre del compacto.
        slot = store.live_count as i32;
    } else {
        // Sin capacidad: no modifica nada.
        return Err(LifecycleError::NoCapacity);
    }

    unsafe {
        *store.x.add(slot as usize) = x;
        *store.y.add(slot as usize) = y;
        *store.vx.add(slot as usize) = vx;
        *store.vy.add(slot as usize) = vy;
        *store.mass.add(slot as usize) = mass;
        *store.radius.add(slot as usize) = radius;
        *store.fx.add(slot as usize) = 0.0;
        *store.fy.add(slot as usize) = 0.0;
        *store.generation.add(slot as usize) += 1;

        let pos = store.live_count;
        *store.live_indices.add(pos) = slot;
        *store.live_pos.add(slot as usize) = pos as i32;
        store.live_count += 1;
    }

    Ok(slot)
}

// ---------------------------------------------------------------------------
// Tests (zero-allocation solo en hot path; los tests pueden usar alloc).
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

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

        // AC2: push LIFO en free_list — free_head = 1, free_list[1] = -1 (vacía antes).
        assert_eq!(store.free_head, 1);
        unsafe { assert_eq!(*store.free_list.add(1), -1); }
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
            assert_eq!(*store.free_list.add(2), -1);
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

        // Marcar slots 1 y 2 como muertos → free_list LIFO: free_head=2, free_list[2]=1, free_list[1]=-1.
        assert!(mark_dead(&mut store, 1).is_ok());
        assert!(mark_dead(&mut store, 2).is_ok());
        assert_eq!(store.free_head, 2);
        unsafe {
            assert_eq!(*store.free_list.add(2), 1);
            assert_eq!(*store.free_list.add(1), -1);
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
        assert_eq!(store.free_head, -1);
    }

    #[test]
    fn ac3_allocate_entity_bumps_when_free_list_empty() {
        let mut store = make_store(2);
        assert_eq!(store.live_count, 2);
        assert_eq!(store.free_head, -1);

        // Store lleno (live_count == count) → no hay bump posible.
        let res = allocate_entity(&mut store, 1.0, 2.0, 3.0, 4.0, 5.0, 0.5);
        assert_eq!(res, Err(LifecycleError::NoCapacity));
        assert_eq!(store.live_count, 2); // sin cambios
    }

    #[test]
    fn ac3_allocate_entity_after_mark_dead_recycles_then_bumps() {
        let mut store = make_store(3);
        assert_eq!(store.live_count, 3);

        // Matar slot 0 → free_head = 0, live_count = 2.
        assert!(mark_dead(&mut store, 0).is_ok());
        assert_eq!(store.live_count, 2);
        assert_eq!(store.free_head, 0);

        // Allocate recicla slot 0 (LIFO), live_count vuelve a 3.
        let res = allocate_entity(&mut store, 9.0, 9.0, 9.0, 9.0, 9.0, 0.5);
        assert!(res.is_ok());
        let slot = res.unwrap();
        assert_eq!(slot, 0);
        assert_eq!(store.live_count, 3);
        assert_eq!(store.free_head, -1);

        // Ahora lleno; siguiente allocate falla.
        let res = allocate_entity(&mut store, 1.0, 2.0, 3.0, 4.0, 5.0, 0.5);
        assert_eq!(res, Err(LifecycleError::NoCapacity));
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
        // Esta prueba es una salvaguarda: llama a las funciones en un bucle y confirma
        // que no hay panic ni comportamiento inesperado.
        let mut store = make_store(100);

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
        assert_eq!(store.live_count, 100);
        assert_eq!(store.free_head, -1);
    }
}
