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


