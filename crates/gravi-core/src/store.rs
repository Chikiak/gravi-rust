//! store.rs — NativeStore: almacenamiento SoA sobre memoria nativa alineada.
//! (Issue 2: Memoria nativa y NativeStore — base mínima para SimulationStore.)

use crate::native_mem::{aligned_alloc, aligned_free};
use std::ptr::NonNull;

/// Almacenamiento SoA (Structure of Arrays) sobre memoria nativa.
///
/// Cada canal (X, Y, Vx, Vy, Fx, Fy, Mass, Radius) es un bloque independiente
/// alineado a 64 B; los canales de índice (Generation, FreeList, LiveIndices,
/// LivePos) son bloques de i32. `count` es la capacidad fija; `live_count` el
/// número de cuerpos vivos. `free_head` es la cabeza de la pila LIFO de slots
/// libres (-1 = vacía). `base_address` apunta al inicio del bloque consolidado.
#[repr(C)]
pub struct NativeStore {
    pub base_address: *mut u8,
    pub x: *mut f32,
    pub y: *mut f32,
    pub vx: *mut f32,
    pub vy: *mut f32,
    pub fx: *mut f32,
    pub fy: *mut f32,
    pub mass: *mut f32,
    pub radius: *mut f32,
    pub generation: *mut i32,
    pub free_list: *mut i32,
    pub live_indices: *mut i32,
    pub live_pos: *mut i32,
    pub count: usize,
    pub live_count: usize,
    pub free_head: i32,
    pub byte_alignment: usize,
    channel_size: usize,
    gen_channel_size: usize,
}

impl NativeStore {
    /// Crea un `NativeStore` con capacidad `count` cuerpos.
    /// Todos los canales se asignan y se cerran a cero.
    pub fn new(count: usize) -> Result<Self, String> {
        if count == 0 {
            return Err("body_count must be > 0".to_string());
        }
        let size = count * std::mem::size_of::<f32>();
        let gen_size = count * std::mem::size_of::<i32>();
        let total_bytes = size * 8 + gen_size * 4;
        let base = unsafe { aligned_alloc(total_bytes, 64) }
            .ok_or_else(|| "allocation failed: out of memory".to_string())?;
        let base_ptr = base.as_ptr() as *mut u8;

        unsafe {
            std::ptr::write_bytes(base_ptr, 0, total_bytes);
        }

        Ok(Self {
            base_address: base_ptr,
            x: base_ptr as *mut f32,
            y: unsafe { base_ptr.add(size) as *mut f32 },
            vx: unsafe { base_ptr.add(size * 2) as *mut f32 },
            vy: unsafe { base_ptr.add(size * 3) as *mut f32 },
            fx: unsafe { base_ptr.add(size * 4) as *mut f32 },
            fy: unsafe { base_ptr.add(size * 5) as *mut f32 },
            mass: unsafe { base_ptr.add(size * 6) as *mut f32 },
            radius: unsafe { base_ptr.add(size * 7) as *mut f32 },
            generation: unsafe { base_ptr.add(size * 8) as *mut i32 },
            free_list: unsafe { base_ptr.add(size * 8 + gen_size) as *mut i32 },
            live_indices: unsafe {
                base_ptr.add(size * 8 + gen_size * 2) as *mut i32
            },
            live_pos: unsafe {
                base_ptr.add(size * 8 + gen_size * 3) as *mut i32
            },
            count,
            live_count: count,
            free_head: -1,
            byte_alignment: 64,
            channel_size: size,
            gen_channel_size: gen_size,
        })
    }

    /// Inicializa los campos de ciclo de vida (llamar tras `new`):
    /// `mass = 1.0` (vivo), `generation = 0`, `free_list` encadenado,
    /// `live_indices` identidad, `live_pos` identidad.
    pub fn init_lifecycle(&mut self) {
        let count = self.count;
        unsafe {
            for i in 0..count {
                *self.mass.add(i) = 1.0;
                *self.radius.add(i) = 0.5;
                *self.generation.add(i) = 0;
                *self.free_list.add(i) = if i + 1 < count { (i + 1) as i32 } else { -1 };
                *self.live_indices.add(i) = i as i32;
                *self.live_pos.add(i) = i as i32;
            }
        }
    }

    /// Devuelve una vista mutable sobre el canal X de `count` elementos.
    pub fn x_slice(&self) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.x, self.count) }
    }

    /// Devuelve una vista mutable sobre el canal Y de `count` elementos.
    pub fn y_slice(&self) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.y, self.count) }
    }

    /// Devuelve una vista mutable sobre el canal Vx de `count` elementos.
    pub fn vx_slice(&self) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.vx, self.count) }
    }

    /// Devuelve una vista mutable sobre el canal Vy de `count` elementos.
    pub fn vy_slice(&self) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.vy, self.count) }
    }

    /// Devuelve una vista mutable sobre el canal de generaciones de `count` elementos.
    pub fn generation_slice(&self) -> &mut [i32] {
        unsafe { std::slice::from_raw_parts_mut(self.generation, self.count) }
    }

    /// Devuelve una vista mutable sobre el canal Fx de `count` elementos.
    pub fn fx_slice(&self) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.fx, self.count) }
    }

    /// Devuelve una vista mutable sobre el canal Fy de `count` elementos.
    pub fn fy_slice(&self) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.fy, self.count) }
    }

    /// Devuelve una vista mutable sobre el canal Mass de `count` elementos.
    pub fn mass_slice(&self) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.mass, self.count) }
    }

    /// Devuelve una vista mutable sobre el canal Radius de `count` elementos.
    pub fn radius_slice(&self) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.radius, self.count) }
    }

    /// Devuelve una vista mutable sobre el canal FreeList de `count` elementos.
    pub fn free_list_slice(&self) -> &mut [i32] {
        unsafe { std::slice::from_raw_parts_mut(self.free_list, self.count) }
    }

    /// Devuelve una vista mutable sobre el canal LiveIndices de `count` elementos.
    pub fn live_indices_slice(&self) -> &mut [i32] {
        unsafe { std::slice::from_raw_parts_mut(self.live_indices, self.count) }
    }

    /// Devuelve una vista mutable sobre el canal LivePos de `count` elementos.
    pub fn live_pos_slice(&self) -> &mut [i32] {
        unsafe { std::slice::from_raw_parts_mut(self.live_pos, self.count) }
    }
}

impl Drop for NativeStore {
    fn drop(&mut self) {
        if self.base_address.is_null() {
            return;
        }
        let total_bytes = self.channel_size * 8 + self.gen_channel_size * 4;
        unsafe {
            aligned_free(
                NonNull::new(self.base_address as *mut u8).unwrap(),
                total_bytes,
                self.byte_alignment,
            );
        }
        self.base_address = std::ptr::null_mut();
        self.x = std::ptr::null_mut();
        self.y = std::ptr::null_mut();
        self.vx = std::ptr::null_mut();
        self.vy = std::ptr::null_mut();
        self.fx = std::ptr::null_mut();
        self.fy = std::ptr::null_mut();
        self.mass = std::ptr::null_mut();
        self.radius = std::ptr::null_mut();
        self.generation = std::ptr::null_mut();
        self.free_list = std::ptr::null_mut();
        self.live_indices = std::ptr::null_mut();
        self.live_pos = std::ptr::null_mut();
        self.count = 0;
        self.live_count = 0;
        self.free_head = -1;
        self.byte_alignment = 0;
    }
}

// CHANNEL_ALIGN, alloc_channel y alloc_generation_channel ya no se usan
// (NativeStore::new usa una sola asignación consolidada).
