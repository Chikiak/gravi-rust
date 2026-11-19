//! store.rs — NativeStore: almacenamiento SoA sobre memoria nativa alineada.
//! (Issue 2: Memoria nativa y NativeStore — base mínima para SimulationStore.)

use crate::native_mem::{aligned_alloc, aligned_free};
use std::ptr::NonNull;

/// Vista cruda vacía y colgante (alineada) para invalidar las cachés en `Drop`.
/// `slice_from_raw_parts_mut` es seguro (solo combina puntero + longitud) pero
/// exige puntero no-nulo y alineado incluso con len 0.
fn dangling_f32_slice() -> *mut [f32] {
    let ptr = NonNull::<f32>::dangling().as_ptr();
    std::ptr::slice_from_raw_parts_mut(ptr, 0)
}

/// Igual que [`dangling_f32_slice`] para los canales de índice (`i32`).
fn dangling_i32_slice() -> *mut [i32] {
    let ptr = NonNull::<i32>::dangling().as_ptr();
    std::ptr::slice_from_raw_parts_mut(ptr, 0)
}

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
    /// Longitud con la que se materializaron las vistas cacheadas en `new`.
    /// Coincide con `count` salvo mutación externa del campo público `count`;
    /// en ese caso `*_slice()` toma la ruta de reconstrucción al vuelo.
    slice_count: usize,
    /// Vistas cacheadas: puntero crudo + longitud materializados UNA sola vez
    /// en `new`. `*_slice()` devuelve `&mut *self.<canal>_cache` sin reconstruir
    /// `from_raw_parts_mut` por llamada. Los canales no se reasignan nunca tras
    /// `new` (solo `Drop` los invalida), así que la caché es válida durante toda
    /// la vida del store; `init_lifecycle` no la refresca porque no mueve buffers.
    x_cache: *mut [f32],
    y_cache: *mut [f32],
    vx_cache: *mut [f32],
    vy_cache: *mut [f32],
    fx_cache: *mut [f32],
    fy_cache: *mut [f32],
    mass_cache: *mut [f32],
    radius_cache: *mut [f32],
    generation_cache: *mut [i32],
    free_list_cache: *mut [i32],
    live_indices_cache: *mut [i32],
    live_pos_cache: *mut [i32],
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

        let x_ptr = base_ptr as *mut f32;
        let y_ptr = unsafe { base_ptr.add(size) as *mut f32 };
        let vx_ptr = unsafe { base_ptr.add(size * 2) as *mut f32 };
        let vy_ptr = unsafe { base_ptr.add(size * 3) as *mut f32 };
        let fx_ptr = unsafe { base_ptr.add(size * 4) as *mut f32 };
        let fy_ptr = unsafe { base_ptr.add(size * 5) as *mut f32 };
        let mass_ptr = unsafe { base_ptr.add(size * 6) as *mut f32 };
        let radius_ptr = unsafe { base_ptr.add(size * 7) as *mut f32 };
        let generation_ptr = unsafe { base_ptr.add(size * 8) as *mut i32 };
        let free_list_ptr = unsafe { base_ptr.add(size * 8 + gen_size) as *mut i32 };
        let live_indices_ptr = unsafe {
            base_ptr.add(size * 8 + gen_size * 2) as *mut i32
        };
        let live_pos_ptr = unsafe { base_ptr.add(size * 8 + gen_size * 3) as *mut i32 };

        Ok(Self {
            base_address: base_ptr,
            x: x_ptr,
            y: y_ptr,
            vx: vx_ptr,
            vy: vy_ptr,
            fx: fx_ptr,
            fy: fy_ptr,
            mass: mass_ptr,
            radius: radius_ptr,
            generation: generation_ptr,
            free_list: free_list_ptr,
            live_indices: live_indices_ptr,
            live_pos: live_pos_ptr,
            count,
            live_count: count,
            free_head: -1,
            byte_alignment: 64,
            channel_size: size,
            gen_channel_size: gen_size,
            slice_count: count,
            // Caché materializada: fat pointer crudo por canal (sin crear
            // referencia todavía). Es la vista que devuelven los `*_slice()`.
            x_cache: std::ptr::slice_from_raw_parts_mut(x_ptr, count),
            y_cache: std::ptr::slice_from_raw_parts_mut(y_ptr, count),
            vx_cache: std::ptr::slice_from_raw_parts_mut(vx_ptr, count),
            vy_cache: std::ptr::slice_from_raw_parts_mut(vy_ptr, count),
            fx_cache: std::ptr::slice_from_raw_parts_mut(fx_ptr, count),
            fy_cache: std::ptr::slice_from_raw_parts_mut(fy_ptr, count),
            mass_cache: std::ptr::slice_from_raw_parts_mut(mass_ptr, count),
            radius_cache: std::ptr::slice_from_raw_parts_mut(radius_ptr, count),
            generation_cache: std::ptr::slice_from_raw_parts_mut(generation_ptr, count),
            free_list_cache: std::ptr::slice_from_raw_parts_mut(free_list_ptr, count),
            live_indices_cache: std::ptr::slice_from_raw_parts_mut(
                live_indices_ptr, count,
            ),
            live_pos_cache: std::ptr::slice_from_raw_parts_mut(live_pos_ptr, count),
        })
    }

    /// Inicializa los campos de ciclo de vida (llamar tras `new`):
    /// `mass = 1.0` (vivo), `generation = 0`, `free_list` encadenado,
    /// `live_indices` identidad, `live_pos` identidad.
    ///
    /// No refresca las cachés: no reasigna ningún buffer ni cambia `count`,
    /// así que las vistas materializadas en `new` siguen siendo válidas.
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
        self.free_head = 0;
    }

    /// Devuelve una vista mutable sobre el canal X de `count` elementos.
    ///
    /// Ruta cacheada (normal): devuelve la vista materializada en `new`
    /// (`&mut *self.x_cache`), puntero estable entre llamadas y sin
    /// `from_raw_parts_mut` por llamada. Solo si el campo público `count` se
    /// mutó externamente (no ocurre en el ciclo normal) se reconstruye la vista
    /// al vuelo; con `&self` no se puede refrescar la caché en ese caso.
    pub fn x_slice(&self) -> &mut [f32] {
        if self.count == self.slice_count {
            unsafe { &mut *self.x_cache }
        } else {
            unsafe { std::slice::from_raw_parts_mut(self.x, self.count) }
        }
    }

    /// Devuelve una vista mutable sobre el canal Y de `count` elementos
    /// (misma caché que [`NativeStore::x_slice`]).
    pub fn y_slice(&self) -> &mut [f32] {
        if self.count == self.slice_count {
            unsafe { &mut *self.y_cache }
        } else {
            unsafe { std::slice::from_raw_parts_mut(self.y, self.count) }
        }
    }

    /// Devuelve una vista mutable sobre el canal Vx de `count` elementos
    /// (misma caché que [`NativeStore::x_slice`]).
    pub fn vx_slice(&self) -> &mut [f32] {
        if self.count == self.slice_count {
            unsafe { &mut *self.vx_cache }
        } else {
            unsafe { std::slice::from_raw_parts_mut(self.vx, self.count) }
        }
    }

    /// Devuelve una vista mutable sobre el canal Vy de `count` elementos
    /// (misma caché que [`NativeStore::x_slice`]).
    pub fn vy_slice(&self) -> &mut [f32] {
        if self.count == self.slice_count {
            unsafe { &mut *self.vy_cache }
        } else {
            unsafe { std::slice::from_raw_parts_mut(self.vy, self.count) }
        }
    }

    /// Devuelve una vista mutable sobre el canal de generaciones de `count`
    /// elementos (misma caché que [`NativeStore::x_slice`]).
    pub fn generation_slice(&self) -> &mut [i32] {
        if self.count == self.slice_count {
            unsafe { &mut *self.generation_cache }
        } else {
            unsafe { std::slice::from_raw_parts_mut(self.generation, self.count) }
        }
    }

    /// Devuelve una vista mutable sobre el canal Fx de `count` elementos
    /// (misma caché que [`NativeStore::x_slice`]).
    pub fn fx_slice(&self) -> &mut [f32] {
        if self.count == self.slice_count {
            unsafe { &mut *self.fx_cache }
        } else {
            unsafe { std::slice::from_raw_parts_mut(self.fx, self.count) }
        }
    }

    /// Devuelve una vista mutable sobre el canal Fy de `count` elementos
    /// (misma caché que [`NativeStore::x_slice`]).
    pub fn fy_slice(&self) -> &mut [f32] {
        if self.count == self.slice_count {
            unsafe { &mut *self.fy_cache }
        } else {
            unsafe { std::slice::from_raw_parts_mut(self.fy, self.count) }
        }
    }

    /// Devuelve una vista mutable sobre el canal Mass de `count` elementos
    /// (misma caché que [`NativeStore::x_slice`]).
    pub fn mass_slice(&self) -> &mut [f32] {
        if self.count == self.slice_count {
            unsafe { &mut *self.mass_cache }
        } else {
            unsafe { std::slice::from_raw_parts_mut(self.mass, self.count) }
        }
    }

    /// Devuelve una vista mutable sobre el canal Radius de `count` elementos
    /// (misma caché que [`NativeStore::x_slice`]).
    pub fn radius_slice(&self) -> &mut [f32] {
        if self.count == self.slice_count {
            unsafe { &mut *self.radius_cache }
        } else {
            unsafe { std::slice::from_raw_parts_mut(self.radius, self.count) }
        }
    }

    /// Devuelve una vista mutable sobre el canal FreeList de `count` elementos
    /// (misma caché que [`NativeStore::x_slice`]).
    pub fn free_list_slice(&self) -> &mut [i32] {
        if self.count == self.slice_count {
            unsafe { &mut *self.free_list_cache }
        } else {
            unsafe { std::slice::from_raw_parts_mut(self.free_list, self.count) }
        }
    }

    /// Devuelve una vista mutable sobre el canal LiveIndices de `count`
    /// elementos (misma caché que [`NativeStore::x_slice`]).
    pub fn live_indices_slice(&self) -> &mut [i32] {
        if self.count == self.slice_count {
            unsafe { &mut *self.live_indices_cache }
        } else {
            unsafe { std::slice::from_raw_parts_mut(self.live_indices, self.count) }
        }
    }

    /// Devuelve una vista mutable sobre el canal LivePos de `count` elementos
    /// (misma caché que [`NativeStore::x_slice`]).
    pub fn live_pos_slice(&self) -> &mut [i32] {
        if self.count == self.slice_count {
            unsafe { &mut *self.live_pos_cache }
        } else {
            unsafe { std::slice::from_raw_parts_mut(self.live_pos, self.count) }
        }
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
        self.slice_count = 0;
        // Invalidar las cachés: los punteros base ya no son válidos.
        self.x_cache = dangling_f32_slice();
        self.y_cache = dangling_f32_slice();
        self.vx_cache = dangling_f32_slice();
        self.vy_cache = dangling_f32_slice();
        self.fx_cache = dangling_f32_slice();
        self.fy_cache = dangling_f32_slice();
        self.mass_cache = dangling_f32_slice();
        self.radius_cache = dangling_f32_slice();
        self.generation_cache = dangling_i32_slice();
        self.free_list_cache = dangling_i32_slice();
        self.live_indices_cache = dangling_i32_slice();
        self.live_pos_cache = dangling_i32_slice();
        self.live_count = 0;
        self.free_head = -1;
        self.byte_alignment = 0;
    }
}

// CHANNEL_ALIGN, alloc_channel y alloc_generation_channel ya no se usan
// (NativeStore::new usa una sola asignación consolidada).

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn store_init_lifecycle_establece_free_head_cero() {
        let n = 5;
        let mut store = NativeStore::new(n).expect("store creation failed");
        store.init_lifecycle();

        assert_eq!(store.free_head, 0);
        for i in 0..(n - 1) {
            unsafe {
                assert_eq!(*store.free_list.add(i), (i + 1) as i32, "free_list[{}] debe ser {}", i, i + 1);
            }
        }
        unsafe {
            assert_eq!(*store.free_list.add(n - 1), -1);
        }
    }

    #[test]
    fn x_slice_cache_puntero_estable_entre_llamadas() {
        let n = 8;
        let mut store = NativeStore::new(n).expect("store creation failed");
        store.init_lifecycle();

        // Primera vista (materializada desde la caché de `new`).
        let (ptr, len) = {
            let s1 = store.x_slice();
            assert_eq!(s1.len(), n, "len debe ser igual a count");
            // Corrección: escritura vía slice visible por el puntero crudo.
            s1[3] = 42.0;
            assert_eq!(unsafe { *store.x.add(3) }, 42.0, "slice y canal deben compartir memoria");
            (s1.as_ptr(), s1.len())
        };

        // Segunda llamada: misma vista cacheada (puntero y longitud estables).
        let s2 = store.x_slice();
        assert_eq!(s2.as_ptr(), ptr, "el puntero debe ser estable entre llamadas (caché)");
        assert_eq!(s2.len(), len, "len debe ser estable entre llamadas (caché)");
        assert_eq!(s2[3], 42.0, "la escritura vía slice debe persistir en el canal");

        // Corrección inversa: escritura por puntero crudo visible vía slice.
        unsafe { *store.x.add(5) = -1.5; }
        assert_eq!(store.x_slice()[5], -1.5, "escritura cruda debe ser visible vía slice");
    }

    #[test]
    fn todas_las_vistas_comparten_la_cache_del_canal() {
        let n = 4;
        let mut store = NativeStore::new(n).expect("store creation failed");
        store.init_lifecycle();

        // Cada vista cacheada apunta a SU canal y tiene longitud `count`.
        let xs = store.x_slice();
        assert_eq!(xs.as_ptr(), store.x as *const f32);
        assert_eq!(xs.len(), n);
        let ys = store.y_slice();
        assert_eq!(ys.as_ptr(), store.y as *const f32);
        assert_eq!(ys.len(), n);
        let masses = store.mass_slice();
        assert_eq!(masses.as_ptr(), store.mass as *const f32);
        let gens = store.generation_slice();
        assert_eq!(gens.as_ptr(), store.generation as *const i32);
        let fl = store.free_list_slice();
        assert_eq!(fl.as_ptr(), store.free_list as *const i32);
        let li = store.live_indices_slice();
        assert_eq!(li.as_ptr(), store.live_indices as *const i32);
        let lp = store.live_pos_slice();
        assert_eq!(lp.as_ptr(), store.live_pos as *const i32);
        assert_eq!(li[2], 2, "live_indices identidad tras init_lifecycle");
        assert_eq!(lp[2], 2, "live_pos identidad tras init_lifecycle");
    }

    #[test]
    #[ignore = "medición de rendimiento: comparar TIMING antes/después del caché"]
    fn timing_x_slice_100k() {
        use std::hint::black_box;
        use std::time::Instant;
        let n = 8;
        let store = NativeStore::new(n).expect("store creation failed");
        let iters = 100_000;
        let start = Instant::now();
        for _ in 0..iters {
            let s = black_box(store.x_slice());
            black_box(s.len());
        }
        let elapsed = start.elapsed();
        println!("TIMING x_slice {} iters en {:?} ({} ns/call)", iters, elapsed, elapsed.as_nanos() / iters as u128);
    }
}