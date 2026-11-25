//! store.rs — NativeStore: almacenamiento SoA sobre memoria nativa alineada.
//! (Issue 2: Memoria nativa y NativeStore — base mínima para SimulationStore.)

use crate::native_mem::{aligned_alloc, aligned_free};
use std::ptr::NonNull;

/// Almacenamiento SoA (Structure of Arrays) sobre memoria nativa.
///
/// Cada canal (X, Y, Vx, Vy, Fx, Fy, Mass, Radius) es un bloque independiente
/// alineado a 64 B; los canales de índice (Generation, FreeList, LiveIndices,
/// LivePos, BodyNext) son bloques de i32. `count` es la capacidad fija;
/// `live_count` el número de cuerpos vivos. `free_head` es la cabeza de la
/// pila LIFO de slots libres (-1 = vacía). `base_address` apunta al inicio del
/// bloque consolidado. `BodyNext` es la cadena cuerpo-a-cuerpo de las hojas
/// LeafMulti del quadtree (vive en el store, no en el arena).
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
    pub body_next: *mut i32,
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
        let total_bytes = size * 8 + gen_size * 5;
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
        let body_next_ptr = unsafe {
            base_ptr.add(size * 8 + gen_size * 4) as *mut i32
        };

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
            body_next: body_next_ptr,
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
                *self.body_next.add(i) = -1;
            }
        }
        self.free_head = 0;
    }

    /// Vista inmutable (compartida, solo lectura) sobre el canal X de `count`
    /// elementos. Varios llamadores pueden tener `&[f32]` a la vez porque el
    /// receptor es `&self`; no hay aliasing mutable.
    pub fn x_slice(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.x, self.count) }
    }

    /// Vista exclusiva (lectura/escritura) sobre el canal X de `count`
    /// elementos. Requiere `&mut self`, que garantiza exclusividad y elimina
    /// el aliasing mutable que tenían las vistas `&mut` obtenidas desde `&self`.
    pub fn x_slice_mut(&mut self) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.x, self.count) }
    }

    /// Vista inmutable (compartida, solo lectura) sobre el canal Y.
    pub fn y_slice(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.y, self.count) }
    }

    /// Vista exclusiva (lectura/escritura) sobre el canal Y.
    pub fn y_slice_mut(&mut self) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.y, self.count) }
    }

    /// Vista inmutable (compartida, solo lectura) sobre el canal Vx.
    pub fn vx_slice(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.vx, self.count) }
    }

    /// Vista exclusiva (lectura/escritura) sobre el canal Vx.
    pub fn vx_slice_mut(&mut self) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.vx, self.count) }
    }

    /// Vista inmutable (compartida, solo lectura) sobre el canal Vy.
    pub fn vy_slice(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.vy, self.count) }
    }

    /// Vista exclusiva (lectura/escritura) sobre el canal Vy.
    pub fn vy_slice_mut(&mut self) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.vy, self.count) }
    }

    /// Vista inmutable (compartida, solo lectura) sobre el canal de
    /// generaciones.
    pub fn generation_slice(&self) -> &[i32] {
        unsafe { std::slice::from_raw_parts(self.generation, self.count) }
    }

    /// Vista exclusiva (lectura/escritura) sobre el canal de generaciones.
    pub fn generation_slice_mut(&mut self) -> &mut [i32] {
        unsafe { std::slice::from_raw_parts_mut(self.generation, self.count) }
    }

    /// Vista inmutable (compartida, solo lectura) sobre el canal Fx.
    pub fn fx_slice(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.fx, self.count) }
    }

    /// Vista exclusiva (lectura/escritura) sobre el canal Fx.
    pub fn fx_slice_mut(&mut self) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.fx, self.count) }
    }

    /// Vista inmutable (compartida, solo lectura) sobre el canal Fy.
    pub fn fy_slice(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.fy, self.count) }
    }

    /// Vista exclusiva (lectura/escritura) sobre el canal Fy.
    pub fn fy_slice_mut(&mut self) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.fy, self.count) }
    }

    /// Vista inmutable (compartida, solo lectura) sobre el canal Mass.
    pub fn mass_slice(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.mass, self.count) }
    }

    /// Vista exclusiva (lectura/escritura) sobre el canal Mass.
    pub fn mass_slice_mut(&mut self) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.mass, self.count) }
    }

    /// Vista inmutable (compartida, solo lectura) sobre el canal Radius.
    pub fn radius_slice(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.radius, self.count) }
    }

    /// Vista exclusiva (lectura/escritura) sobre el canal Radius.
    pub fn radius_slice_mut(&mut self) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.radius, self.count) }
    }

    /// Vista inmutable (compartida, solo lectura) sobre el canal FreeList.
    pub fn free_list_slice(&self) -> &[i32] {
        unsafe { std::slice::from_raw_parts(self.free_list, self.count) }
    }

    /// Vista exclusiva (lectura/escritura) sobre el canal FreeList.
    pub fn free_list_slice_mut(&mut self) -> &mut [i32] {
        unsafe { std::slice::from_raw_parts_mut(self.free_list, self.count) }
    }

    /// Vista inmutable (compartida, solo lectura) sobre el canal LiveIndices.
    pub fn live_indices_slice(&self) -> &[i32] {
        unsafe { std::slice::from_raw_parts(self.live_indices, self.count) }
    }

    /// Vista exclusiva (lectura/escritura) sobre el canal LiveIndices.
    pub fn live_indices_slice_mut(&mut self) -> &mut [i32] {
        unsafe { std::slice::from_raw_parts_mut(self.live_indices, self.count) }
    }

    /// Vista inmutable (compartida, solo lectura) sobre el canal LivePos.
    pub fn live_pos_slice(&self) -> &[i32] {
        unsafe { std::slice::from_raw_parts(self.live_pos, self.count) }
    }

    /// Vista exclusiva (lectura/escritura) sobre el canal LivePos.
    pub fn live_pos_slice_mut(&mut self) -> &mut [i32] {
        unsafe { std::slice::from_raw_parts_mut(self.live_pos, self.count) }
    }

    /// Todas las vistas mutables (lectura/escritura) sobre los canales del
    /// store, obtenidas con un único `&mut self`. Útil para kernels que escriben
    /// varios canales en un mismo bucle (siembra, reflejo de bordes). La división
    /// interna usa punteros crudos sobre regiones de memoria disjuntas, por lo
    /// que es segura y no tiene coste en tiempo de ejecución.
    pub fn channels_mut(&mut self) -> ChannelsMut<'_> {
        let count = self.count;
        ChannelsMut {
            x: unsafe { std::slice::from_raw_parts_mut(self.x, count) },
            y: unsafe { std::slice::from_raw_parts_mut(self.y, count) },
            vx: unsafe { std::slice::from_raw_parts_mut(self.vx, count) },
            vy: unsafe { std::slice::from_raw_parts_mut(self.vy, count) },
            fx: unsafe { std::slice::from_raw_parts_mut(self.fx, count) },
            fy: unsafe { std::slice::from_raw_parts_mut(self.fy, count) },
            mass: unsafe { std::slice::from_raw_parts_mut(self.mass, count) },
            radius: unsafe { std::slice::from_raw_parts_mut(self.radius, count) },
            generation: unsafe {
                std::slice::from_raw_parts_mut(self.generation, count)
            },
            free_list: unsafe {
                std::slice::from_raw_parts_mut(self.free_list, count)
            },
            live_indices: unsafe {
                std::slice::from_raw_parts_mut(self.live_indices, count)
            },
            live_pos: unsafe {
                std::slice::from_raw_parts_mut(self.live_pos, count)
            },
            body_next: unsafe {
                std::slice::from_raw_parts_mut(self.body_next, count)
            },
        }
    }
}

/// Vistas mutables sobre todos los canales de `NativeStore`, tomadas con un
/// único `&mut self`. Cada campo es un `&mut [T]` sobre su canal; al apuntar a
/// regiones de memoria disjuntas no existe aliasing entre ellos.
pub struct ChannelsMut<'a> {
    pub x: &'a mut [f32],
    pub y: &'a mut [f32],
    pub vx: &'a mut [f32],
    pub vy: &'a mut [f32],
    pub fx: &'a mut [f32],
    pub fy: &'a mut [f32],
    pub mass: &'a mut [f32],
    pub radius: &'a mut [f32],
    pub generation: &'a mut [i32],
    pub free_list: &'a mut [i32],
    pub live_indices: &'a mut [i32],
    pub live_pos: &'a mut [i32],
    pub body_next: &'a mut [i32],
}

impl Drop for NativeStore {
    fn drop(&mut self) {
        if self.base_address.is_null() {
            return;
        }
        let total_bytes = self.channel_size * 8 + self.gen_channel_size * 5;
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
        self.body_next = std::ptr::null_mut();
        self.count = 0;
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

        // Primera vista: escritura vía slice visible por el puntero crudo.
        let (ptr, len) = {
            let s1 = store.x_slice_mut();
            assert_eq!(s1.len(), n, "len debe ser igual a count");
            s1[3] = 42.0;
            assert_eq!(unsafe { *s1.as_ptr().add(3) }, 42.0, "slice y canal deben compartir memoria");
            (s1.as_ptr(), s1.len())
        }; // s1 se descarta aquí; el borrow de &mut store termina.

        // Segunda llamada: mismo puntero y longitud (reconstrucción al vuelo es estable).
        let s2 = store.x_slice();
        assert_eq!(s2.as_ptr(), ptr, "el puntero debe ser estable entre llamadas");
        assert_eq!(s2.len(), len, "len debe ser estable entre llamadas");
        assert_eq!(s2[3], 42.0, "la escritura vía slice debe persistir en el canal");
        let _ = s2;

        // Corrección inversa: escritura por puntero crudo visible vía slice.
        unsafe { *store.x.add(5) = -1.5; }
        let s3 = store.x_slice();
        assert_eq!(s3[5], -1.5, "escritura cruda debe ser visible vía slice");
    }

    #[test]
    fn todas_las_vistas_apuntan_a_su_canal() {
        let n = 4;
        let mut store = NativeStore::new(n).expect("store creation failed");
        store.init_lifecycle();

        // `*_slice(&self)` devuelve vistas compartidas &[T] (coexisten entre sí);
        // `*_slice_mut(&mut self)` es la vista exclusiva. Todas apuntan al canal
        // correspondiente sin copiar.
        macro_rules! check_slice {
            ($slice:expr, $ptr:expr, $len:expr) => {{
                let s = $slice;
                assert_eq!(s.as_ptr(), $ptr as *const _);
                assert_eq!(s.len(), $len);
            }};
        }

        check_slice!(store.x_slice(), store.x, n);
        check_slice!(store.y_slice(), store.y, n);
        check_slice!(store.mass_slice(), store.mass, n);
        check_slice!(store.generation_slice(), store.generation, n);
        check_slice!(store.free_list_slice(), store.free_list, n);
        check_slice!(store.live_indices_slice(), store.live_indices, n);
        check_slice!(store.live_pos_slice(), store.live_pos, n);

        unsafe {
            assert_eq!(*store.live_indices.add(2), 2, "live_indices identidad tras init_lifecycle");
            assert_eq!(*store.live_pos.add(2), 2, "live_pos identidad tras init_lifecycle");
        }
    }

    #[test]
    #[ignore = "medición de rendimiento: reconstrucción al vuelo por llamada"]
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