//! spatial_grid.rs — SpatialGrid + enumerador de pares (Issue 9, Onda 3).
//!
//! Rejilla espacial de celdas fijas preasignada sobre memoria nativa, con
//! listas enlazadas implícitas (`bucket_head` + `next`) por bucket.
//!
//! Estructuras:
//! - `SpatialGrid`: POD `repr(C)` sobre buffers unmanaged (`bucket_head`,
//!   `next`) de índices de entidades.
//! - `SpatialPairEnumerator`: struct por valor, `MoveNext`/`Current` con
//!   `I`/`J`, emite cada par (i, j) con j > i exactamente una vez, saltando
//!   slots muertos.
//!
//! Invariantes del hot path (zero-allocation):
//! - Sin `Vec`, `String`, `format!`, `Box`, closures que capturen.
//! - Sólo punteros crudos, aritmética de enteros y bucles `while`/`for`.
//! - `rebuild_grid` resetea cabeceras a -1 e inserta solo cuerpos vivos,
//!   coordenadas clampadas al borde.
//! - Nunca escribe fuera de `bucket_head` (bounds check con comparación sin
//!   signo en `get_cell`).

use crate::store::NativeStore;
use std::alloc::{alloc, dealloc, Layout};
use std::ptr::NonNull;

// ============================================================================
// SpatialGrid — POD repr(C) sobre buffers unmanaged
// ============================================================================

/// Rejilla espacial de celdas fijas preasignada.
///
/// `bucket_head` tiene `bucket_count` entradas (una por celda), `next` tiene
/// `body_count_max` entradas (una por slot de entidad) formando listas
/// enlazadas implícitas por bucket.
#[repr(C)]
pub struct SpatialGrid {
    pub bucket_head: *mut i32,
    pub next: *mut i32,
    pub cell_size: f32,
    pub inv_cell_size: f32,
    pub grid_width: i32,
    pub grid_height: i32,
    pub bucket_count: i32,
    pub body_count_max: i32,
}

/// Errores del SpatialGrid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpatialGridError {
    ZeroCapacity,
    AllocationFailed,
}

impl SpatialGrid {
    /// Reserva una rejilla de `grid_width x grid_height` buckets con capacidad
    /// para `body_count_max` entidades. Ambos buffers alineados a 64 B en una
    /// sola asignación consolidada.
    pub fn allocate(
        grid_width: i32,
        grid_height: i32,
        body_count_max: i32,
    ) -> Result<Self, SpatialGridError> {
        if grid_width <= 0 || grid_height <= 0 || body_count_max <= 0 {
            return Err(SpatialGridError::ZeroCapacity);
        }
        let bucket_count = (grid_width as usize) * (grid_height as usize);

        let head_bytes = bucket_count * core::mem::size_of::<i32>();
        let next_bytes = (body_count_max as usize) * core::mem::size_of::<i32>();
        let total_bytes = head_bytes + next_bytes;
        let layout = match Layout::from_size_align(total_bytes, 64) {
            Ok(l) => l,
            Err(_) => return Err(SpatialGridError::AllocationFailed),
        };
        let ptr = unsafe { alloc(layout) };
        let base = match NonNull::new(ptr as *mut u8) {
            Some(p) => p,
            None => return Err(SpatialGridError::AllocationFailed),
        };

        // Inicializar bucket_head a -1 (0xFF en cada byte de i32).
        unsafe {
            std::ptr::write_bytes(base.as_ptr(), 0xFF, total_bytes);
        }

        let bucket_head = base.as_ptr() as *mut i32;
        let next = unsafe { base.as_ptr().add(head_bytes) as *mut i32 };

        Ok(Self {
            bucket_head,
            next,
            cell_size: 64.0f32,
            inv_cell_size: 1.0f32 / 64.0f32,
            grid_width,
            grid_height,
            bucket_count: bucket_count as i32,
            body_count_max,
        })
    }

    /// Libera el buffer consolidado. No hace nada si `bucket_head` es nulo.
    pub unsafe fn dispose(grid: &mut Self) {
        if grid.bucket_head.is_null() {
            return;
        }
        let bucket_count = grid.bucket_count as usize;
        let body_count_max = grid.body_count_max as usize;
        let total_bytes =
            bucket_count * core::mem::size_of::<i32>() + body_count_max * core::mem::size_of::<i32>();
        let layout = match Layout::from_size_align(total_bytes, 64) {
            Ok(l) => l,
            Err(_) => return,
        };
        unsafe {
            dealloc(grid.bucket_head as *mut u8, layout);
        }
        grid.bucket_head = core::ptr::null_mut();
        grid.next = core::ptr::null_mut();
        grid.bucket_count = 0;
        grid.body_count_max = 0;
    }

    /// Cabeza de la lista del bucket (cx, cy) o -1 si vacío/fuera de rango.
    /// Comparación sin signo para el rango (coordenadas negativas -> -1).
    #[inline(always)]
    pub fn get_cell(&self, cx: i32, cy: i32) -> i32 {
        if (cx as u32) >= (self.grid_width as u32) || (cy as u32) >= (self.grid_height as u32) {
            return -1;
        }
        let b = (cy as usize) * (self.grid_width as usize) + (cx as usize);
        unsafe { *self.bucket_head.add(b) }
    }

    /// Configura cell_size e inv_cell_size (llamar tras allocate).
    #[inline(always)]
    pub fn set_cell_size(&mut self, cell_size: f32) {
        self.cell_size = cell_size;
        self.inv_cell_size = 1.0 / cell_size;
    }

    /// Cabeza de la lista de la vecina `neighbor_idx` (0..9) del bucket
    /// `bucket`. Retorna -1 si la vecina está fuera de rango o vacía.
    #[inline(always)]
    pub fn neighbor_head(&self, bucket: usize, neighbor_idx: usize) -> i32 {
        let gw = self.grid_width as usize;
        let bx = (bucket % gw) as i32;
        let by = (bucket / gw) as i32;
        let dx = [ -1i32, 0, 1, -1, 0, 1, -1, 0, 1 ][neighbor_idx];
        let dy = [ -1i32, -1, -1, 0, 0, 0, 1, 1, 1 ][neighbor_idx];
        self.get_cell(bx + dx, by + dy)
    }
}

impl Drop for SpatialGrid {
    fn drop(&mut self) {
        if !self.bucket_head.is_null() {
            unsafe { Self::dispose(self) };
        }
    }
}

// ============================================================================
// Rebuild del grid (O(N + B))
// ============================================================================

/// Reconstruye el grid sobre los cuerpos vivos del `store`.
///
/// - Resetea todas las cabeceras de bucket a -1 (bucle O(B)).
/// - Para cada cuerpo vivo, computa `cell_x = x * inv_cell_size`, `cell_y =
///   y * inv_cell_size`, clampeado a [0, grid_width-1] / [0, grid_height-1],
///   mapea a `bucket_idx = cell_y * grid_width + cell_x` y prepende a la
///   lista del bucket: `next[i] = bucket_head[b]; bucket_head[b] = i`.
///
/// Zero-allocation: solo punteros crudos y bucles `while`. Nunca escribe
/// fuera de `bucket_head`.
pub fn rebuild_grid(grid: &SpatialGrid, store: &NativeStore) {
    unsafe {
        let bucket_count = grid.bucket_count as usize;
        let mut b: usize = 0;
        while b < bucket_count {
            *grid.bucket_head.add(b) = -1;
            b += 1;
        }

        let live_count = store.live_count;
        let x = store.x;
        let y = store.y;
        let live_indices = store.live_indices;

        let gw = grid.grid_width as usize;
        let inv_cell = grid.inv_cell_size;
        let max_cx = grid.grid_width - 1;
        let max_cy = grid.grid_height - 1;

        let mut p: usize = 0;
        while p < live_count {
            let i = *live_indices.add(p) as usize;
            let px = *x.add(i);
            let py = *y.add(i);

            let mut cx = (px * inv_cell) as i32;
            let mut cy = (py * inv_cell) as i32;
            if cx < 0 {
                cx = 0;
            } else if cx > max_cx {
                cx = max_cx;
            }
            if cy < 0 {
                cy = 0;
            } else if cy > max_cy {
                cy = max_cy;
            }

            let bucket_idx = (cy as usize) * gw + (cx as usize);
            let head = *grid.bucket_head.add(bucket_idx);
            *grid.next.add(i) = head;
            *grid.bucket_head.add(bucket_idx) = i as i32;

            p += 1;
        }
    }
}

// ============================================================================
// SpatialPairEnumerator — struct por valor, sin asignaciones
// ============================================================================

/// Par de índices de slot (i, j) con j > i.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpatialPair {
    pub i: i32,
    pub j: i32,
}

/// Enumerador de pares de la rejilla por valor.
///
/// Recorre cada bucket y sus 8 vecinas (3x3). Para cada cuerpo vivo `i` en el
/// bucket actual, camina la lista de cada vecina emitiendo (i, j) solo si
/// `j > i`. Los slots muertos (mass < 0) se descartan al leer la lista.
///
/// Uso:
///   let mut e = SpatialPairEnumerator::new(&grid, store);
///   while e.move_next() {
///       let p = e.current();
///       // usar p.i, p.j
///   }
#[derive(Clone, Copy)]
pub struct SpatialPairEnumerator<'a> {
    grid: &'a SpatialGrid,
    store: &'a NativeStore,
    /// Bucket actual para i.
    bucket_cursor: usize,
    /// Cuerpo i actual (nodo en lista del bucket_cursor).
    current_i: i32,
    /// Índice de vecina (0..9) para current_i.
    neighbor_idx: usize,
    /// Nodo j actual de la lista de la vecina.
    current_j: i32,
    /// Par encontrado listo para retornar.
    current_pair: SpatialPair,
}

impl<'a> SpatialPairEnumerator<'a> {
    /// Crea el enumerador sobre el grid y store dados.
    #[inline(always)]
    pub fn new(grid: &'a SpatialGrid, store: &'a NativeStore) -> Self {
        Self {
            grid,
            store,
            bucket_cursor: 0,
            current_i: -1,
            neighbor_idx: 0,
            current_j: -1,
            current_pair: SpatialPair { i: -1, j: -1 },
        }
    }

    /// Avanza al siguiente par válido. Retorna `false` cuando se agotan los
    /// buckets y listas.
    #[inline(always)]
    pub fn move_next(&mut self) -> bool {
        unsafe {
            loop {
                // Fase 1: asegurar current_i vivo válido.
                if self.current_i < 0 {
                    let bucket_count = self.grid.bucket_count as usize;
                    let mut found = false;
                    while self.bucket_cursor < bucket_count {
                        let head = *self.grid.bucket_head.add(self.bucket_cursor);
                        let mut cand = head;
                        while cand >= 0 {
                            if *self.store.mass.add(cand as usize) >= 0.0 {
                                self.current_i = cand;
                                self.neighbor_idx = 0;
                                self.current_j = -1;
                                found = true;
                                break;
                            }
                            cand = *self.grid.next.add(cand as usize);
                        }
                        if found {
                            break;
                        }
                        self.bucket_cursor += 1;
                    }
                    if !found {
                        return false; // no más buckets con cuerpos vivos
                    }
                }

                // Fase 2: procesar vecinas de current_i.
                while self.neighbor_idx < 9 {
                    // current_j se preserva entre llamadas a move_next; solo
                    // (re)iniciar desde la cabeza de la vecina cuando se
                    // empieza una vecina nueva (current_j < 0).
                    if self.current_j < 0 {
                        let vec_head = self.grid
                            .neighbor_head(self.bucket_cursor, self.neighbor_idx);
                        self.current_j = vec_head;
                    }
                    while self.current_j >= 0 {
                        let j = self.current_j;
                        self.current_j = *self.grid.next.add(j as usize);
                        // j muerto -> saltar.
                        if *self.store.mass.add(j as usize) < 0.0 {
                            continue;
                        }
                        // j == i -> mismo cuerpo, saltar.
                        if j == self.current_i {
                            continue;
                        }
                        // Emitir solo si j > i (cada par no ordenado una vez).
                        if j > self.current_i {
                            self.current_pair = SpatialPair {
                                i: self.current_i,
                                j,
                            };
                            return true;
                        }
                        // j < i: ya se emitió cuando i era el menor.
                    }
                    self.neighbor_idx += 1;
                }

                // Fase 3: avanzar current_i a siguiente cuerpo vivo en su bucket.
                let mut cand = *self.grid.next.add(self.current_i as usize);
                let mut found_next = false;
                while cand >= 0 {
                    if *self.store.mass.add(cand as usize) >= 0.0 {
                        self.current_i = cand;
                        self.neighbor_idx = 0;
                        self.current_j = -1;
                        found_next = true;
                        break;
                    }
                    cand = *self.grid.next.add(cand as usize);
                }
                if found_next {
                    continue; // vuelve al inicio: Fase 1 ve current_i >= 0, va a Fase 2
                }
                // No más vivos en este bucket: pasar al siguiente.
                self.bucket_cursor += 1;
                self.current_i = -1;
                self.neighbor_idx = 0;
                self.current_j = -1;
            }
        }
    }

    /// Retorna el par actual (solo llamar tras `move_next() == true`).
    #[inline(always)]
    pub fn current(&self) -> SpatialPair {
        self.current_pair
    }
}

