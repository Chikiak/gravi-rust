//! heap_verifier — verificador de asignaciones del heap (Onda 4).
//! Zero-allocation en el hot path: sin `Vec`, `String`, `format!`, `Box` ni
//! closures. `begin_frame`/`end_frame` miden el delta neto de bytes del heap
//! con un `GlobalAlloc` personalizado sobre el allocador del sistema.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicUsize, Ordering};

// ---------------------------------------------------------------------------
// Contador global de bytes del heap (netos, nunca negativos).
// thread_local para evitar carreras entre hilos de prueba que corren en
// paralelo (cargo test ejecuta tests de un mismo binary en hilos distintos).
// ---------------------------------------------------------------------------

thread_local! {
    static HEAP_BYTES: AtomicUsize = AtomicUsize::new(0);
}

/// Accesos al contador del heap del hilo actual (evita carreras entre hilos
/// de prueba que corren en paralelo bajo `cargo test`).
fn heap_add(n: usize) {
    HEAP_BYTES.with(|c| c.fetch_add(n, Ordering::Relaxed));
}
fn heap_sub(n: usize) {
    HEAP_BYTES.with(|c| c.fetch_sub(n, Ordering::Relaxed));
}
fn heap_load() -> usize {
    HEAP_BYTES.with(|c| c.load(Ordering::Relaxed))
}
fn heap_store(n: usize) {
    HEAP_BYTES.with(|c| c.store(n, Ordering::Relaxed));
}

/// Allocador que envuelve `System` y cuenta asignaciones/liberaciones del heap.
pub struct CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            heap_add(layout.size());
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            heap_add(layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        heap_sub(layout.size());
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let old = layout.size();
        let ptr = unsafe { System.realloc(ptr, layout, new_size) };
        if !ptr.is_null() {
            let delta = new_size as isize - old as isize;
            if delta > 0 {
                heap_add(delta as usize);
            } else {
                heap_sub((-delta) as usize);
            }
        }
        ptr
    }
}

/// Realiza una asignación de heap real pasando por CountingAllocator.
/// Usada por tests de integración para verificar que HEAP_BYTES cuenta
/// asignaciones reales.
pub fn probe_allocate_usize() -> Box<usize> {
    Box::new(1usize)
}

// ---------------------------------------------------------------------------
// Verificador por frame (single-thread: usa UnsafeCell para evitar `Cell`
// + borrows mutables en el hot path).
// ---------------------------------------------------------------------------

/// Reporte de un frame verificado. `Copy` y sin asignación en su construcción.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeapReport {
    pub frame_id: u64,
    pub heap_bytes_start: usize,
    pub heap_bytes_end: usize,
    pub delta_bytes: isize,
    pub max_delta_bytes: isize,
}

/// Verificador ligero de heap por frame.
///
/// Uso:
/// ```ignore
/// let mut hv = HeapVerifier::new();
/// hv.begin_frame();
/// // ... hot path (sin asignar) ...
/// let report = hv.end_frame();
/// assert_eq!(report.delta_bytes, 0);
/// ```
pub struct HeapVerifier {
    frame_id: u64,
    start_bytes: UnsafeCell<usize>,
    max_delta: isize,
}

impl Default for HeapVerifier {
    fn default() -> Self {
        Self::new()
    }
}

impl HeapVerifier {
    /// Crea un verificador con `max_delta = 0` y `frame_id = 0`.
    pub fn new() -> Self {
        Self {
            frame_id: 0,
            start_bytes: UnsafeCell::new(0),
            max_delta: 0,
        }
    }

    /// Marca el inicio de un frame: captura los bytes del heap actuales.
    /// Debe llamarse antes del hot path del frame.
    pub fn begin_frame(&mut self) {
        self.frame_id = self.frame_id.wrapping_add(1);
        unsafe {
            *self.start_bytes.get() = heap_load();
        }
    }

    /// Marca el fin de un frame y devuelve el reporte. Actualiza
    /// `max_delta` si el delta neto de este frame lo supera.
    pub fn end_frame(&mut self) -> HeapReport {
        let end = heap_load();
        let start = unsafe { *self.start_bytes.get() };
        let delta = end as isize - start as isize;
        if delta > self.max_delta {
            self.max_delta = delta;
        }
        HeapReport {
            frame_id: self.frame_id,
            heap_bytes_start: start,
            heap_bytes_end: end,
            delta_bytes: delta,
            max_delta_bytes: self.max_delta,
        }
    }

    /// Devuelve el delta máximo observado hasta ahora (bytes).
    pub fn max_delta(&self) -> isize {
        self.max_delta
    }

    /// Reinicia `max_delta` a 0 (útil entre suites de verificación).
    pub fn reset_max(&mut self) {
        self.max_delta = 0;
    }

    /// Reinicia el contador global `HEAP_BYTES` a 0.
    /// Útil para aislar mediciones entre tests que comparten el allocador global.
    pub fn reset_heap_bytes(&mut self) {
        heap_store(0);
    }

    /// Devuelve los bytes del heap actualmente contabilizados.
    pub fn current_heap_bytes(&self) -> usize {
        heap_load()
    }
}


