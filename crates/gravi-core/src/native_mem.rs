//! native_mem — wrappers seguros sobre `alloc`/`dealloc` de la API global.
//! (Issue 2: Memoria nativa y NativeStore — base mínima para SimulationStore.)

use std::alloc::{alloc, dealloc, Layout};
use std::ptr::NonNull;

/// Reserva un bloque de memoria nativa alineado a `align` bytes.
/// Devuelve `None` si la asignación falla.
pub unsafe fn aligned_alloc(size: usize, align: usize) -> Option<NonNull<[u8]>> {
    let layout = Layout::from_size_align(size, align).ok()?;
    let ptr = unsafe { alloc(layout) };
    NonNull::new(ptr).map(|p| NonNull::slice_from_raw_parts(p, size))
}

/// Libera un bloque previamente reservado con `aligned_alloc`.
/// `ptr` debe ser no nulo y haber sido obtenido de `aligned_alloc` con el
/// mismo `size` y `align`.
pub unsafe fn aligned_free(ptr: NonNull<u8>, size: usize, align: usize) {
    let layout = match Layout::from_size_align(size, align) {
        Ok(l) => l,
        Err(_) => return,
    };
    unsafe {
        dealloc(ptr.as_ptr(), layout);
    }
}
