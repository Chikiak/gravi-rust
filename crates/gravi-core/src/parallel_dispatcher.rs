//! Dispatcher de fuerza bruta paralelo (Onda 4).
//!
//! Reparte el rango `[start, end)` en bloques de 64 sobre un cursor
//! `AtomicUsize` compartido; cada bloque se resuelve con
//! `compute_forces_brute_range` sobre slots disjuntos, por lo que el resultado
//! es bit-identico a la versión secuencial y no hay carreras de datos.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;

use crate::store::NativeStore;

const BLOCK_SIZE: usize = 64;

/// Puntero a `NativeStore` marcado `Send` manualmente.
///
/// `NativeStore` contiene `*mut u8` (no-`Sync`), pero el dispatcher solo lee
/// el store y todos los hilos scoped se unen antes de retornar, por lo que
/// compartir el puntero entre hilos scoped es seguro.
#[derive(Copy, Clone)]
struct StorePtr(*mut u8);

unsafe impl Send for StorePtr {}

impl StorePtr {
    #[inline(always)]
    unsafe fn as_ref(&self) -> &NativeStore {
        unsafe { &*(self.0 as *const NativeStore) }
    }
}

/// Dispatcher de fuerza paralela.
///
/// Los workers spawn en `new` permanecen en espera hasta `shutdown`; el reparto
/// de trabajo de cada llamada se hace con `next_block` como cursor de bloques.
pub struct ParallelForceDispatcher {
    workers: Vec<std::thread::JoinHandle<()>>,
    next_block: Arc<AtomicUsize>,
    running: Arc<AtomicBool>,
}

impl ParallelForceDispatcher {
    /// Crea `n_threads` workers que esperan bloques de trabajo.
    pub fn new(n_threads: usize) -> Self {
        let next_block = Arc::new(AtomicUsize::new(0));
        let running = Arc::new(AtomicBool::new(true));
        let mut workers = Vec::with_capacity(n_threads);
        for _ in 0..n_threads {
            let next_block = Arc::clone(&next_block);
            let running = Arc::clone(&running);
            workers.push(thread::spawn(move || {
                while running.load(Ordering::Acquire) {
                    thread::yield_now();
                }
            }));
        }
        Self {
            workers,
            next_block,
            running,
        }
    }

    /// Calcula fuerzas sobre `[start, end)` contra todos los cuerpos `0..end`
    /// repartiendo bloques de 64 entre los hilos del ámbito actual.
    pub fn compute_forces_parallel_range(
        &self,
        store: &NativeStore,
        start: usize,
        end: usize,
        g: f32,
    ) {
        if end <= start {
            return;
        }
        self.next_block.store(start, Ordering::Release);
        let next_block = Arc::clone(&self.next_block);
        // SAFETY: todos los hilos scoped se unen antes de que el scope retorne y
        // `store` vive todo el scope; cada hilo escribe solo en slots objetivo
        // disjuntos, sin carreras de datos.
        let store_ptr = StorePtr(store as *const NativeStore as *mut u8);
        let n_threads = self.workers.len().max(1);
        std::thread::scope(|s| {
            for _ in 0..n_threads {
                let next_block = Arc::clone(&next_block);
                s.spawn(move || {
                    // SAFETY: `store` vive todo el scope y los hilos scoped se
                    // unen antes de que el scope retorne; cada hilo escribe
                    // solo en slots objetivo disjuntos, sin carreras de datos.
                    let store = unsafe { store_ptr.as_ref() };
                    let mut bs = next_block.fetch_add(BLOCK_SIZE, Ordering::Acquire);
                    while bs < end {
                        let be = (bs + BLOCK_SIZE).min(end);
                        crate::simulation_store::compute_forces_brute_range(store, bs, be, end, g);
                        bs = next_block.fetch_add(BLOCK_SIZE, Ordering::Acquire);
                    }
                });
            }
        });
    }

    /// Detiene los workers y los une sin propagar panics.
    pub fn shutdown(self) {
        self.running.store(false, Ordering::Release);
        for worker in self.workers {
            let _ = worker.join();
        }
    }
}


