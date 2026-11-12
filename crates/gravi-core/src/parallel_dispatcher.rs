//! Dispatcher de fuerza bruta paralelo (Onda 4).
//!
//! Reparte el rango `[start, end)` en bloques de 64 sobre un cursor
//! `AtomicUsize` compartido; cada bloque se resuelve con
//! `compute_forces_brute_range` sobre slots disjuntos, por lo que el resultado
//! es bit-identico a la versión secuencial y no hay carreras de datos.
//!
//! Los workers se crean una sola vez en `new` y duermen en una condvar hasta
//! que `compute_forces_parallel_range` despacha trabajo por frame; no hay
//! busy-wait ni creación de hilos por frame.

use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

use crate::store::NativeStore;

const BLOCK_SIZE: usize = 64;

/// Puntero a `NativeStore` marcado `Send` manualmente.
///
/// `NativeStore` contiene `*mut u8` (no-`Sync`), pero el dispatcher solo lee
/// el store y todos los workers terminan antes de que `compute_forces_parallel_range`
/// retorne, por lo que compartir el puntero entre hilos es seguro.
#[derive(Copy, Clone)]
struct StorePtr(*mut u8);

unsafe impl Send for StorePtr {}

impl StorePtr {
    #[inline(always)]
    unsafe fn as_ref(&self) -> &NativeStore {
        unsafe { &*(self.0 as *const NativeStore) }
    }
}

/// Datos compartidos entre el dispatcher y sus workers (viven en `Arc` para
/// ser 'static en los hilos spawn).
struct Shared {
    /// Cursor de bloques compartido (índice de bloque actual).
    next_block: AtomicUsize,
    /// Límite superior (exclusivo) del rango de objetivos del frame actual.
    end: AtomicUsize,
    /// Puntero al store del frame actual (bits como usize).
    store_bits: AtomicUsize,
    /// `g` del frame actual (bits como u32).
    g_bits: AtomicU32,
}

/// Estado compartido entre el dispatcher y los workers (sincronizado por
/// `Mutex` + `Condvar`).
struct State {
    /// Época del trabajo actual; se incrementa en cada despacho. Los workers
    /// comparan contra su `my_epoch` para saber si hay trabajo nuevo.
    epoch: u64,
    /// Workers aún procesando bloques del frame actual.
    active: usize,
    /// Señal de apagado: los workers deben salir.
    shutdown: bool,
}

/// Dispatcher de fuerza paralela con pool de workers persistentes.
///
/// Los workers spawn en `new` y duermen (condvar) mientras no haya trabajo;
/// despertan solo al despachar trabajo por frame y vuelven a dormir al
/// terminar su parte. `shutdown` une todos los workers sin propagar panics.
pub struct ParallelForceDispatcher {
    workers: Vec<thread::JoinHandle<()>>,
    shared: Arc<Shared>,
    inner: Arc<(Mutex<State>, Condvar)>,
    n_threads: usize,
}

impl ParallelForceDispatcher {
    /// Crea `n_threads` workers que duermen esperando bloques de trabajo.
    pub fn new(n_threads: usize) -> Self {
        let shared = Arc::new(Shared {
            next_block: AtomicUsize::new(0),
            end: AtomicUsize::new(0),
            store_bits: AtomicUsize::new(0),
            g_bits: AtomicU32::new(0),
        });
        let inner = Arc::new((
            Mutex::new(State {
                epoch: 0,
                active: 0,
                shutdown: false,
            }),
            Condvar::new(),
        ));
        let mut workers = Vec::with_capacity(n_threads);
        for _ in 0..n_threads {
            let shared = Arc::clone(&shared);
            let inner = Arc::clone(&inner);
            workers.push(thread::spawn(move || {
                let (m, c) = &*inner;
                let mut my_epoch = 0u64;
                loop {
                    // Esperar hasta que haya trabajo nuevo (epoch cambió) o apagado.
                    let mut st = m.lock().unwrap();
                    while st.epoch == my_epoch && !st.shutdown {
                        st = c.wait(st).unwrap();
                    }
                    if st.shutdown {
                        break;
                    }
                    my_epoch = st.epoch;
                    drop(st);

                    // Procesar bloques hasta cubrir el rango asignado.
                    let g = f32::from_bits(shared.g_bits.load(Ordering::Acquire));
                    let store_ptr = StorePtr(shared.store_bits.load(Ordering::Acquire) as *mut u8);
                    let store = unsafe { store_ptr.as_ref() };
                    let end = shared.end.load(Ordering::Acquire);
                    let mut bs = shared.next_block.fetch_add(BLOCK_SIZE, Ordering::Acquire);
                    while bs < end {
                        let be = (bs + BLOCK_SIZE).min(end);
                        crate::simulation_store::compute_forces_brute_range(store, bs, be, end, g);
                        bs = shared.next_block.fetch_add(BLOCK_SIZE, Ordering::Acquire);
                    }

                    // Marcar este worker como inactivo y notificar al dispatcher.
                    let mut st = m.lock().unwrap();
                    st.active -= 1;
                    if st.active == 0 {
                        c.notify_all();
                    }
                }
            }));
        }
        Self {
            workers,
            shared,
            inner,
            n_threads,
        }
    }

    /// Calcula fuerzas sobre `[start, end)` contra todos los cuerpos `0..end`
    /// repartiendo bloques de 64 entre los workers del pool.
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
        // Sin workers: ejecutar de forma secuencial en el hilo llamante.
        if self.n_threads == 0 {
            crate::simulation_store::compute_forces_brute_range(store, start, end, end, g);
            return;
        }
        // Publicar parámetros del frame (Release, sincronizado con la condvar).
        self.shared.next_block.store(start, Ordering::Release);
        self.shared.end.store(end, Ordering::Release);
        self.shared
            .store_bits
            .store(store as *const NativeStore as *mut u8 as usize, Ordering::Release);
        self.shared.g_bits.store(g.to_bits(), Ordering::Release);

        let (m, c) = &*self.inner;
        {
            let mut st = m.lock().unwrap();
            st.epoch = st.epoch.wrapping_add(1);
            st.active = self.n_threads;
            c.notify_all();
        }
        // Esperar a que todos los workers terminen el frame.
        let mut st = m.lock().unwrap();
        while st.active > 0 {
            st = c.wait(st).unwrap();
        }
    }

    /// Detiene los workers y los une sin propagar panics.
    pub fn shutdown(self) {
        let (m, c) = &*self.inner;
        {
            let mut st = m.lock().unwrap();
            st.shutdown = true;
            st.epoch = st.epoch.wrapping_add(1);
            c.notify_all();
        }
        for worker in self.workers {
            let _ = worker.join();
        }
    }
}
