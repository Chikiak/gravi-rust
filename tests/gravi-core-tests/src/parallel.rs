//! Pruebas de integración extraídas de crates/gravi-core/src/parallel_dispatcher.rs
//! (fuerza bruta paralelo, Issue 14 / Onda 4).

use gravi_core::parallel_dispatcher::ParallelForceDispatcher;
use gravi_core::simulation_store::compute_forces_brute_range;
use gravi_core::store::NativeStore;

fn make_store(n: usize) -> NativeStore {
    let store = NativeStore::new(n).expect("NativeStore::new falló");
    for i in 0..n {
        unsafe {
            *store.x.add(i) = (i as f32) * 0.7 - 50.0;
            *store.y.add(i) = (i as f32) * 1.1 - 30.0;
            *store.mass.add(i) = 1.0 + (i as f32) * 0.3;
        }
    }
    store
}

#[test]
fn ac_parallel_bit_identical_to_sequential() {
    let n = 200;
    let g = 1.0f32;

    let seq = make_store(n);
    compute_forces_brute_range(&seq, 0, n, n, g);
    let seq_fx = unsafe { std::slice::from_raw_parts(seq.fx, n) }.to_vec();
    let seq_fy = unsafe { std::slice::from_raw_parts(seq.fy, n) }.to_vec();

    let par = make_store(n);
    let dispatcher = ParallelForceDispatcher::new(4);
    dispatcher.compute_forces_parallel_range(&par, 0, n, g);
    dispatcher.shutdown();
    let par_fx = unsafe { std::slice::from_raw_parts(par.fx, n) }.to_vec();
    let par_fy = unsafe { std::slice::from_raw_parts(par.fy, n) }.to_vec();

    assert_eq!(seq_fx, par_fx, "Fx no bit-identico");
    assert_eq!(seq_fy, par_fy, "Fy no bit-identico");
}

#[test]
fn ac_shutdown_no_panic() {
    let dispatcher = ParallelForceDispatcher::new(2);
    dispatcher.shutdown();
}

#[test]
/// Metodología: medir tiempos secuencial y paralelo con warm-up + iteraciones,
/// reportar medianas y speedup. Regla de decisión de `#[ignore]`: quitar SOLO si
/// estable con speedup >= 1.05; si no, mantener con causa escrita y cifras.
fn ac_speedup_gt_one_at_5000() {
    const N: usize = 5000;
    const G: f32 = 1.0;
    const WARMUP: usize = 3;
    const ITERS: usize = 7;

    let n_threads = std::thread::available_parallelism()
        .map(|p| p.get().max(2))
        .unwrap_or(4);

    // Secuencial: warm-up (descartadas) + ITERS con make_store fresca fuera del timer.
    let mut t_seq_vec: Vec<std::time::Duration> = Vec::with_capacity(ITERS);
    for _ in 0..WARMUP {
        let store = make_store(N);
        compute_forces_brute_range(&store, 0, N, N, G);
    }
    for _ in 0..ITERS {
        let store = make_store(N);
        let t0 = std::time::Instant::now();
        compute_forces_brute_range(&store, 0, N, N, G);
        let elapsed = t0.elapsed();
        t_seq_vec.push(elapsed);
        println!("t_seq iter={:?}", elapsed);
    }
    t_seq_vec.sort_unstable();
    let t_seq_med = t_seq_vec[ITERS / 2];

    // Paralelo: un único dispatcher fuera del bucle; shutdown después, coste aparte.
    let par = make_store(N);
    let dispatcher = ParallelForceDispatcher::new(n_threads);
    let mut t_par_vec: Vec<std::time::Duration> = Vec::with_capacity(ITERS);
    for _ in 0..WARMUP {
        dispatcher.compute_forces_parallel_range(&par, 0, N, G);
    }
    for _ in 0..ITERS {
        let t0 = std::time::Instant::now();
        dispatcher.compute_forces_parallel_range(&par, 0, N, G);
        let elapsed = t0.elapsed();
        t_par_vec.push(elapsed);
        println!("t_par iter={:?}", elapsed);
    }
    let t0 = std::time::Instant::now();
    dispatcher.shutdown();
    let t_shutdown = t0.elapsed();
    t_par_vec.sort_unstable();
    let t_par_med = t_par_vec[ITERS / 2];

    let speedup = t_seq_med.as_secs_f64() / t_par_med.as_secs_f64();
    println!(
        "t_seq(med)={:?} t_par(med)={:?} t_shutdown={:?} speedup={:.2}x n_threads={} warmup={} iters={}",
        t_seq_med,
        t_par_med,
        t_shutdown,
        speedup,
        n_threads,
        WARMUP,
        ITERS
    );

    assert!(
        t_par_med < t_seq_med,
        "sin speedup: seq={:?} par={:?}",
        t_seq_med,
        t_par_med
    );
}
