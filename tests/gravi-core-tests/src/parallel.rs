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
#[ignore = "medición de speedup; activar manualmente si es rápida"]
fn ac_speedup_gt_one_at_5000() {
    let n = 5000;
    let g = 1.0f32;

    let seq = make_store(n);
    let t0 = std::time::Instant::now();
    compute_forces_brute_range(&seq, 0, n, n, g);
    let t_seq = t0.elapsed();

    let par = make_store(n);
    let n_threads = std::thread::available_parallelism()
        .map(|p| p.get().max(2))
        .unwrap_or(4);
    let dispatcher = ParallelForceDispatcher::new(n_threads);
    let t0 = std::time::Instant::now();
    dispatcher.compute_forces_parallel_range(&par, 0, n, g);
    dispatcher.shutdown();
    let t_par = t0.elapsed();

    assert!(
        t_par < t_seq,
        "sin speedup: seq={:?} par={:?}",
        t_seq,
        t_par
    );
}
