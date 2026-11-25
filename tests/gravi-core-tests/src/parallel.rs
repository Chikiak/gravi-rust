//! Pruebas de integración extraídas de crates/gravi-core/src/parallel_dispatcher.rs
//! (fuerza bruta paralelo, Issue 14 / Onda 4; Barnes-Hut paralelo, R4-04).

use gravi_core::constants::PhysicsConstants;
use gravi_core::parallel_dispatcher::ParallelForceDispatcher;
use gravi_core::quadtree::{
    build_quadtree, compute_forces_barnes_hut_range, compute_quadtree_masses, root_from_live_bodies,
    QuadtreeArena,
};
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

/// Store con `n` cuerpos vivos (lifecycle inicializado, masas y posiciones
/// llenas). Todos los slots están vivos: `live_count == count`.
fn make_store_bh(n: usize) -> NativeStore {
    let mut store = NativeStore::new(n).expect("NativeStore::new falló");
    store.init_lifecycle();
    for i in 0..n {
        unsafe {
            *store.x.add(i) = (i as f32) * 0.7 - 50.0;
            *store.y.add(i) = (i as f32) * 1.1 - 30.0;
            *store.mass.add(i) = 1.0 + (i as f32) * 0.3;
        }
    }
    store
}

/// Construye un store con `count` slots y `live` vivos en índices pares
/// (0, 2, 4, ...), más el quadtree correspondiente. Devuelve (store, arena).
fn bh_store_muros(count: usize, live: usize) -> (NativeStore, QuadtreeArena) {
    let mut store = make_store_bh(count);
    store.live_count = live;
    unsafe {
        for p in 0..live {
            *store.live_indices.add(p) = (2 * p) as i32;
        }
        for i in live..count {
            *store.mass.add(i) = 0.0;
        }
    }
    let (cx, cy, half) = root_from_live_bodies(&store);
    assert!(half > 0.0, "raiz del quadtree vacía con {} vivos", live);
    let mut arena = QuadtreeArena::allocate(count).expect("arena");
    let _ = build_quadtree(&store, &mut arena, cx, cy, half);
    let _ = compute_quadtree_masses(&mut arena, &store);
    (store, arena)
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
fn ac_barnes_hut_parallel_bit_identical_to_sequential() {
    let n = 200;
    let g = 1.0f32;
    let softening = PhysicsConstants::default().softening;
    let theta = PhysicsConstants::default().barnes_hut_theta;

    let seq_store = make_store_bh(n);
    let (cx, cy, half) = root_from_live_bodies(&seq_store);
    assert!(half > 0.0);
    let mut seq_arena = QuadtreeArena::allocate(n).expect("arena");
    let _ = build_quadtree(&seq_store, &mut seq_arena, cx, cy, half);
    let _ = compute_quadtree_masses(&mut seq_arena, &seq_store);
    compute_forces_barnes_hut_range(
        &seq_arena,
        &seq_store,
        0,
        n,
        g,
        softening,
        theta,
    );
    let seq_fx = unsafe { std::slice::from_raw_parts(seq_store.fx, n) }.to_vec();
    let seq_fy = unsafe { std::slice::from_raw_parts(seq_store.fy, n) }.to_vec();

    let par_store = make_store_bh(n);
    let (cx, cy, half) = root_from_live_bodies(&par_store);
    assert!(half > 0.0);
    let mut par_arena = QuadtreeArena::allocate(n).expect("arena");
    let _ = build_quadtree(&par_store, &mut par_arena, cx, cy, half);
    let _ = compute_quadtree_masses(&mut par_arena, &par_store);
    let dispatcher = ParallelForceDispatcher::new(4);
    dispatcher.compute_forces_barnes_hut_parallel(
        &par_arena,
        &par_store,
        0,
        n,
        g,
        softening,
        theta,
    );
    dispatcher.shutdown();
    let par_fx = unsafe { std::slice::from_raw_parts(par_store.fx, n) }.to_vec();
    let par_fy = unsafe { std::slice::from_raw_parts(par_store.fy, n) }.to_vec();

    assert_eq!(seq_fx, par_fx, "Fx no bit-identico (Barnes-Hut paralelo)");
    assert_eq!(seq_fy, par_fy, "Fy no bit-identico (Barnes-Hut paralelo)");
}

#[test]
/// Regresión R6-02: tras un dispatch Barnes-Hut (kernel_bits=1), un dispatch de
/// fuerza bruta sobre el mismo dispatcher debe publicar kernel=0 para que los
/// workers calculen bruta y no reusen el traversal BH del frame anterior.
fn ac_brute_tras_barnes_hut_usa_kernel_bruto() {
    let n = 200;
    let g = 1.0f32;
    let softening = PhysicsConstants::default().softening;
    let theta = PhysicsConstants::default().barnes_hut_theta;

    // (a) Barnes-Hut primero: deja kernel_bits = 1 publicado en el dispatcher.
    let bh_store = make_store_bh(n);
    let (cx, cy, half) = root_from_live_bodies(&bh_store);
    assert!(half > 0.0);
    let mut bh_arena = QuadtreeArena::allocate(n).expect("arena");
    let _ = build_quadtree(&bh_store, &mut bh_arena, cx, cy, half);
    let _ = compute_quadtree_masses(&mut bh_arena, &bh_store);

    let dispatcher = ParallelForceDispatcher::new(4);
    dispatcher.compute_forces_barnes_hut_parallel(
        &bh_arena,
        &bh_store,
        0,
        n,
        g,
        softening,
        theta,
    );

    // (b) Luego fuerza bruta sobre un store fresco con los mismos datos de cuerpos
    //     (misma fórmula que make_store: x = i*0.7 - 50, mass = 1 + i*0.3).
    let par = make_store(n);
    dispatcher.compute_forces_parallel_range(&par, 0, n, g);
    dispatcher.shutdown();
    let par_fx = unsafe { std::slice::from_raw_parts(par.fx, n) }.to_vec();
    let par_fy = unsafe { std::slice::from_raw_parts(par.fy, n) }.to_vec();

    // (c) Referencia: bruta secuencial sobre store idéntico.
    let seq = make_store(n);
    compute_forces_brute_range(&seq, 0, n, n, g);
    let seq_fx = unsafe { std::slice::from_raw_parts(seq.fx, n) }.to_vec();
    let seq_fy = unsafe { std::slice::from_raw_parts(seq.fy, n) }.to_vec();

    assert_eq!(seq_fx, par_fx, "Fx no bit-identico: bruta tras Barnes-Hut debe usar kernel bruto");
    assert_eq!(seq_fy, par_fy, "Fy no bit-identico: bruta tras Barnes-Hut debe usar kernel bruto");
}

#[test]
fn ac_barnes_hut_parallel_salta_muros() {
    let count = 200;
    let live = 100;
    let g = 1.0f32;
    let softening = PhysicsConstants::default().softening;
    let theta = PhysicsConstants::default().barnes_hut_theta;

    let (seq_store, seq_arena) = bh_store_muros(count, live);
    compute_forces_barnes_hut_range(&seq_arena, &seq_store, 0, live, g, softening, theta);
    let seq_fx = unsafe { std::slice::from_raw_parts(seq_store.fx, count) }.to_vec();
    let seq_fy = unsafe { std::slice::from_raw_parts(seq_store.fy, count) }.to_vec();

    let (par_store, par_arena) = bh_store_muros(count, live);
    let dispatcher = ParallelForceDispatcher::new(4);
    dispatcher.compute_forces_barnes_hut_parallel(
        &par_arena,
        &par_store,
        0,
        live,
        g,
        softening,
        theta,
    );
    dispatcher.shutdown();
    let par_fx = unsafe { std::slice::from_raw_parts(par_store.fx, count) }.to_vec();
    let par_fy = unsafe { std::slice::from_raw_parts(par_store.fy, count) }.to_vec();

    // Muros entre vivos (slots impares): deben seguir a cero en ambos.
    for i in 0..count {
        if i % 2 == 1 {
            assert_eq!(seq_fx[i], 0.0, "muro fx[{i}] no es cero (secuencial)");
            assert_eq!(seq_fy[i], 0.0, "muro fy[{i}] no es cero (secuencial)");
            assert_eq!(par_fx[i], 0.0, "muro fx[{i}] no es cero (paralelo)");
            assert_eq!(par_fy[i], 0.0, "muro fy[{i}] no es cero (paralelo)");
        }
    }
    // Vivos (índices pares): bit-identico entre secuencial y paralelo.
    for p in 0..live {
        let i = 2 * p;
        assert_eq!(seq_fx[i], par_fx[i], "Fx vivo[{i}] no bit-identico");
        assert_eq!(seq_fy[i], par_fy[i], "Fy vivo[{i}] no bit-identico");
    }
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
