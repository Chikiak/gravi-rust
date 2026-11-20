---
id: R4-04
title: Paralelizar traversal Barnes-Hut con el dispatcher de fuerza
phase: R4-rendimiento
status: done
branch: feat/r4-04-paralelizar-traversal-barnes-hut
---

## Scope

El traversal Barnes-Hut de `update_physics` en `crates/gravi-ui/src/app.rs`
debe usar `ParallelForceDispatcher` (pool de workers persistente, ya usado por
la ruta de fuerza bruta) en lugar de la llamada secuencial directa a
`compute_forces_barnes_hut_range`. El dispatcher se extiende para soportar
dos kernels (0 = fuerza bruta, 1 = Barnes-Hut) publicados por frame a través
de atómicos (`arena_bits`, `softening_bits`, `theta_bits`, `kernel_bits`),
sin alterar el contrato de `compute_forces_barnes_hut_range` en `quadtree.rs`
(sin cambios en ese archivo).

## Don't do

- No modificar `Scope` ni `Acceptance Criteria` de este issue.
- No pasar un argumento extra `end` a `compute_forces_barnes_hut_range`
  (firma exacta: 7 argumentos).
- No cambiar rama de git ni hacer `git push`; no usar `git reset --hard`.

## Acceptance Criteria

- AC1: `tests/gravi-core-tests/src/parallel.rs` —
  `ac_barnes_hut_parallel_bit_identical_to_sequential`: n=200, ejecución
  secuencial vs paralela con 4 hilos, `fx`/`fy` bit-identicos
  (`assert_eq`, tolerancia 0).
- AC2: `ac_barnes_hut_parallel_salta_muros`: count=200, live=100, slots
  muertos entre vivos (`live_indices` = índices pares), comparar `fx`/`fy`
  para 0..live.

## Implementación

- `crates/gravi-core/src/parallel_dispatcher.rs`:
  - import `crate::quadtree::QuadtreeArena`;
  - `Shared` gana `arena_bits: AtomicUsize`, `softening_bits: AtomicU32`,
    `theta_bits: AtomicU32`, `kernel_bits: AtomicU32` (inicializados a 0 en
    `new()`);
  - worker loop carga `kernel_bits`/`arena_bits`/`softening_bits`/`theta_bits`
    (Acquire) y, dentro de `while bs < end`, llama a
    `compute_forces_barnes_hut_range(arena, store, bs, be, g, softening, theta)`
    si `kernel == 1`, salvo `compute_forces_brute_range(store, bs, be, end, g)`
    en caso contrario;
  - nuevo método
    `compute_forces_barnes_hut_parallel(&self, arena, store, start, end, g,
    softening, theta)` que publica parámetros (Release, `kernel_bits` último),
    bump de epoch, `notify_all` y espera en condvar hasta `active == 0`;
    `n_threads == 0` → secuencial en el llamante.
- `crates/gravi-ui/src/app.rs`: rama `BarnesHut` de `update_physics` llama a
  `self.dispatcher.compute_forces_barnes_hut_parallel(...)` tras
  `build_quadtree` + `compute_quadtree_masses`; se elimina
  `compute_forces_barnes_hut_range` del `use` de quadtree.
- `tests/gravi-core-tests/src/parallel.rs`: helper `make_store_bh(n)` +
  tests AC1 y AC2.

## Notas de implementación

- Rama: `feat/r4-04-paralelizar-traversal-barnes-hut` (desde `main`).
- Commit único: `R4-04: paralelizar traversal Barnes-Hut con el dispatcher`.
- Merge `--no-ff` a `main` y borrado de rama feature tras CI verde.
- Orden de publicación de atómicos: `kernel_bits` se almacena último para
  que un worker que observe `kernel == 1` vea ya `arena_bits`,
  `softening_bits` y `theta_bits` (happens-before vía release-acquire en
  `kernel_bits`).
- Sin cambios en `crates/gravi-core/src/quadtree.rs`.
