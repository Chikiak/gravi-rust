//! gravi-core — Motor de gravitación 2D N-body (porte de GraviSharp.Core a Rust).
//!
//! Contrato: zero-allocation en el hot path de simulación, almacenamiento SoA sobre
//! memoria nativa alineada, índices generacionales, quadtree plano para Barnes-Hut,
//! rejilla espacial para colisiones y kernels de fuerza brute-force / Barnes-Hut con
//! SIMD portable (std::simd) y fallback scalar.
//!
//! Todo el hot path vive en bloques `unsafe` con punteros crudos (*mut f32, *mut i32),
//! equivalente a los punteros unmanaged de C#.

#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(non_snake_case)]
#![feature(portable_simd)]

pub mod constants;
pub mod physics_step;
pub mod static_attractor;
pub mod native_mem;
pub mod store;
pub mod simulation_store;
pub mod entity_lifecycle;
pub mod quadtree;
pub mod spatial_grid;
pub mod collisions;
pub mod diagnostics;
pub mod verifier;
pub mod scenario;
pub mod parallel_dispatcher;
