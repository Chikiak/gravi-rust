//! Atractor estático (POD).
//!
//! Cuerpo inmune a la gravedad que ejerce atracción sobre los demás.
//! Traducción fiel de `GraviSharp.Core.StaticAttractor`.

use crate::store::NativeStore;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct StaticAttractor {
    pub x: f32,
    pub y: f32,
    pub mass: f32,
}

impl StaticAttractor {
    /// Crea un atractor estático en `(x, y)` con la masa dada.
    pub fn new(x: f32, y: f32, mass: f32) -> Self {
        Self { x, y, mass }
    }
}

/// Acumula las fuerzas de los atractores estáticos sobre los cuerpos vivos
/// (sección 2 de `ComputeForcesBruteWithAttractors` en C#).
/// Zero-alloc: solo punteros de canal y aritmética de punto flotante.
pub fn apply_attractor_forces(store: &mut NativeStore, attractors: &[StaticAttractor], g: f32, softening: f32) {
    if attractors.is_empty() {
        return;
    }
    let live_count = store.live_count;
    if live_count == 0 {
        return;
    }
    let softening_sq = softening * softening;
    unsafe {
        let x = store.x;
        let y = store.y;
        let fx = store.fx;
        let fy = store.fy;
        let live_indices = store.live_indices;
        for attr in attractors.iter() {
            let am = attr.mass;
            if am <= 0.0 {
                continue;
            }
            let ax = attr.x;
            let ay = attr.y;
            let g_am = g * am;
            for p in 0..live_count {
                let i = *live_indices.add(p) as usize;
                let dx = ax - *x.add(i);
                let dy = ay - *y.add(i);
                let dist_sq = dx * dx + dy * dy + softening_sq;
                let inv_dist3 = 1.0f32 / (dist_sq * dist_sq.sqrt());
                let fbh = g_am * inv_dist3;
                *fx.add(i) += fbh * dx;
                *fy.add(i) += fbh * dy;
            }
        }
    }
}
