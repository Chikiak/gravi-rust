//! Diagnósticos de física del motor N-body.
//! Zero-allocation en el hot path: solo lecturas sobre slices existentes
//! (`&NativeStore`), sin `Vec`, `String`, `format!`, `Box` ni closures que
//! capturen. Todos los cálculos son deterministas (IEEE 754).

use crate::constants::{PhysicsConstants, SOFTENING_SQ};
use crate::store::NativeStore;

/// Resultado agregado de los diagnósticos de física de un frame.
#[derive(Debug, Clone, Copy)]
pub struct Diagnostics {
    pub kinetic_energy: f32,
    pub potential_energy: f32,
    pub total_energy: f32,
    pub linear_momentum_x: f32,
    pub linear_momentum_y: f32,
    pub center_of_mass_x: f32,
    pub center_of_mass_y: f32,
    pub flops_per_frame: u64,
}

/// Energía cinética total: Σ 0.5 * m_i * v_i²  (solo cuerpos vivos).
pub fn kinetic_energy(store: &NativeStore) -> Result<f32, String> {
    let live_count = store.live_count;
    debug_assert!(live_count > 0, "kinetic_energy: store vacío");
    let live_indices = store.live_indices_slice();
    let mass = store.mass_slice();
    let vx = store.vx_slice();
    let vy = store.vy_slice();
    let mut ke: f32 = 0.0;
    let mut i = 0;
    while i < live_count {
        let idx = live_indices[i] as usize;
        let m = mass[idx];
        if m < 0.0 {
            i += 1;
            continue;
        }
        let vx2 = vx[idx] * vx[idx];
        let vy2 = vy[idx] * vy[idx];
        ke = ke + 0.5 * m * (vx2 + vy2);
        i += 1;
    }
    Ok(ke)
}

/// Energía potencial total (negativa): Σ -G * m_i * m_j / sqrt(r2 + SOFTENING_SQ)  (solo cuerpos vivos).
pub fn potential_energy(store: &NativeStore) -> Result<f32, String> {
    let live_count = store.live_count;
    debug_assert!(live_count > 0, "potential_energy: store vacío");
    let live_indices = store.live_indices_slice();
    let mass = store.mass_slice();
    let x = store.x_slice();
    let y = store.y_slice();
    let g = PhysicsConstants::default().g;
    let mut pe: f64 = 0.0;
    let mut i = 0;
    while i < live_count {
        let ii = live_indices[i] as usize;
        let mi = mass[ii];
        if mi < 0.0 {
            i += 1;
            continue;
        }
        let mut j = i + 1;
        while j < live_count {
            let jj = live_indices[j] as usize;
            let mj = mass[jj];
            if mj < 0.0 {
                j += 1;
                continue;
            }
            let dx = (x[jj] - x[ii]) as f64;
            let dy = (y[jj] - y[ii]) as f64;
            let dist2 = dx * dx + dy * dy + SOFTENING_SQ as f64;
            let inv_dist = dist2.sqrt().recip();
            pe = pe - (g as f64) * (mi as f64) * (mj as f64) * inv_dist;
            j += 1;
        }
        i += 1;
    }
    Ok(pe as f32)
}

/// Momento lineal total: (Σ m_i * vx_i, Σ m_i * vy_i)  (solo cuerpos vivos).
pub fn linear_momentum(store: &NativeStore) -> Result<(f32, f32), String> {
    let live_count = store.live_count;
    debug_assert!(live_count > 0, "linear_momentum: store vacío");
    let live_indices = store.live_indices_slice();
    let mass = store.mass_slice();
    let vx = store.vx_slice();
    let vy = store.vy_slice();
    let mut px: f32 = 0.0;
    let mut py: f32 = 0.0;
    let mut i = 0;
    while i < live_count {
        let idx = live_indices[i] as usize;
        let m = mass[idx];
        if m < 0.0 {
            i += 1;
            continue;
        }
        px = px + m * vx[idx];
        py = py + m * vy[idx];
        i += 1;
    }
    Ok((px, py))
}

/// Centro de masas: (Σ m_i * x_i / M, Σ m_i * y_i / M)  (solo cuerpos vivos).
pub fn center_of_mass(store: &NativeStore) -> Result<(f32, f32), String> {
    let live_count = store.live_count;
    debug_assert!(live_count > 0, "center_of_mass: store vacío");
    let live_indices = store.live_indices_slice();
    let mass = store.mass_slice();
    let x = store.x_slice();
    let y = store.y_slice();
    let mut mx: f32 = 0.0;
    let mut my: f32 = 0.0;
    let mut total_mass: f32 = 0.0;
    let mut i = 0;
    while i < live_count {
        let idx = live_indices[i] as usize;
        let m = mass[idx];
        if m < 0.0 {
            i += 1;
            continue;
        }
        mx = mx + m * x[idx];
        my = my + m * y[idx];
        total_mass = total_mass + m;
        i += 1;
    }
    if total_mass.abs() < 1e-7f32 {
        return Ok((0.0, 0.0));
    }
    Ok((mx / total_mass, my / total_mass))
}

/// Operaciones de punto flotante por frame: N*N*12.
pub fn calculate_flops_per_frame(body_count: usize) -> u64 {
    (body_count as u64) * (body_count as u64) * 12
}

/// Agrega todos los diagnósticos en una sola estructura.
pub fn compute_diagnostics(store: &NativeStore) -> Result<Diagnostics, String> {
    let ke = kinetic_energy(store)?;
    let pe = potential_energy(store)?;
    let (px, py) = linear_momentum(store)?;
    let (cmx, cmy) = center_of_mass(store)?;
    let flops = calculate_flops_per_frame(store.count);
    Ok(Diagnostics {
        kinetic_energy: ke,
        potential_energy: pe,
        total_energy: ke + pe,
        linear_momentum_x: px,
        linear_momentum_y: py,
        center_of_mass_x: cmx,
        center_of_mass_y: cmy,
        flops_per_frame: flops,
    })
}


