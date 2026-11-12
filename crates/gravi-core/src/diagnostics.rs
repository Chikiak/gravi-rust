//! Diagnósticos de física del motor N-body.
//! Zero-allocation en el hot path: solo lecturas sobre slices existentes
//! (`&NativeStore`), sin `Vec`, `String`, `format!`, `Box` ni closures que
//! capturen. Todos los cálculos son deterministas (IEEE 754).

use crate::constants::PhysicsConstants;
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

/// Energía cinética total: Σ 0.5 * m_i * v_i²  (sobre todos los cuerpos).
pub fn kinetic_energy(store: &NativeStore) -> Result<f32, String> {
    let count = store.count;
    debug_assert!(count > 0, "kinetic_energy: store vacío");
    let mass = store.mass_slice();
    let vx = store.vx_slice();
    let vy = store.vy_slice();
    let mut ke: f32 = 0.0;
    let mut i = 0;
    while i < count {
        let m = mass[i];
        let vx2 = vx[i] * vx[i];
        let vy2 = vy[i] * vy[i];
        ke = ke + 0.5 * m * (vx2 + vy2);
        i += 1;
    }
    Ok(ke)
}

/// Energía potencial total (negativa): Σ -G * m_i * m_j / r_ij  (i < j).
pub fn potential_energy(store: &NativeStore) -> Result<f32, String> {
    let count = store.count;
    debug_assert!(count > 0, "potential_energy: store vacío");
    let mass = store.mass_slice();
    let x = store.x_slice();
    let y = store.y_slice();
    let g = PhysicsConstants::default().g;
    let mut pe: f32 = 0.0;
    let mut i = 0;
    while i < count {
        let mut j = i + 1;
        while j < count {
            let dx = x[j] - x[i];
            let dy = y[j] - y[i];
            let dist2 = dx * dx + dy * dy;
            let dist = dist2.sqrt();
            // Inverso de la distancia sin división flotante explícita:
            // 1/dist vía operador recíproco (rcp), nunca `/`.
            let inv_dist = dist.recip();
            pe = pe - g * mass[i] * mass[j] * inv_dist;
            j += 1;
        }
        i += 1;
    }
    Ok(pe)
}

/// Momento lineal total: (Σ m_i * vx_i, Σ m_i * vy_i).
pub fn linear_momentum(store: &NativeStore) -> Result<(f32, f32), String> {
    let count = store.count;
    debug_assert!(count > 0, "linear_momentum: store vacío");
    let mass = store.mass_slice();
    let vx = store.vx_slice();
    let vy = store.vy_slice();
    let mut px: f32 = 0.0;
    let mut py: f32 = 0.0;
    let mut i = 0;
    while i < count {
        let m = mass[i];
        px = px + m * vx[i];
        py = py + m * vy[i];
        i += 1;
    }
    Ok((px, py))
}

/// Centro de masas: (Σ m_i * x_i / M, Σ m_i * y_i / M).
pub fn center_of_mass(store: &NativeStore) -> Result<(f32, f32), String> {
    let count = store.count;
    debug_assert!(count > 0, "center_of_mass: store vacío");
    let mass = store.mass_slice();
    let x = store.x_slice();
    let y = store.y_slice();
    let mut mx: f32 = 0.0;
    let mut my: f32 = 0.0;
    let mut total_mass: f32 = 0.0;
    let mut i = 0;
    while i < count {
        let m = mass[i];
        mx = mx + m * x[i];
        my = my + m * y[i];
        total_mass = total_mass + m;
        i += 1;
    }
    debug_assert!(total_mass != 0.0, "center_of_mass: masa total cero");
    if total_mass == 0.0 {
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


