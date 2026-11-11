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

// ---------------------------------------------------------------------------
// Tests (Issue 8 — criterios de aceptación)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Store de 2 cuerpos con masa unitaria (canales de masa a 1.0).
    fn store_two_bodies() -> NativeStore {
        let store = NativeStore::new(2).unwrap();
        let mass = store.mass_slice();
        mass[0] = 1.0;
        mass[1] = 1.0;
        store
    }

    #[test]
    fn ac_kinetic_energy_two_bodies_unit_speed() {
        let store = store_two_bodies();
        let vx = store.vx_slice();
        let vy = store.vy_slice();
        vx[0] = 1.0;
        vy[0] = 0.0;
        vx[1] = 1.0;
        vy[1] = 0.0;
        let ke = kinetic_energy(&store).unwrap();
        assert!((ke - 1.0).abs() < 1e-4, "ke={}", ke);
    }

    #[test]
    fn ac_potential_energy_is_negative() {
        let store = store_two_bodies();
        let x = store.x_slice();
        let y = store.y_slice();
        x[0] = 0.0;
        y[0] = 0.0;
        x[1] = 1.0;
        y[1] = 0.0;
        let pe = potential_energy(&store).unwrap();
        assert!(pe < 0.0, "pe={} (esperada negativa)", pe);
    }

    #[test]
    fn ac_linear_momentum_two_body_opposing_approx_zero() {
        let store = store_two_bodies();
        let vx = store.vx_slice();
        let vy = store.vy_slice();
        vx[0] = 1.0;
        vy[0] = 0.0;
        vx[1] = -1.0;
        vy[1] = 0.0;
        let (px, py) = linear_momentum(&store).unwrap();
        assert!(px.abs() < 1e-4, "px={}", px);
        assert_eq!(py, 0.0);
    }

    #[test]
    fn ac_center_of_mass_two_equal_bodies() {
        let store = store_two_bodies();
        let x = store.x_slice();
        let y = store.y_slice();
        x[0] = 0.0;
        y[0] = 0.0;
        x[1] = 2.0;
        y[1] = 0.0;
        let (cmx, cmy) = center_of_mass(&store).unwrap();
        assert!((cmx - 1.0).abs() < 1e-4, "cmx={}", cmx);
        assert_eq!(cmy, 0.0);
    }

    #[test]
    fn ac_calculate_flops_per_frame_n_squared_times_12() {
        assert_eq!(calculate_flops_per_frame(0), 0);
        assert_eq!(calculate_flops_per_frame(1), 12);
        assert_eq!(calculate_flops_per_frame(4), 192);
        assert_eq!(calculate_flops_per_frame(16), 3072);
    }

    #[test]
    fn ac_compute_diagnostics_aggregates_all_fields() {
        let store = store_two_bodies();
        let vx = store.vx_slice();
        vx[0] = 1.0;
        vx[1] = -1.0;
        let d = compute_diagnostics(&store).unwrap();
        assert!((d.kinetic_energy - 1.0).abs() < 1e-4);
        assert!(d.potential_energy < 0.0);
        assert!(d.linear_momentum_x.abs() < 1e-4);
        assert_eq!(d.linear_momentum_y, 0.0);
        assert_eq!(d.flops_per_frame, 2 * 2 * 12);
        // total_energy = ke + pe por construcción.
        assert_eq!(d.total_energy, d.kinetic_energy + d.potential_energy);
    }
}
