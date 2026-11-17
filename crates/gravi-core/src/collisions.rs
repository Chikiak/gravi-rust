//! collisions.rs — Colisiones elásticas e inelásticas (Issue 10, Onda 3, parte A).
//!
//! Implementa `resolve_elastic_collision` (rebote elástico con restitución y
//! separación posicional) y `resolve_collisions` (detección sobre `SpatialGrid`
//! + resolución elástica o inelástica/fusión).
//!
//! Contrato de zero-allocation en el hot path:
//! - Sin `Vec`, `String`, `format!`, `Box`, closures que capturen ni LINQ.
//! - Sólo punteros crudos, aritmética de enteros y bucles `while`/`for`.
//! - Sin división flotante en cálculos de simulación (inversas precalculadas).
//! - `mass[slot] >= 0.0` como centinela de slot vivo (Issue 4 / Issue 9).
//! Recompile-forzado: marcador de versión 2 (separación posicional siempre).

use crate::constants::MASS_EPS;
use crate::entity_lifecycle::{is_live, LifecycleError};
use crate::spatial_grid::SpatialGrid;
use crate::store::NativeStore;

/// Configuración de colisiones (POD, Copy, `repr(C)`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CollisionConfig {
    /// Coeficiente de restitución `e` (0 = inelástica perfecta, 1 = elástica pura).
    pub elastic_restitution: f32,
    /// Umbral de velocidad relativa al cuadrado: si `v_rel^2 < umbral` y hay
    /// overlap, se trata como inelástica (fusión).
    pub inelastic_threshold_sq: f32,
    /// Si `true`, fusiones inelásticas están habilitadas; si `false`, toda
    /// colisión se resuelve como elástica (incluso con baja velocidad relativa).
    pub enable_fusion: bool,
    /// Factor de escala del radio tras una fusión: `radius = sqrt(mass) *
    /// radius_per_sqrt_mass` (área ∝ masa en 2D). Con masa 1.0 y
    /// `radius_per_sqrt_mass = 0.5` se recupera el radio por defecto del
    /// store (`0.5`).
    pub radius_per_sqrt_mass: f32,
}

impl Default for CollisionConfig {
    fn default() -> Self {
        Self {
            elastic_restitution: 0.8f32,
            inelastic_threshold_sq: 0.25f32, // (0.5)^2
            enable_fusion: true,
            radius_per_sqrt_mass: 0.5f32,
        }
    }
}

/// Resuelve una colisión elástica entre los cuerpos `i` y `j` (impulse-based
/// con coeficiente de restitución `e`) más separación posicional mínima.
///
/// Fórmulas:
/// - `j = -(1+e) * v_rel_n / (1/m_i + 1/m_j)` (impulso scalar)
/// - `v_i' = v_i - j * n / m_i` ; `v_j' = v_j + j * n / m_j`
/// - Separación posicional: `overlap = r_sum - dist`; se apartan `overlap/2`
///   cada uno a lo largo de la normal.
///
/// Zero-allocation: sólo lecturas/escrituras sobre canales del `store`.
#[inline(always)]
pub unsafe fn resolve_elastic_collision(
    store: &mut NativeStore,
    i: usize,
    j: usize,
    restitution: f32,
) {
    unsafe {
        let x = store.x;
        let y = store.y;
        let vx = store.vx;
        let vy = store.vy;
        let mass = store.mass;
        let radius = store.radius;

        let xi = *x.add(i);
        let yi = *y.add(i);
        let xj = *x.add(j);
        let yj = *y.add(j);

        let dx = xj - xi;
        let dy = yj - yi;
        let dist_sq = dx * dx + dy * dy;

        let ri = *radius.add(i);
        let rj = *radius.add(j);
        let r_sum = ri + rj;
        let r_sum_sq = r_sum * r_sum;

        // No hay overlap (o mismos cuerpos): nada que resolver.
        if dist_sq >= r_sum_sq || dist_sq <= 0.0f32 {
            return;
        }

        // Solo cuerpos que se acercan (v_rel_n < 0) reciben impulso.
        let rvx = *vx.add(j) - *vx.add(i);
        let rvy = *vy.add(j) - *vy.add(i);
        let dist = dist_sq.sqrt();
        let inv_dist = 1.0f32 / dist;
        let nx = dx * inv_dist;
        let ny = dy * inv_dist;
        let v_rel_n = rvx * nx + rvy * ny;
        if v_rel_n < 0.0f32 {
            let mi = *mass.add(i);
            let mj = *mass.add(j);
            // Inversas precalculadas: evita división flotante en el hot path.
            let inv_mi = if mi.abs() > MASS_EPS { 1.0f32 / mi } else { 0.0f32 };
            let inv_mj = if mj.abs() > MASS_EPS { 1.0f32 / mj } else { 0.0f32 };
            let inv_sum = inv_mi + inv_mj;

            let impulse = -(1.0f32 + restitution) * v_rel_n / inv_sum;
            let ix = impulse * nx;
            let iy = impulse * ny;

            *vx.add(i) -= ix * inv_mi;
            *vy.add(i) -= iy * inv_mi;
            *vx.add(j) += ix * inv_mj;
            *vy.add(j) += iy * inv_mj;
        }

        // Separación posicional mínima (evita hundimiento/atrapamiento).
        // Se aplica siempre que hay overlap, incluso si v_rel_n >= 0.
        let overlap = r_sum - dist;
        let half_overlap = overlap * 0.5f32;
        *x.add(i) -= nx * half_overlap;
        *y.add(i) -= ny * half_overlap;
        *x.add(j) += nx * half_overlap;
        *y.add(j) += ny * half_overlap;
    }
}

/// Resuelve una fusión inelástica (colisión perfectamente inelástica)
/// entre `i` (superviviente) y `j` (absorbido).
///
/// Fórmulas (centro de masa, conservación de masa y momento lineal):
/// - `M = m_i + m_j` ; `inv_M = 1 / M` (inversa precalculada: sin división
///   flotante por masa en el hot path)
/// - `(x_i, y_i) = (m_i * p_i + m_j * p_j) * inv_M`
/// - `(vx_i, vy_i) = (m_i * v_i + m_j * v_j) * inv_M`
/// - `Mass[i] = M`
/// - `Radius[i] = sqrt(M) * radius_per_sqrt_mass`
///
/// `j` se marca muerto mediante `entity_lifecycle::mark_dead`:
/// `mass[j] = -1.0`, `generation[j]++`, push LIFO en `free_list`,
/// `live_count--`.
///
/// Zero-allocation: lecturas/escrituras sobre canales del `store` más la
/// llamada a `mark_dead` (solo punteros crudos y aritmética de enteros;
/// sin `Vec`/`String`/`format!`/`Box`/closures).
#[inline(always)]
pub unsafe fn resolve_inelastic_collision(
    store: &mut NativeStore,
    i: usize,
    j: usize,
    radius_per_sqrt_mass: f32,
) -> Result<(), LifecycleError> {
    unsafe {
        let mass = store.mass;
        let x = store.x;
        let y = store.y;
        let vx = store.vx;
        let vy = store.vy;
        let radius = store.radius;

        let mi = *mass.add(i);
        let mj = *mass.add(j);
        let new_m = mi + mj;
        // Inversa precalculada: evita división flotante en el hot path.
        let inv_new_m = if new_m.abs() > MASS_EPS { 1.0f32 / new_m } else { 0.0f32 };

        let xi = *x.add(i);
        let yi = *y.add(i);
        let xj = *x.add(j);
        let yj = *y.add(j);
        let vxi = *vx.add(i);
        let vyi = *vy.add(i);
        let vxj = *vx.add(j);
        let vyj = *vy.add(j);

        // Centro de masas: posición y velocidad del cuerpo fusionado.
        *x.add(i) = (mi * xi + mj * xj) * inv_new_m;
        *y.add(i) = (mi * yi + mj * yj) * inv_new_m;
        *vx.add(i) = (mi * vxi + mj * vxj) * inv_new_m;
        *vy.add(i) = (mi * vyi + mj * vyj) * inv_new_m;
        *mass.add(i) = new_m;
        // Radio: área ∝ masa en 2D → r = sqrt(m) * radius_per_sqrt_mass.
        *radius.add(i) = new_m.sqrt() * radius_per_sqrt_mass;

        // Absorbido marcado muerto: centinela masa = -1.0, generación++,
        // pila LIFO (free_list), live_count--.
        crate::entity_lifecycle::mark_dead(store, j as i32)
    }
}

/// Resuelve todas las colisiones del frame sobre el `store` usando el
/// `SpatialGrid` para obtener pares candidatos (célula + 8 vecinas).
///
/// Para cada par (i, j) vivo con `j > i`:
/// - Si hay overlap esférico (`dist < r_sum`) y `v_rel_n < 0`:
///   - si `enable_fusion` y `v_rel_n^2 < inelastic_threshold_sq` → fusión
///     inelástica (masa+momento conservados, `j` marcado muerto).
///   - si no → colisión elástica (`resolve_elastic_collision`).
///
/// Devuelve el número de colisiones resueltas (elásticas + fusiones).
///
/// Zero-allocation: punteros crudos sobre `bucket_head`/`next` del grid y
/// canales del `store`; ningún `Vec`/`String`/closure.
#[inline(always)]
pub unsafe fn resolve_collisions(
    store: &mut NativeStore,
    grid: &SpatialGrid,
    cfg: CollisionConfig,
) -> i32 {
    unsafe {
        let bucket_count = grid.bucket_count as usize;
        let grid_width = grid.grid_width as usize;
        let bucket_head = grid.bucket_head;
        let next = grid.next;

        let x = store.x;
        let y = store.y;
        let vx = store.vx;
        let vy = store.vy;
        let mass = store.mass;
        let radius = store.radius;

        let mut collisions: i32 = 0;

        // Offsets de las 9 celdas (la propia + 8 vecinas), índice 0..9.
        const NEIGHBOR_DX: [i32; 9] = [-1, 0, 1, -1, 0, 1, -1, 0, 1];
        const NEIGHBOR_DY: [i32; 9] = [-1, -1, -1, 0, 0, 0, 1, 1, 1];

        let mut bucket: usize = 0;
        while bucket < bucket_count {
            let bx = (bucket % grid_width) as i32;
            let by = (bucket / grid_width) as i32;

            // Recorrer cada cuerpo i en la lista del bucket actual.
            let mut i_node = *bucket_head.add(bucket);
            while i_node >= 0 {
                let i = i_node as usize;
                i_node = *next.add(i);

                if !is_live(store, i as i32) {
                    continue;
                }

                let xi = *x.add(i);
                let yi = *y.add(i);
                let ri = *radius.add(i);
                let m_i = *mass.add(i);
                let inv_mi = if m_i.abs() > MASS_EPS { 1.0f32 / m_i } else { 0.0f32 };
                let vx_i = *vx.add(i);
                let vy_i = *vy.add(i);

                let mut ni: usize = 0;
                while ni < 9 {
                    let cx = bx + NEIGHBOR_DX[ni];
                    let cy = by + NEIGHBOR_DY[ni];
                    // Bounds check sin signo (coordenadas negativas -> fuera de rango).
                    if (cx as u32) >= (grid.grid_width as u32)
                        || (cy as u32) >= (grid.grid_height as u32)
                    {
                        ni += 1;
                        continue;
                    }
                    let nb_idx = (cy as usize) * grid_width + (cx as usize);

                    let mut j_node = *bucket_head.add(nb_idx);
                    while j_node >= 0 {
                        let j = j_node as usize;
                        j_node = *next.add(j);

                        if j <= i {
                            continue; // evita pares duplicados y el mismo cuerpo
                        }
                        if !is_live(store, j as i32) {
                            continue;
                        }

                        let dx = *x.add(j) - xi;
                        let dy = *y.add(j) - yi;
                        let dist_sq = dx * dx + dy * dy;

                        let rj = *radius.add(j);
                        let r_sum = ri + rj;
                        let r_sum_sq = r_sum * r_sum;
                        if dist_sq >= r_sum_sq || dist_sq <= 0.0f32 {
                            continue;
                        }

                        let dist = dist_sq.sqrt();
                        let inv_dist = 1.0f32 / dist;
                        let nx = dx * inv_dist;
                        let ny = dy * inv_dist;

                        let rvx = *vx.add(j) - vx_i;
                        let rvy = *vy.add(j) - vy_i;
                        let v_rel_n = rvx * nx + rvy * ny;
                        if v_rel_n >= 0.0f32 {
                            continue; // alejándose: sin colisión
                        }

                        let m_j = *mass.add(j);
                        let inv_mj = if m_j.abs() > MASS_EPS { 1.0f32 / m_j } else { 0.0f32 };
                        let inv_sum = inv_mi + inv_mj;

                        let v_rel_n_sq = v_rel_n * v_rel_n;
                        let do_fusion = cfg.enable_fusion
                            && v_rel_n_sq < cfg.inelastic_threshold_sq;

                        if do_fusion {
                            // Fusión inelástica: conserva masa y momento,
                            // centro de masa, radio = sqrt(M) *
                            // radius_per_sqrt_mass, j marcado muerto.
                            match resolve_inelastic_collision(
                                store,
                                i,
                                j,
                                cfg.radius_per_sqrt_mass,
                            ) {
                                Ok(()) => collisions += 1,
                                Err(_) => {}
                            }
                        } else {
                            // Colisión elástica impulse-based.
                            let impulse = -(1.0f32 + cfg.elastic_restitution)
                                * v_rel_n
                                / inv_sum;
                            let ix = impulse * nx;
                            let iy = impulse * ny;

                            *vx.add(i) -= ix * inv_mi;
                            *vy.add(i) -= iy * inv_mi;
                            *vx.add(j) += ix * inv_mj;
                            *vy.add(j) += iy * inv_mj;

                            // Separación posicional mínima.
                            let overlap = r_sum - dist;
                            let half_overlap = overlap * 0.5f32;
                            *x.add(i) -= nx * half_overlap;
                            *y.add(i) -= ny * half_overlap;
                            *x.add(j) += nx * half_overlap;
                            *y.add(j) += ny * half_overlap;
                            collisions += 1;
                        }
                    }
                    ni += 1;
                }
            }
            bucket += 1;
        }

        collisions
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::MASS_EPS;

    /// Masa por debajo de `MASS_EPS` se trata como cero efectiva: la inversa
    /// es 0.0, por tanto el cuerpo no recibe impulso y su velocidad no cambia.
    #[test]
    fn mass_below_epsilon_is_treated_as_zero() {
        let mut store = NativeStore::new(2).expect("store creation failed");
        store.init_lifecycle();

        // Cuerpo 0: masa bajo epsilon (1e-8 < MASS_EPS = 1e-6, no nula exacta).
        unsafe {
            *store.mass.add(0) = 1e-8f32;
            // Cuerpo 1: masa unitaria.
            *store.mass.add(1) = 1.0f32;
            // Posiciones superpuestas (0 < dist < r_sum).
            *store.x.add(0) = 0.0f32;
            *store.y.add(0) = 0.0f32;
            *store.x.add(1) = 0.1f32;
            *store.y.add(1) = 0.0f32;
            // Velocidades de acercamiento (v_rel_n < 0).
            *store.vx.add(0) = 1.0f32;
            *store.vy.add(0) = 0.0f32;
            *store.vx.add(1) = -1.0f32;
            *store.vy.add(1) = 0.0f32;
        }

        assert!(1e-8f32.abs() < MASS_EPS, "fixture: masa del cuerpo 0 bajo epsilon");

        let vx0_antes = unsafe { *store.vx.add(0) };
        let vy0_antes = unsafe { *store.vy.add(0) };

        unsafe { resolve_elastic_collision(&mut store, 0, 1, 1.0f32) };

        let vx0_despues = unsafe { *store.vx.add(0) };
        let vy0_despues = unsafe { *store.vy.add(0) };

        assert_eq!(vx0_despues, vx0_antes, "masa < MASS_EPS: vx no debe cambiar");
        assert_eq!(vy0_despues, vy0_antes, "masa < MASS_EPS: vy no debe cambiar");
    }
}

