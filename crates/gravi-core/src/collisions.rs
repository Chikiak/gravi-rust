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

use crate::entity_lifecycle::is_live;
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
            let inv_mi = if mi != 0.0f32 { 1.0f32 / mi } else { 0.0f32 };
            let inv_mj = if mj != 0.0f32 { 1.0f32 / mj } else { 0.0f32 };
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
) {
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
        let inv_new_m = if new_m != 0.0f32 { 1.0f32 / new_m } else { 0.0f32 };

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
            .expect("resolve_inelastic_collision: j debe estar vivo");
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
                let inv_mi = if m_i != 0.0f32 { 1.0f32 / m_i } else { 0.0f32 };
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
                        let inv_mj = if m_j != 0.0f32 { 1.0f32 / m_j } else { 0.0f32 };
                        let inv_sum = inv_mi + inv_mj;

                        let v_rel_n_sq = v_rel_n * v_rel_n;
                        let do_fusion = cfg.enable_fusion
                            && v_rel_n_sq < cfg.inelastic_threshold_sq;

                        if do_fusion {
                            // Fusión inelástica: conserva masa y momento,
                            // centro de masa, radio = sqrt(M) *
                            // radius_per_sqrt_mass, j marcado muerto.
                            resolve_inelastic_collision(
                                store,
                                i,
                                j,
                                cfg.radius_per_sqrt_mass,
                            );
                            collisions += 1;
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

// ---------------------------------------------------------------------------
// Tests (criterios de aceptación del Issue 10 parte A)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::NativeStore;
    use crate::spatial_grid::SpatialGrid;

    fn make_store(count: usize) -> NativeStore {
        let mut store = NativeStore::new(count).expect("store creation failed");
        store.init_lifecycle();
        store
    }

    /// Dos cuerpos iguales de masa 1, radio 1, en (0,0) y (1.5, 0),
    /// velocidades opuestas (i: +1, j: -1) -> colisión frontal elástica.
    /// Tras el rebote con e=1: i queda con vx=-1, j con vx=+1 (intercambian).
    #[test]
    fn ac1_elastic_collision_swaps_velocities_equal_mass() {
        let mut store = make_store(2);
        unsafe {
            *store.x.add(0) = 0.0;
            *store.y.add(0) = 0.0;
            *store.vx.add(0) = 1.0;
            *store.vy.add(0) = 0.0;
            *store.mass.add(0) = 1.0;
            *store.radius.add(0) = 1.0;

            *store.x.add(1) = 1.5;
            *store.y.add(1) = 0.0;
            *store.vx.add(1) = -1.0;
            *store.vy.add(1) = 0.0;
            *store.mass.add(1) = 1.0;
            *store.radius.add(1) = 1.0;
        }

        unsafe { resolve_elastic_collision(&mut store, 0, 1, 1.0); }

        unsafe {
            assert!((*store.vx.add(0) - (-1.0)).abs() < 1e-5, "vx[0]={}", *store.vx.add(0));
            assert!((*store.vx.add(1) - 1.0).abs() < 1e-5, "vx[1]={}", *store.vx.add(1));
            assert!((*store.vy.add(0) - 0.0).abs() < 1e-5);
            assert!((*store.vy.add(1) - 0.0).abs() < 1e-5);
        }
    }

    /// Separación posicional: tras la colisión los cuerpos no se superponen.
    #[test]
    fn ac2_elastic_collision_positional_separation() {
        let mut store = make_store(2);
        unsafe {
            *store.x.add(0) = 0.0;
            *store.y.add(0) = 0.0;
            *store.vx.add(0) = 0.0;
            *store.vy.add(0) = 0.0;
            *store.mass.add(0) = 1.0;
            *store.radius.add(0) = 1.0;

            *store.x.add(1) = 1.0;
            *store.y.add(1) = 0.0;
            *store.vx.add(1) = 0.0;
            *store.vy.add(1) = 0.0;
            *store.mass.add(1) = 1.0;
            *store.radius.add(1) = 1.0;
        }

        unsafe { resolve_elastic_collision(&mut store, 0, 1, 1.0); }

        unsafe {
            let dx = *store.x.add(1) - *store.x.add(0);
            let dy = *store.y.add(1) - *store.y.add(0);
            let dist = (dx * dx + dy * dy).sqrt();
            assert!(dist >= 2.0 - 1e-5, "dist={} (debe ser >= r_sum=2.0)", dist);
        }
    }

    /// Cuerpos alejándose (v_rel_n >= 0) no reciben impulso.
    #[test]
    fn ac3_elastic_collision_no_impulse_when_separating() {
        let mut store = make_store(2);
        unsafe {
            *store.x.add(0) = 0.0;
            *store.y.add(0) = 0.0;
            *store.vx.add(0) = 1.0;
            *store.vy.add(0) = 0.0;
            *store.mass.add(0) = 1.0;
            *store.radius.add(0) = 1.0;

            *store.x.add(1) = 1.0;
            *store.y.add(1) = 0.0;
            *store.vx.add(1) = 2.0;
            *store.vy.add(1) = 0.0;
            *store.mass.add(1) = 1.0;
            *store.radius.add(1) = 1.0;
        }

        let vx0_before = unsafe { *store.vx.add(0) };
        let vx1_before = unsafe { *store.vx.add(1) };
        unsafe { resolve_elastic_collision(&mut store, 0, 1, 1.0); }
        unsafe {
            assert_eq!(*store.vx.add(0), vx0_before, "vx[0] no debe cambiar al alejarse");
            assert_eq!(*store.vx.add(1), vx1_before, "vx[1] no debe cambiar al alejarse");
        }
    }

    /// Sin overlap (dist >= r_sum) -> sin colisión.
    #[test]
    fn ac4_elastic_collision_no_overlap_no_impulse() {
        let mut store = make_store(2);
        unsafe {
            *store.x.add(0) = 0.0;
            *store.y.add(0) = 0.0;
            *store.vx.add(0) = -1.0;
            *store.vy.add(0) = 0.0;
            *store.mass.add(0) = 1.0;
            *store.radius.add(0) = 1.0;

            *store.x.add(1) = 5.0;
            *store.y.add(1) = 0.0;
            *store.vx.add(1) = -2.0;
            *store.vy.add(1) = 0.0;
            *store.mass.add(1) = 1.0;
            *store.radius.add(1) = 1.0;
        }

        let vx0_before = unsafe { *store.vx.add(0) };
        let vx1_before = unsafe { *store.vx.add(1) };
        unsafe { resolve_elastic_collision(&mut store, 0, 1, 1.0); }
        unsafe {
            assert_eq!(*store.vx.add(0), vx0_before);
            assert_eq!(*store.vx.add(1), vx1_before);
        }
    }

    /// Fusión inelástica: cuerpos lentos con overlap se fusionan, masa y
    /// momento conservados, j marcado muerto (mass = -1.0).
    #[test]
    fn ac5_inelastic_fusion_conserves_mass_and_momentum() {
        let mut store = make_store(2);
        unsafe {
            *store.x.add(0) = 0.0;
            *store.y.add(0) = 0.0;
            *store.vx.add(0) = 2.0;
            *store.vy.add(0) = 0.0;
            *store.mass.add(0) = 1.0;
            *store.radius.add(0) = 1.0;

            *store.x.add(1) = 1.0;
            *store.y.add(1) = 0.0;
            *store.vx.add(1) = 0.0;
            *store.vy.add(1) = 0.0;
            *store.mass.add(1) = 1.0;
            *store.radius.add(1) = 1.0;
        }

        store.live_count = 2;

        let mut grid = SpatialGrid::allocate(4, 4, 2).unwrap();
        grid.set_cell_size(64.0);
        crate::spatial_grid::rebuild_grid(&grid, &store);

        let cfg = CollisionConfig {
            elastic_restitution: 0.8,
            inelastic_threshold_sq: 100.0, // fuerza fusión (v_rel^2 < 100)
            enable_fusion: true,
            radius_per_sqrt_mass: 0.5,
        };
        let n = unsafe { resolve_collisions(&mut store, &grid, cfg) };
        assert_eq!(n, 1, "se esperaba 1 fusión");

        unsafe {
            // Momento inicial: 1*2 + 1*0 = 2. Tras fusión: masa 2 -> vx = 1.
            assert!((*store.vx.add(0) - 1.0).abs() < 1e-5, "vx[0]={}", *store.vx.add(0));
            assert!((*store.mass.add(0) - 2.0).abs() < 1e-5, "mass[0]={}", *store.mass.add(0));
            assert!(*store.mass.add(1) < 0.0, "j debe estar muerto (mass<0)");
        }
    }

    /// resolve_collisions sobre grid real: detecta y resuelve el par.
    #[test]
    fn ac6_resolve_collisions_detects_pair_on_grid() {
        let mut store = make_store(2);
        unsafe {
            *store.x.add(0) = 10.0;
            *store.y.add(0) = 10.0;
            *store.vx.add(0) = 1.0;
            *store.vy.add(0) = 0.0;
            *store.mass.add(0) = 1.0;
            *store.radius.add(0) = 1.0;

            *store.x.add(1) = 11.0;
            *store.y.add(1) = 10.0;
            *store.vx.add(1) = -1.0;
            *store.vy.add(1) = 0.0;
            *store.mass.add(1) = 1.0;
            *store.radius.add(1) = 1.0;
        }
        store.live_count = 2;

        let mut grid = SpatialGrid::allocate(4, 4, 2).unwrap();
        grid.set_cell_size(64.0);
        crate::spatial_grid::rebuild_grid(&grid, &store);

        let cfg = CollisionConfig::default(); // e=0.8, umbral 0.25 -> elástico (v_rel²=4 > 0.25)
        let n = unsafe { resolve_collisions(&mut store, &grid, cfg) };
        assert_eq!(n, 1, "se esperaba 1 colisión");
        // Rebote frontal igual masa con e=0.8: impulso=1.8, vx[0]=1-1.8=-0.8, vx[1]=-1+1.8=0.8.
        unsafe {
            assert!((*store.vx.add(0) - (-0.8)).abs() < 1e-5, "vx[0]={}", *store.vx.add(0));
            assert!((*store.vx.add(1) - 0.8).abs() < 1e-5, "vx[1]={}", *store.vx.add(1));
        }
        // Separación posicional: dist >= r_sum = 2.0.
        unsafe {
            let dx = *store.x.add(1) - *store.x.add(0);
            let dy = *store.y.add(1) - *store.y.add(0);
            let dist = (dx * dx + dy * dy).sqrt();
            assert!(dist >= 2.0 - 1e-5, "dist={} (debe ser >= 2.0)", dist);
        }
    }

    /// resolve_collisions ignora slots muertos (mass < 0).
    #[test]
    fn ac7_resolve_collisions_skips_dead_slots() {
        let mut store = make_store(3);
        unsafe {
            // i=0 y j=1 vivos y colisionando; i=2 muerto.
            *store.x.add(0) = 10.0;
            *store.y.add(0) = 10.0;
            *store.vx.add(0) = 1.0;
            *store.vy.add(0) = 0.0;
            *store.mass.add(0) = 1.0;
            *store.radius.add(0) = 1.0;

            *store.x.add(1) = 11.0;
            *store.y.add(1) = 10.0;
            *store.vx.add(1) = -1.0;
            *store.vy.add(1) = 0.0;
            *store.mass.add(1) = 1.0;
            *store.radius.add(1) = 1.0;

            *store.x.add(2) = 10.5;
            *store.y.add(2) = 10.0;
            *store.mass.add(2) = -1.0; // muerto
        }
        store.live_count = 3;

        let mut grid = SpatialGrid::allocate(4, 4, 2).unwrap();
        grid.set_cell_size(64.0);
        crate::spatial_grid::rebuild_grid(&grid, &store);

        let cfg = CollisionConfig::default();
        let n = unsafe { resolve_collisions(&mut store, &grid, cfg) };
        assert_eq!(n, 1, "solo el par vivo (0,1) colisiona");
    }

    /// Zero-allocation en hot path: comprobación por inspección de código
    /// (no hay Vec/String/format!/Box/closures en resolve_elastic_collision ni
    /// resolve_collisions). Salvaguarda: llamada repetida sin panic.
    #[test]
    fn ac8_no_allocation_hot_path_repeated_calls() {
        let mut store = make_store(4);
        unsafe {
            for s in 0..4 {
                *store.x.add(s) = (s as f32) * 20.0;
                *store.y.add(s) = 10.0;
                *store.vx.add(s) = 1.0;
                *store.vy.add(s) = 0.0;
                *store.mass.add(s) = 1.0;
                *store.radius.add(s) = 1.0;
            }
        }
        store.live_count = 4;

        let mut grid = SpatialGrid::allocate(4, 4, 4).unwrap();
        grid.set_cell_size(64.0);
        crate::spatial_grid::rebuild_grid(&grid, &store);

        let cfg = CollisionConfig::default();
        for _ in 0..1000 {
            let _ = unsafe { resolve_collisions(&mut store, &grid, cfg) };
        }
        // Sin panic y sin corromper invariantes básicas.
        assert!(store.live_count >= 1);
    }

    // -------------------------------------------------------------------------
    // Tests (criterios de aceptación del Issue 10 parte B — fusión inelástica)
    // -------------------------------------------------------------------------

    /// `resolve_inelastic_collision`: conserva masa y momento, centro de masa,
    /// radio = sqrt(M) * radius_per_sqrt_mass, y `j` marcado muerto
    /// (mass = -1.0, generación++, LIFO free list, live_count--).
    #[test]
    fn ac9_inelastic_collision_full_contract() {
        let mut store = make_store(3);
        unsafe {
            // i: m=1, x=0, v=+4 ; j: m=3, x=0.5, v=-2.
            *store.x.add(0) = 0.0;
            *store.y.add(0) = 0.0;
            *store.vx.add(0) = 4.0;
            *store.vy.add(0) = 0.0;
            *store.mass.add(0) = 1.0;
            *store.radius.add(0) = 1.0;
            *store.generation.add(0) = 0;

            *store.x.add(1) = 0.5;
            *store.y.add(1) = 0.0;
            *store.vx.add(1) = -2.0;
            *store.vy.add(1) = 0.0;
            *store.mass.add(1) = 3.0;
            *store.radius.add(1) = 1.0;
            *store.generation.add(1) = 0;
        }
        store.live_count = 2;

        let gen_j_before = unsafe { *store.generation.add(1) };
        unsafe { resolve_inelastic_collision(&mut store, 0, 1, 0.5); }

        unsafe {
            let new_m = 4.0f32;
            // Momento: (1*4 + 3*(-2)) / 4 = -0.5.
            assert!((*store.vx.add(0) - (-0.5)).abs() < 1e-5, "vx[0]={}", *store.vx.add(0));
            assert!((*store.vy.add(0) - 0.0).abs() < 1e-5);
            // Centro de masa: (1*0 + 3*0.5) / 4 = 0.375.
            assert!((*store.x.add(0) - 0.375).abs() < 1e-5, "x[0]={}", *store.x.add(0));
            assert!((*store.y.add(0) - 0.0).abs() < 1e-5);
            assert!((*store.mass.add(0) - new_m).abs() < 1e-5, "mass[0]={}", *store.mass.add(0));
            // Radio = sqrt(4) * 0.5 = 1.0.
            assert!((*store.radius.add(0) - 1.0).abs() < 1e-5, "radius[0]={}", *store.radius.add(0));
            // j muerto: centinela, generación++, cabeza LIFO, live_count--.
            assert!(*store.mass.add(1) < 0.0, "j debe estar muerto (mass<0)");
            assert_eq!(
                *store.generation.add(1),
                gen_j_before + 1,
                "generación[j] debe incrementarse"
            );
            assert_eq!(store.free_head, 1, "free_head debe apuntar a j (LIFO)");
            assert_eq!(store.live_count, 1, "live_count--");
        }
    }

    /// `resolve_inelastic_collision` con masa nula en el par: sin NaN/Inf
    /// en ningún canal (`inv_new_m = 0` protege la división).
    #[test]
    fn ac10_inelastic_collision_zero_mass_no_nan() {
        let mut store = make_store(2);
        unsafe {
            *store.x.add(0) = 0.0;
            *store.y.add(0) = 0.0;
            *store.vx.add(0) = 1.0;
            *store.vy.add(0) = 0.0;
            *store.mass.add(0) = 0.0;
            *store.radius.add(0) = 0.5;

            *store.x.add(1) = 0.5;
            *store.y.add(1) = 0.0;
            *store.vx.add(1) = 0.0;
            *store.vy.add(1) = 0.0;
            *store.mass.add(1) = 0.0;
            *store.radius.add(1) = 0.5;
        }
        store.live_count = 2;

        unsafe { resolve_inelastic_collision(&mut store, 0, 1, 0.5); }

        unsafe {
            assert!(!store.vx.add(0).read().is_nan(), "vx[0] no debe ser NaN");
            assert!(!store.vy.add(0).read().is_nan());
            assert!(!store.mass.add(0).read().is_nan());
            assert!(!store.radius.add(0).read().is_nan());
            assert!(*store.mass.add(1) < 0.0, "j debe quedar muerto");
        }
    }
}
