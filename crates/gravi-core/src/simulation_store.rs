//! simulation_store.rs — Parte B: clear_forces, compute_forces_brute,
//! compute_forces_brute_range, integrate_symplectic_euler.
//! (Issue 5: Kerneles de fuerza vectorizados.)
//! (Issue 3: SimulationStore — allocate/dispose/seed.)

use crate::constants::PhysicsConstants;
use crate::store::NativeStore;

/// Configuración de creación de un store de simulación.
#[derive(Debug, Clone, Copy)]
pub struct SimConfig {
    pub body_count: usize,
    pub width: f32,
    pub height: f32,
}

/// Crea y asigna un `NativeStore` con `body_count` cuerpos.
///
/// Valida `body_count > 0` y `width > 0`, `height > 0`.
/// Tras allocate: `live_count == count`, `free_head == -1`, todas las
/// generaciones == 0.
pub fn allocate(config: SimConfig) -> Result<NativeStore, String> {
    if config.body_count == 0 {
        return Err("body_count must be > 0".to_string());
    }
    if config.width <= 0.0 {
        return Err("width must be > 0".to_string());
    }
    if config.height <= 0.0 {
        return Err("height must be > 0".to_string());
    }
    NativeStore::new(config.body_count)
}

/// Libera los recursos de un store. Idempotente: tras llamar, `live_count` y
/// `free_head` quedan invalidados; llamar dos veces no causa doble liberación
/// (Drop es el encargado único de liberar la memoria nativa).
pub fn dispose(store: &mut NativeStore) {
    if store.count == 0 {
        return;
    }
    // Invalidar lógicamente los contadores para que un segundo dispose sea
    // un no-op detectable. Drop seguirá liberando la memoria una sola vez.
    let chan = store.count * std::mem::size_of::<f32>();
    unsafe {
        std::ptr::write_bytes(store.x, 0, chan);
        std::ptr::write_bytes(store.y, 0, chan);
        std::ptr::write_bytes(store.vx, 0, chan);
        std::ptr::write_bytes(store.vy, 0, chan);
        std::ptr::write_bytes(
            store.generation,
            0,
            store.count * std::mem::size_of::<i32>(),
        );
    }
    store.live_count = 0;
    store.free_head = -1;
}

/// RNG determinista module-private (SplitMix64), sin dependencias externas.
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u32) -> Self {
        Self((seed as u64).wrapping_add(0x9E37_79B9_7F4A_7C15))
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// `f32` en [0, 1) a partir de los 24 bits altos de la palabra de 64 bits.
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / ((1u32 << 24) as f32)
    }
}

const TAU: f32 = 2.0 * std::f32::consts::PI;

/// Siembra los cuerpos con X/Y uniformes en [0, width) x [0, height) vía RNG,
/// ángulo = next_f32()*2π y velocidad de magnitud constante `speed`.
/// Mass=1.0, Radius=0.5; tail (live_count=count, free_head=-1, generaciones=0);
/// finaliza con `clear_forces(store)`.
pub fn seed_linear(store: &mut NativeStore, width: f32, height: f32, speed: f32, seed: u32) {
    let mut rng = SplitMix64::new(seed);
    let count = store.count;
    let c = store.channels_mut();
    for i in 0..count {
        c.x[i] = rng.next_f32() * width;
        c.y[i] = rng.next_f32() * height;
        let angle = rng.next_f32() * TAU;
        c.vx[i] = angle.cos() * speed;
        c.vy[i] = angle.sin() * speed;
        c.mass[i] = 1.0;
        c.radius[i] = 0.5;
    }
    for g in c.generation.iter_mut() {
        *g = 0;
    }
    drop(c);
    store.live_count = count;
    store.free_head = -1;
    clear_forces(store);
}

/// Siembra una nube aleatoria centrada (aproximación CLT, m=1) con velocidades
/// térmicas. X/Y clampeadas a [0, width-1] / [0, height-1]. Zero-alloc.
pub fn seed_random_cloud(
    store: &mut NativeStore,
    width: f32,
    height: f32,
    spread: f32,
    thermal_speed: f32,
    seed: u32,
) {
    let mut rng = SplitMix64::new(seed);
    let count = store.count;
    let cx = width * 0.5;
    let cy = height * 0.5;
    let x_max = width - 1.0;
    let y_max = height - 1.0;
    let c = store.channels_mut();
    for i in 0..count {
        let normal = rng.next_f32() + rng.next_f32() + rng.next_f32() - 1.5;
        let r = normal * spread;
        let theta = rng.next_f32() * TAU;
        let mut xi = cx + r * theta.cos();
        if xi < 0.0 {
            xi = 0.0;
        } else if xi >= width {
            xi = x_max;
        }
        let mut yi = cy + r * theta.sin();
        if yi < 0.0 {
            yi = 0.0;
        } else if yi >= height {
            yi = y_max;
        }
        c.x[i] = xi;
        c.y[i] = yi;
        c.vx[i] = thermal_speed * (rng.next_f32() - 0.5);
        c.vy[i] = thermal_speed * (rng.next_f32() - 0.5);
        c.mass[i] = 1.0;
        c.radius[i] = 0.5;
    }
    for g in c.generation.iter_mut() {
        *g = 0;
    }
    drop(c);
    store.live_count = count;
    store.free_head = -1;
    clear_forces(store);
}

/// Siembra un campo uniforme aleatorio sobre toda la pantalla con velocidades
/// isótropas: módulo sqrt(u)*max_speed y dirección uniforme en [0, 2π). Zero-alloc.
pub fn seed_uniform_field(store: &mut NativeStore, width: f32, height: f32, max_speed: f32, seed: u32) {
    let mut rng = SplitMix64::new(seed);
    let count = store.count;
    let c = store.channels_mut();
    for i in 0..count {
        c.x[i] = rng.next_f32() * width;
        c.y[i] = rng.next_f32() * height;
        let u = rng.next_f32();
        let speed_mag = u.sqrt() * max_speed;
        let angle = rng.next_f32() * TAU;
        c.vx[i] = speed_mag * angle.cos();
        c.vy[i] = speed_mag * angle.sin();
        c.mass[i] = 1.0;
        c.radius[i] = 0.5;
    }
    for g in c.generation.iter_mut() {
        *g = 0;
    }
    drop(c);
    store.live_count = count;
    store.free_head = -1;
    clear_forces(store);
}

/// Siembra un disco orbital con velocidad tangencial kepleriana alrededor de
/// una masa central. `inner_radius` se clampa a >= 1.0. Zero-alloc.
pub fn seed_orbital_disk(
    store: &mut NativeStore,
    width: f32,
    height: f32,
    inner_radius: f32,
    outer_radius: f32,
    central_mass: f32,
    softening: f32,
    seed: u32,
) {
    let mut rng = SplitMix64::new(seed);
    let count = store.count;
    let cx = width * 0.5;
    let cy = height * 0.5;
    let g = PhysicsConstants::default().g;
    let inner = inner_radius.max(1.0);
    let soft_sq = softening * softening;
    let c = store.channels_mut();
    for i in 0..count {
        let r = inner + rng.next_f32() * (outer_radius - inner);
        let theta = rng.next_f32() * TAU;
        c.x[i] = cx + r * theta.cos();
        c.y[i] = cy + r * theta.sin();
        let r_sq_soft = r * r + soft_sq;
        let v = r * (g * central_mass / (r_sq_soft * r_sq_soft.sqrt())).sqrt();
        c.vx[i] = -v * theta.sin();
        c.vy[i] = v * theta.cos();
        c.mass[i] = 1.0;
        c.radius[i] = 0.5;
    }
    for g in c.generation.iter_mut() {
        *g = 0;
    }
    drop(c);
    store.live_count = count;
    store.free_head = -1;
    clear_forces(store);
}

/// Clampa X/Y a los límites [0, width] / [0, height] e invierte la velocidad
/// (Vx/Vy) del cuerpo que toca un borde.
pub fn reflect_bounds(store: &mut NativeStore, width: f32, height: f32) {
    let count = store.count;
    let c = store.channels_mut();
    for i in 0..count {
        if c.mass[i] < 0.0 {
            continue;
        }
        if c.x[i] < 0.0 {
            c.x[i] = 0.0;
            c.vx[i] = -c.vx[i];
        } else if c.x[i] > width {
            c.x[i] = width;
            c.vx[i] = -c.vx[i];
        }
        if c.y[i] < 0.0 {
            c.y[i] = 0.0;
            c.vy[i] = -c.vy[i];
        } else if c.y[i] > height {
            c.y[i] = height;
            c.vy[i] = -c.vy[i];
        }
    }
}

// ============================================================================
// Parte B — Kerneles de fuerza (Issue 5)
// ============================================================================

use std::simd::f32x8;
use std::simd::prelude::SimdFloat;
use std::simd::StdFloat;

/// Umbral de suavizado al cuadrado (evita división flotante en el hot path).
use crate::constants::SOFTENING_SQ;

/// `clear_forces` — pone a cero Fx/Fy de todos los slots vivos.
#[inline(always)]
pub fn clear_forces(store: &mut NativeStore) {
    unsafe {
        let fx = store.fx;
        let fy = store.fy;
        let count = store.count;
        let mut i = 0usize;
        // Unrolling 2x.
        while i + 1 < count {
            *fx.add(i) = 0.0f32;
            *fy.add(i) = 0.0f32;
            *fx.add(i + 1) = 0.0f32;
            *fy.add(i + 1) = 0.0f32;
            i += 2;
        }
        if i < count {
            *fx.add(i) = 0.0f32;
            *fy.add(i) = 0.0f32;
        }
    }
}

/// `compute_forces_brute_range` — fuerza bruta sobre el rango `[start, end)` de
/// objetivos contra todos los cuerpos `0..count`. Usa f32x8 con unrolling 2x.
/// Cuando `live_count != count` (hay slots muertos, masa < 0) cae al camino
/// escalar O(N²) (`compute_forces_brute_range_scalar`), que salta tanto
/// objetivos como fuentes con `mass < 0`.
///
/// Devuelve (sum_fx, sum_fy) acumulados por objetivo (uno por slot en el rango).
#[inline(always)]
pub fn compute_forces_brute_range(
    store: &NativeStore,
    start: usize,
    end: usize,
    count: usize,
    g: f32,
) {
    if store.live_count != store.count {
        compute_forces_brute_range_scalar(store, start, end, count, g);
        return;
    }
    unsafe {
        let x = store.x;
        let y = store.y;
        let mass = store.mass;
        let fx = store.fx;
        let fy = store.fy;
        let g8 = f32x8::splat(g);
        let soft8 = f32x8::splat(SOFTENING_SQ);
        let mut i = start;
        while i < end {
            let xi = *x.add(i);
            let yi = *y.add(i);
            let mut accx = 0.0f32;
            let mut accy = 0.0f32;
            let mut j = 0usize;
            // Unrolling 2x: procesa 16 cuerpos por iteración externa.
            while j + 15 < count {
                // Bloque 0: j..j+8 (carga directa desde puntero crudo, sin slice temporal)
                let mx0 = f32x8::from_array(std::ptr::read_unaligned(mass.add(j).cast::<[f32; 8]>()) );
                let dx0 = f32x8::from_array(std::ptr::read_unaligned(x.add(j).cast::<[f32; 8]>()) ) - f32x8::splat(xi);
                let dy0 = f32x8::from_array(std::ptr::read_unaligned(y.add(j).cast::<[f32; 8]>()) ) - f32x8::splat(yi);
                let r2_0 = dx0 * dx0 + dy0 * dy0 + soft8;
                let invr_0 = r2_0.recip().sqrt(); // 1/sqrt(d^2 + soft^2)
                let invr3_0 = invr_0 * invr_0 * invr_0;
                let f0 = g8 * mx0 * invr3_0;
                accx += (f0 * dx0).reduce_sum();
                accy += (f0 * dy0).reduce_sum();

                // Bloque 1: j+8..j+16 (carga directa desde puntero crudo, sin slice temporal)
                let mx1 = f32x8::from_array(std::ptr::read_unaligned(mass.add(j + 8).cast::<[f32; 8]>()) );
                let dx1 = f32x8::from_array(std::ptr::read_unaligned(x.add(j + 8).cast::<[f32; 8]>()) ) - f32x8::splat(xi);
                let dy1 = f32x8::from_array(std::ptr::read_unaligned(y.add(j + 8).cast::<[f32; 8]>()) ) - f32x8::splat(yi);
                let r2_1 = dx1 * dx1 + dy1 * dy1 + soft8;
                let invr_1 = r2_1.recip().sqrt(); // 1/sqrt(d^2 + soft^2)
                let invr3_1 = invr_1 * invr_1 * invr_1;
                let f1 = g8 * mx1 * invr3_1;
                accx += (f1 * dx1).reduce_sum();
                accy += (f1 * dy1).reduce_sum();

                j += 16;
            }
            // Resto scalar (j..count).
            while j < count {
                let dx = *x.add(j) - xi;
                let dy = *y.add(j) - yi;
                let r2 = dx * dx + dy * dy + SOFTENING_SQ;
                let invr = 1.0f32 / r2.sqrt(); // scalar: usa sqrt (una división implícita no, pero OK por ser resto)
                let invr3 = invr * invr * invr;
                let f = g * *mass.add(j) * invr3;
                accx += f * dx;
                accy += f * dy;
                j += 1;
            }
            *fx.add(i) += accx;
            *fy.add(i) += accy;
            i += 1;
        }
    }
}

/// `compute_forces_brute_range` fallback scalar (live_count != count).
#[inline(always)]
fn compute_forces_brute_range_scalar(
    store: &NativeStore,
    start: usize,
    end: usize,
    count: usize,
    g: f32,
) {
    unsafe {
        let x = store.x;
        let y = store.y;
        let mass = store.mass;
        let fx = store.fx;
        let fy = store.fy;
        let mut i = start;
        while i < end {
            // Objetivo muerto: no escribir fuerza.
            if *mass.add(i) < 0.0f32 {
                i += 1;
                continue;
            }
            let xi = *x.add(i);
            let yi = *y.add(i);
            let mut accx = 0.0f32;
            let mut accy = 0.0f32;
            let mut j = 0usize;
            while j < count {
                // Fuente muerta: no aporta contribución.
                if *mass.add(j) < 0.0f32 {
                    j += 1;
                    continue;
                }
                let dx = *x.add(j) - xi;
                let dy = *y.add(j) - yi;
                let r2 = dx * dx + dy * dy + SOFTENING_SQ;
                let invr = 1.0f32 / r2.sqrt();
                let invr3 = invr * invr * invr;
                let f = g * *mass.add(j) * invr3;
                accx += f * dx;
                accy += f * dy;
                j += 1;
            }
            *fx.add(i) += accx;
            *fy.add(i) += accy;
            i += 1;
        }
    }
}

/// `compute_forces_brute` — fuerza bruta sobre todos los cuerpos vivos.
#[inline(always)]
pub fn compute_forces_brute(store: &NativeStore, g: f32) {
    compute_forces_brute_range(store, 0, store.count, store.count, g);
}

/// `integrate_symplectic_euler` — integra posiciones y velocidades (Euler
/// semisimpléctico) para todos los slots vivos.
#[inline(always)]
pub fn integrate_symplectic_euler(store: &mut NativeStore, dt: f32, damping: f32) {
    unsafe {
        let x = store.x;
        let y = store.y;
        let vx = store.vx;
        let vy = store.vy;
        let fx = store.fx;
        let fy = store.fy;
        let mass = store.mass;
        let live_indices = store.live_indices;
        let live_count = store.live_count;
        let dt8 = f32x8::splat(dt);
        let damping8 = f32x8::splat(damping);

        // Iterar solo sobre slots vivos: live_indices[0..live_count].
        let mut q = 0usize;
        while q + 7 < live_count {
            // Bloque de 8: solo SIMD si todos los 8 son vivos (masa >= 0).
            let idx0 = *live_indices.add(q) as usize;
            let idx1 = *live_indices.add(q + 1) as usize;
            let idx2 = *live_indices.add(q + 2) as usize;
            let idx3 = *live_indices.add(q + 3) as usize;
            let idx4 = *live_indices.add(q + 4) as usize;
            let idx5 = *live_indices.add(q + 5) as usize;
            let idx6 = *live_indices.add(q + 6) as usize;
            let idx7 = *live_indices.add(q + 7) as usize;

            let all_live =
                *mass.add(idx0) >= 0.0f32
                    && *mass.add(idx1) >= 0.0f32
                    && *mass.add(idx2) >= 0.0f32
                    && *mass.add(idx3) >= 0.0f32
                    && *mass.add(idx4) >= 0.0f32
                    && *mass.add(idx5) >= 0.0f32
                    && *mass.add(idx6) >= 0.0f32
                    && *mass.add(idx7) >= 0.0f32;

            if all_live {
                let fx8 = f32x8::from_array([
                    *fx.add(idx0), *fx.add(idx1), *fx.add(idx2), *fx.add(idx3),
                    *fx.add(idx4), *fx.add(idx5), *fx.add(idx6), *fx.add(idx7),
                ]);
                let fy8 = f32x8::from_array([
                    *fy.add(idx0), *fy.add(idx1), *fy.add(idx2), *fy.add(idx3),
                    *fy.add(idx4), *fy.add(idx5), *fy.add(idx6), *fy.add(idx7),
                ]);
                let m8 = f32x8::from_array([
                    *mass.add(idx0), *mass.add(idx1), *mass.add(idx2), *mass.add(idx3),
                    *mass.add(idx4), *mass.add(idx5), *mass.add(idx6), *mass.add(idx7),
                ]);
                let vx8 = f32x8::from_array([
                    *vx.add(idx0), *vx.add(idx1), *vx.add(idx2), *vx.add(idx3),
                    *vx.add(idx4), *vx.add(idx5), *vx.add(idx6), *vx.add(idx7),
                ]);
                let vy8 = f32x8::from_array([
                    *vy.add(idx0), *vy.add(idx1), *vy.add(idx2), *vy.add(idx3),
                    *vy.add(idx4), *vy.add(idx5), *vy.add(idx6), *vy.add(idx7),
                ]);
                let x8 = f32x8::from_array([
                    *x.add(idx0), *x.add(idx1), *x.add(idx2), *x.add(idx3),
                    *x.add(idx4), *x.add(idx5), *x.add(idx6), *x.add(idx7),
                ]);
                let y8 = f32x8::from_array([
                    *y.add(idx0), *y.add(idx1), *y.add(idx2), *y.add(idx3),
                    *y.add(idx4), *y.add(idx5), *y.add(idx6), *y.add(idx7),
                ]);

                let inv_m = f32x8::splat(1.0f32) / m8;
                let ax = fx8 * inv_m;
                let ay = fy8 * inv_m;

                let nvx = (vx8 + ax * dt8) * damping8;
                let nvy = (vy8 + ay * dt8) * damping8;

                let nx = x8 + nvx * dt8;
                let ny = y8 + nvy * dt8;
                let nx_a = nx.to_array();
                let ny_a = ny.to_array();
                let nvx_a = nvx.to_array();
                let nvy_a = nvy.to_array();

                *x.add(idx0) = nx_a[0];
                *x.add(idx1) = nx_a[1];
                *x.add(idx2) = nx_a[2];
                *x.add(idx3) = nx_a[3];
                *x.add(idx4) = nx_a[4];
                *x.add(idx5) = nx_a[5];
                *x.add(idx6) = nx_a[6];
                *x.add(idx7) = nx_a[7];
                *y.add(idx0) = ny_a[0];
                *y.add(idx1) = ny_a[1];
                *y.add(idx2) = ny_a[2];
                *y.add(idx3) = ny_a[3];
                *y.add(idx4) = ny_a[4];
                *y.add(idx5) = ny_a[5];
                *y.add(idx6) = ny_a[6];
                *y.add(idx7) = ny_a[7];
                *vx.add(idx0) = nvx_a[0];
                *vx.add(idx1) = nvx_a[1];
                *vx.add(idx2) = nvx_a[2];
                *vx.add(idx3) = nvx_a[3];
                *vx.add(idx4) = nvx_a[4];
                *vx.add(idx5) = nvx_a[5];
                *vx.add(idx6) = nvx_a[6];
                *vx.add(idx7) = nvx_a[7];
                *vy.add(idx0) = nvy_a[0];
                *vy.add(idx1) = nvy_a[1];
                *vy.add(idx2) = nvy_a[2];
                *vy.add(idx3) = nvy_a[3];
                *vy.add(idx4) = nvy_a[4];
                *vy.add(idx5) = nvy_a[5];
                *vy.add(idx6) = nvy_a[6];
                *vy.add(idx7) = nvy_a[7];
            } else {
                // Fallback scalar por slot vivo en el bloque.
                for k in 0..8usize {
                    let idx = *live_indices.add(q + k) as usize;
                    let m = *mass.add(idx);
                    if m < 0.0f32 {
                        continue;
                    }
                    let inv_m = 1.0f32 / m;
                    let ax = *fx.add(idx) * inv_m;
                    let ay = *fy.add(idx) * inv_m;
                    let nvx = (*vx.add(idx) + ax * dt) * damping;
                    let nvy = (*vy.add(idx) + ay * dt) * damping;
                    *x.add(idx) += nvx * dt;
                    *y.add(idx) += nvy * dt;
                    *vx.add(idx) = nvx;
                    *vy.add(idx) = nvy;
                }
            }

            q += 8;
        }
        // Resto scalar por slot vivo.
        while q < live_count {
            let idx = *live_indices.add(q) as usize;
            let m = *mass.add(idx);
            if m < 0.0f32 {
                q += 1;
                continue;
            }
            let inv_m = 1.0f32 / m;
            let ax = *fx.add(idx) * inv_m;
            let ay = *fy.add(idx) * inv_m;
            let nvx = (*vx.add(idx) + ax * dt) * damping;
            let nvy = (*vy.add(idx) + ay * dt) * damping;
            *x.add(idx) += nvx * dt;
            *y.add(idx) += nvy * dt;
            *vx.add(idx) = nvx;
            *vy.add(idx) = nvy;
            q += 1;
        }
    }
}


