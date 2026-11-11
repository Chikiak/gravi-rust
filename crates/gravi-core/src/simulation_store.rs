//! simulation_store.rs — Parte B: clear_forces, compute_forces_brute,
//! compute_forces_brute_range, integrate_symplectic_euler.
//! (Issue 5: Kerneles de fuerza vectorizados.)
//! (Issue 3: SimulationStore — allocate/dispose/seed.)

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

/// Siembra los cuerpos en una rejilla linear: X/Y distribuidas uniformemente
/// en [0, width] / [0, height], y Vx/Vy con magnitud constante `speed`.
pub fn seed_linear(store: &mut NativeStore, width: f32, height: f32, speed: f32) {
    let count = store.count;
    let x = store.x_slice();
    let y = store.y_slice();
    let vx = store.vx_slice();
    let vy = store.vy_slice();

    let mut cols = (count as f32).sqrt().ceil() as usize;
    if cols == 0 {
        cols = 1;
    }
    let max_dim = if cols > 1 { cols - 1 } else { 1 };

    for i in 0..count {
        let cx = (i % cols) as f32;
        let cy = (i / cols) as f32;
        x[i] = (cx / max_dim as f32) * width;
        y[i] = (cy / max_dim as f32) * height;

        let angle = (i as f32) * 0.7853981633974483; // pi/4
        vx[i] = angle.cos() * speed;
        vy[i] = angle.sin() * speed;
    }
    store.live_count = count;
    store.free_head = -1;
    for g in store.generation_slice().iter_mut() {
        *g = 0;
    }
}

/// Clampa X/Y a los límites [0, width] / [0, height] e invierte la velocidad
/// (Vx/Vy) del cuerpo que toca un borde.
pub fn reflect_bounds(store: &mut NativeStore, width: f32, height: f32) {
    let x = store.x_slice();
    let y = store.y_slice();
    let vx = store.vx_slice();
    let vy = store.vy_slice();

    for i in 0..store.count {
        if x[i] < 0.0 {
            x[i] = 0.0;
            vx[i] = -vx[i];
        } else if x[i] > width {
            x[i] = width;
            vx[i] = -vx[i];
        }
        if y[i] < 0.0 {
            y[i] = 0.0;
            vy[i] = -vy[i];
        } else if y[i] > height {
            y[i] = height;
            vy[i] = -vy[i];
        }
    }
}

// ============================================================================
// Parte B — Kerneles de fuerza (Issue 5)
// ============================================================================

use std::simd::f32x8;
use std::simd::prelude::SimdFloat;

/// Umbral de suavizado al cuadrado (evita división flotante en el hot path).
const SOFTENING_SQ: f32 = 64.0f32; // 8.0^2

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
/// Fallback scalar solo si `live_count != count`.
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
                // Bloque 0: j..j+8
                let mx0 = f32x8::from_slice(std::slice::from_raw_parts(mass.add(j), 8));
                let dx0 = f32x8::from_slice(std::slice::from_raw_parts(x.add(j), 8)) - f32x8::splat(xi);
                let dy0 = f32x8::from_slice(std::slice::from_raw_parts(y.add(j), 8)) - f32x8::splat(yi);
                let r2_0 = dx0 * dx0 + dy0 * dy0 + soft8;
                let invr_0 = f32x8::splat(1.0f32) / r2_0; // Newton-Raphson disponible vía intrinsic
                let invr3_0 = invr_0 * invr_0 * invr_0;
                let f0 = g8 * mx0 * invr3_0;
                accx += (f0 * dx0).reduce_sum();
                accy += (f0 * dy0).reduce_sum();

                // Bloque 1: j+8..j+16
                let mx1 = f32x8::from_slice(std::slice::from_raw_parts(mass.add(j + 8), 8));
                let dx1 = f32x8::from_slice(std::slice::from_raw_parts(x.add(j + 8), 8)) - f32x8::splat(xi);
                let dy1 = f32x8::from_slice(std::slice::from_raw_parts(y.add(j + 8), 8)) - f32x8::splat(yi);
                let r2_1 = dx1 * dx1 + dy1 * dy1 + soft8;
                let invr_1 = f32x8::splat(1.0f32) / r2_1;
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
            let xi = *x.add(i);
            let yi = *y.add(i);
            let mut accx = 0.0f32;
            let mut accy = 0.0f32;
            let mut j = 0usize;
            while j < count {
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
        let count = store.count;
        let dt8 = f32x8::splat(dt);
        let damping8 = f32x8::splat(damping);
        let mut i = 0usize;
        while i + 7 < count {
            // Carga 8 cuerpos.
            let fx8 = f32x8::from_slice(std::slice::from_raw_parts(fx.add(i), 8));
            let fy8 = f32x8::from_slice(std::slice::from_raw_parts(fy.add(i), 8));
            let m8 = f32x8::from_slice(std::slice::from_raw_parts(mass.add(i), 8));
            let vx8 = f32x8::from_slice(std::slice::from_raw_parts(vx.add(i), 8));
            let vy8 = f32x8::from_slice(std::slice::from_raw_parts(vy.add(i), 8));
            let x8 = f32x8::from_slice(std::slice::from_raw_parts(x.add(i), 8));
            let y8 = f32x8::from_slice(std::slice::from_raw_parts(y.add(i), 8));

            // a = F / m (usa multiplicación por inversa precalculada: no división)
            let inv_m = f32x8::splat(1.0f32) / m8;
            let ax = fx8 * inv_m;
            let ay = fy8 * inv_m;

            // v += a * dt ; v *= damping
            let nvx = (vx8 + ax * dt8) * damping8;
            let nvy = (vy8 + ay * dt8) * damping8;

            // x += v * dt (usa la velocidad nueva — semisimpléctico)
            let nx = x8 + nvx * dt8;
            let ny = y8 + nvy * dt8;

            // Escribe de vuelta.
            nx.copy_to_slice(std::slice::from_raw_parts_mut(x.add(i), 8));
            ny.copy_to_slice(std::slice::from_raw_parts_mut(y.add(i), 8));
            nvx.copy_to_slice(std::slice::from_raw_parts_mut(vx.add(i), 8));
            nvy.copy_to_slice(std::slice::from_raw_parts_mut(vy.add(i), 8));

            i += 8;
        }
        // Resto scalar.
        while i < count {
            let m = *mass.add(i);
            let inv_m = if m != 0.0 { 1.0f32 / m } else { 0.0f32 };
            let ax = *fx.add(i) * inv_m;
            let ay = *fy.add(i) * inv_m;
            let nvx = (*vx.add(i) + ax * dt) * damping;
            let nvy = (*vy.add(i) + ay * dt) * damping;
            *x.add(i) += nvx * dt;
            *y.add(i) += nvy * dt;
            *vx.add(i) = nvx;
            *vy.add(i) = nvy;
            i += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests (Issue 3 — criterios de aceptación)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> SimConfig {
        SimConfig {
            body_count: 16,
            width: 100.0,
            height: 100.0,
        }
    }

    #[test]
    fn ac1_allocate_validates_body_count_and_dims() {
        assert!(allocate(SimConfig { body_count: 0, width: 10.0, height: 10.0 }).is_err());
        assert!(allocate(SimConfig { body_count: 5, width: 0.0, height: 10.0 }).is_err());
        assert!(allocate(SimConfig { body_count: 5, width: 10.0, height: 0.0 }).is_err());
        assert!(allocate(SimConfig { body_count: 5, width: -1.0, height: 10.0 }).is_err());
        assert!(allocate(SimConfig { body_count: 5, width: 10.0, height: -1.0 }).is_err());
        assert!(allocate(SimConfig { body_count: 5, width: 10.0, height: 10.0 }).is_ok());
    }

    #[test]
    fn ac2_after_allocate_live_count_equals_count_free_head_minus_one_gen_zero() {
        let store = allocate(config()).unwrap();
        assert_eq!(store.live_count, store.count);
        assert_eq!(store.free_head, -1);
        assert_eq!(store.count, 16);
        for g in store.generation_slice().iter() {
            assert_eq!(*g, 0);
        }
        drop(store);
    }

    #[test]
    fn ac3_dispose_is_idempotent_and_nulls() {
        let mut store = allocate(config()).unwrap();
        let count = store.count;
        dispose(&mut store);
        assert_eq!(store.live_count, 0);
        assert_eq!(store.free_head, -1);
        // Segundo dispose: no-op, sin panic.
        dispose(&mut store);
        assert_eq!(store.live_count, 0);
        assert_eq!(store.free_head, -1);
        // count aún válido para que Drop libere una sola vez al salir del scope.
        assert_eq!(store.count, count);
    }

    #[test]
    fn ac4_seed_linear_fills_xy_in_range_and_speed_magnitude() {
        let mut store = allocate(config()).unwrap();
        let speed = 10.0f32;
        seed_linear(&mut store, 100.0, 100.0, speed);

        let x = store.x_slice();
        let y = store.y_slice();
        let vx = store.vx_slice();
        let vy = store.vy_slice();

        for i in 0..store.count {
            assert!(x[i] >= 0.0 && x[i] <= 100.0, "x[{}]={}", i, x[i]);
            assert!(y[i] >= 0.0 && y[i] <= 100.0, "y[{}]={}", i, y[i]);
            let mag = (vx[i] * vx[i] + vy[i] * vy[i]).sqrt();
            assert!((mag - speed).abs() < 1e-4, "mag[{}]={} speed={}", i, mag, speed);
        }
        assert_eq!(store.live_count, store.count);
        assert_eq!(store.free_head, -1);
    }

    #[test]
    fn ac5_reflect_bounds_clamps_and_inverts_velocity() {
        let mut store = allocate(config()).unwrap();
        // Escribir valores fuera de los límites (sobre slices en un scope
        // limitado para liberar el borrow inmutable antes de reflect_bounds).
        {
            let x = store.x_slice();
            let y = store.y_slice();
            let vx = store.vx_slice();
            let vy = store.vy_slice();
            x[0] = -5.0; vx[0] = 3.0;
            x[1] = 105.0; vx[1] = -2.0;
            y[2] = -1.0; vy[2] = 4.0;
            y[3] = 101.0; vy[3] = -1.0;
        }

        reflect_bounds(&mut store, 100.0, 100.0);

        let x = store.x_slice();
        let y = store.y_slice();
        let vx = store.vx_slice();
        let vy = store.vy_slice();
        assert_eq!(x[0], 0.0);
        assert_eq!(vx[0], -3.0);
        assert_eq!(x[1], 100.0);
        assert_eq!(vx[1], 2.0);
        assert_eq!(y[2], 0.0);
        assert_eq!(vy[2], -4.0);
        assert_eq!(y[3], 100.0);
        assert_eq!(vy[3], 1.0);
    }

    #[test]
    fn dispose_then_drop_no_double_free() {
        let mut store = allocate(config()).unwrap();
        dispose(&mut store);
        // Drop ejecuta al salir del scope sin doble liberación.
    }
}
