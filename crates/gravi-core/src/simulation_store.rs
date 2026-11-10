//! simulation_store.rs — Parte A: allocate, dispose, seed_linear, reflect_bounds.
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
