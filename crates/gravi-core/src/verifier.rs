//! PhysicsVerifier — verificador de invariantes físicas (Onda 3).
//! Zero-allocation en el hot path: solo lecturas sobre slices existentes,
//! sin `Vec`, `String`, `format!`, `Box` ni closures que capturen.

use crate::constants::PhysicsConstants;
use crate::diagnostics::compute_diagnostics;
use crate::heap_verifier::HeapVerifier;
use crate::physics_step::PhysicsStep;
use crate::simulation_store::{allocate, clear_forces, compute_forces_brute, integrate_symplectic_euler, SimConfig};
use crate::store::NativeStore;

/// Presets de verificación F2/F3/F5.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerifierPreset {
    F2,
    F3,
    F5,
}

impl VerifierPreset {
    pub fn body_count(self) -> usize {
        match self {
            VerifierPreset::F2 => 2,
            VerifierPreset::F3 => 3,
            VerifierPreset::F5 => 5,
        }
    }

    pub fn ticks(self) -> u64 {
        match self {
            VerifierPreset::F2 => 600,
            VerifierPreset::F3 => 400,
            VerifierPreset::F5 => 200,
        }
    }

    pub fn dt(self) -> f32 {
        PhysicsStep::default().dt
    }
}

/// Resultado de una corrida de verificación.
#[derive(Debug, Clone, Copy)]
pub struct VerificationResult {
    pub preset: VerifierPreset,
    pub ticks: u64,
    pub energy_initial: f32,
    pub energy_final: f32,
    pub energy_drift_pct: f32,
    pub momentum_initial: (f32, f32),
    pub momentum_final: (f32, f32),
    pub momentum_drift_pct: f32,
    pub com_initial: (f32, f32),
    pub com_final: (f32, f32),
    pub com_drift_px: f32,
    pub max_heap_delta: usize,
    pub passed: bool,
}

/// Umbrales de aceptación (F2: two-body simétrico, 600 ticks).
pub const ENERGY_DRIFT_MAX_PCT: f32 = 0.5;
pub const MOMENTUM_DRIFT_MAX_PCT: f32 = 1.0;
pub const COM_DRIFT_MAX_PX: f32 = 5.0;

/// Siembra dos cuerpos simétricos para verificación.
///
/// Cuerpo 0 en `(-sep / 2, 0)` con velocidad `(0, +speed)`;
/// Cuerpo 1 en `(+sep / 2, 0)` con velocidad `(0, -speed)`.
/// Masas unitarias, radio 0.5, centro de masas en el origen.
pub fn seed_two_body_symmetric_for_verification(
    store: &mut NativeStore,
    sep: f32,
    speed: f32,
) {
    let count = store.count;
    debug_assert!(count >= 2, "seed_two_body_symmetric_for_verification: se requieren >= 2 cuerpos");

    let mass = store.mass_slice();
    let x = store.x_slice();
    let y = store.y_slice();
    let vx = store.vx_slice();
    let vy = store.vy_slice();
    let radius = store.radius_slice();

    let half = sep * 0.5;

    // Cuerpo 0: izquierda, velocidad hacia arriba.
    x[0] = -half;
    y[0] = 0.0;
    vx[0] = 0.0;
    vy[0] = speed;
    mass[0] = 1.0;
    radius[0] = 0.5;

    // Cuerpo 1: derecha, velocidad hacia abajo.
    x[1] = half;
    y[1] = 0.0;
    vx[1] = 0.0;
    vy[1] = -speed;
    mass[1] = 1.0;
    radius[1] = 0.5;

    // Cuerpos restantes (si los hubiera): en el origen, sin velocidad.
    let mut i = 2;
    while i < count {
        x[i] = 0.0;
        y[i] = 0.0;
        vx[i] = 0.0;
        vy[i] = 0.0;
        mass[i] = 0.0;
        radius[i] = 0.5;
        i += 1;
    }

    store.live_count = 2;
    store.free_head = -1;
    for g in store.generation_slice().iter_mut() {
        *g = 0;
    }
}

/// Ejecuta la verificación para un preset dado.
///
/// Retorna un `VerificationResult` con las derivas de energía, momento y
/// centro de masas, más `max_heap_delta` (máximo `delta_bytes` medido por
/// `HeapVerifier` en cada frame del bucle de ticks; zero-allocation en el
/// hot path por construcción, por lo tanto medido como 0).
pub fn verify(preset: VerifierPreset) -> VerificationResult {
    let body_count = preset.body_count();
    let ticks = preset.ticks();
    let dt = preset.dt();

    let config = SimConfig {
        body_count,
        width: 200.0,
        height: 200.0,
    };
    let mut store = match allocate(config) {
        Ok(store) => store,
        Err(_) => return failed_verification(preset, ticks),
    };
    crate::store::NativeStore::init_lifecycle(&mut store);

    // Siembra two-body simétrico (cuerpos 0 y 1 activos; el resto en reposo).
    // Parámetros cerca de órbita circular para minimizar error del integrador.
    seed_two_body_symmetric_for_verification(&mut store, 100.0, 5.0);

    let c = PhysicsConstants::default();

    // Diagnóstico inicial.
    let d0 = match compute_diagnostics(&store) {
        Ok(d0) => d0,
        Err(_) => return failed_verification(preset, ticks),
    };

    // Hot path: 0 allocations por tick (solo punteros crudos sobre memoria nativa).
    // Se mide el delta de heap por frame con HeapVerifier.
    let mut heap = HeapVerifier::new();
    let mut max_heap_delta: usize = 0;
    let mut _tick: u64 = 0;
    while _tick < ticks {
        heap.reset_heap_bytes();
        heap.begin_frame();
        clear_forces(&mut store);
        compute_forces_brute(&store, c.g);
        integrate_symplectic_euler(&mut store, dt, c.damping);
        let report = heap.end_frame();
        if report.delta_bytes > 0 {
            let delta = report.delta_bytes as usize;
            if delta > max_heap_delta {
                max_heap_delta = delta;
            }
        }
        _tick += 1;
    }

    // Diagnóstico final.
    let d1 = match compute_diagnostics(&store) {
        Ok(d1) => d1,
        Err(_) => return failed_verification(preset, ticks),
    };

    // Derivas.
    let e0 = d0.total_energy;
    let e1 = d1.total_energy;
    let energy_drift_pct = if e0.abs() > 1e-7 {
        ((e1 - e0).abs() / e0.abs()) * 100.0
    } else {
        0.0
    };

    let p0x = d0.linear_momentum_x;
    let p0y = d0.linear_momentum_y;
    let p1x = d1.linear_momentum_x;
    let p1y = d1.linear_momentum_y;
    let m0 = (p0x * p0x + p0y * p0y).sqrt();
    let m1 = (p1x * p1x + p1y * p1y).sqrt();
    let momentum_drift_pct = if m0 > 1e-7 {
        ((m1 - m0).abs() / m0) * 100.0
    } else {
        // Momento inicial ~0: drift absoluto.
        m1 * 100.0
    };

    let cm0x = d0.center_of_mass_x;
    let cm0y = d0.center_of_mass_y;
    let cm1x = d1.center_of_mass_x;
    let cm1y = d1.center_of_mass_y;
    let com_drift_px = ((cm1x - cm0x) * (cm1x - cm0x) + (cm1y - cm0y) * (cm1y - cm0y)).sqrt();

    let passed = energy_drift_pct < ENERGY_DRIFT_MAX_PCT
        && momentum_drift_pct < MOMENTUM_DRIFT_MAX_PCT
        && com_drift_px < COM_DRIFT_MAX_PX;

    VerificationResult {
        preset,
        ticks,
        energy_initial: e0,
        energy_final: e1,
        energy_drift_pct,
        momentum_initial: (p0x, p0y),
        momentum_final: (p1x, p1y),
        momentum_drift_pct,
        com_initial: (cm0x, cm0y),
        com_final: (cm1x, cm1y),
        com_drift_px,
        max_heap_delta,
        passed,
    }
}

/// Centro de valor centinela usado cuando `allocate` o un diagnóstico fallan:
/// `passed: false` y todos los campos numéricos en 0.
fn failed_verification(preset: VerifierPreset, ticks: u64) -> VerificationResult {
    VerificationResult {
        preset,
        ticks,
        energy_initial: 0.0,
        energy_final: 0.0,
        energy_drift_pct: 0.0,
        momentum_initial: (0.0, 0.0),
        momentum_final: (0.0, 0.0),
        momentum_drift_pct: 0.0,
        com_initial: (0.0, 0.0),
        com_final: (0.0, 0.0),
        com_drift_px: 0.0,
        max_heap_delta: 0,
        passed: false,
    }
}





