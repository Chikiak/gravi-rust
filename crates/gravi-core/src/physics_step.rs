//! PhysicsStep — paso de simulación (POD, Copy, 16 B).
//! (Issue 1: PODs y constantes — base mínima para SimulationStore.)

use crate::constants::PhysicsConstants;

/// Paso de simulación: dt, softening, damping, G.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhysicsStep {
    pub dt: f32,
    pub softening: f32,
    pub damping: f32,
    pub g: f32,
}

impl PhysicsStep {
    /// Crea un `PhysicsStep` con `dt` dado y defaults de `PhysicsConstants`
    /// para los campos no especificados.
    pub fn create(dt: f32) -> Self {
        let c = PhysicsConstants::default();
        Self {
            dt,
            softening: c.softening,
            damping: c.damping,
            g: c.g,
        }
    }

    /// Crea un `PhysicsStep` con todos los campos especificados.
    pub fn new(dt: f32, softening: f32, damping: f32, g: f32) -> Self {
        Self {
            dt,
            softening,
            damping,
            g,
        }
    }
}

impl Default for PhysicsStep {
    fn default() -> Self {
        Self::create(0.016) // ~60 FPS
    }
}
