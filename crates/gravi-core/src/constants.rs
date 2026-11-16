//! PhysicsConstants — valores por defecto del motor de gravitación.
//! (Issue 1: PODs y constantes — base mínima para SimulationStore.)

/// Suavizado al cuadrado para evitar singularidades (8.0² = 64.0).
pub const SOFTENING_SQ: f32 = 64.0f32;

/// Constantes físicas por defecto del motor N-body.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhysicsConstants {
    /// Constante de gravitación G.
    pub g: f32,
    /// Suavizado (softening) para evitar singularidades.
    pub softening: f32,
    /// Amortiguamiento (damping) aplicado en la integración.
    pub damping: f32,
    /// Umbral de apertura (theta) de Barnes-Hut.
    pub barnes_hut_theta: f32,
}

impl Default for PhysicsConstants {
    fn default() -> Self {
        Self {
            g: 1500.0,
            softening: 8.0,
            damping: 1.0,
            barnes_hut_theta: 0.3,
        }
    }
}
