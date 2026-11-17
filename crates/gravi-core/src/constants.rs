//! PhysicsConstants — valores por defecto del motor de gravitación.
//! (Issue 1: PODs y constantes — base mínima para SimulationStore.)

/// Suavizado al cuadrado para evitar singularidades.
pub const SOFTENING_SQ: f32 = 64.0f32;

/// Umbral por debajo del cual una masa se trata como cero efectiva.
/// Evita divisiones por masa (casi) nula en el hot path de colisiones.
/// Reutilizable por pruebas de R2-05.
pub const MASS_EPS: f32 = 1e-6f32;

/// Constante de gravitación por defecto.
pub const DEFAULT_G: f32 = 1500.0;
/// Suavizado por defecto.
pub const DEFAULT_SOFTENING: f32 = 8.0;
/// Amortiguamiento por defecto.
pub const DEFAULT_DAMPING: f32 = 1.0;
/// Umbral de apertura Barnes-Hut por defecto (theta).
pub const DEFAULT_BARNES_HUT_THETA: f32 = 0.3;

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
            g: DEFAULT_G,
            softening: DEFAULT_SOFTENING,
            damping: DEFAULT_DAMPING,
            barnes_hut_theta: DEFAULT_BARNES_HUT_THETA,
        }
    }
}
