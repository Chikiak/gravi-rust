//! Atractor estático (POD).
//!
//! Cuerpo inmune a la gravedad que ejerce atracción sobre los demás.
//! Traducción fiel de `GraviSharp.Core.StaticAttractor`.

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct StaticAttractor {
    pub x: f32,
    pub y: f32,
    pub mass: f32,
}

impl StaticAttractor {
    /// Crea un atractor estático en `(x, y)` con la masa dada.
    pub fn new(x: f32, y: f32, mass: f32) -> Self {
        Self { x, y, mass }
    }
}
