//! render_buffer.rs — Framebuffer de píxeles RGBA sobre `raylib::Image`.
//!
//! RenderBuffer gestiona un buffer de píxeles `u32` (RGBA8888) sobre la memoria
//! de una `Image` de raylib, rasteriza cuerpos vivos y atrayentes estáticos,
//! y sube el frame completo a una `Texture2D` con una única llamada a
//! `update_texture`. El buffer se reserva una vez en `new` y se reutiliza
//! frame a frame: sin asignaciones en el camino caliente.
//!
//! El formato de memoria es RGBA8888 en orden little-endian (byte[0] = R,
//! byte[3] = A), por lo que un `u32` leído desde `image.data` contiene el píxel
//! empaquetado directamente.

#![deny(warnings)]

use gravi_core::static_attractor::StaticAttractor;
use gravi_core::store::NativeStore;
use raylib::error::Error;
use raylib::prelude::*;

/// Píxel blanco RGBA8888 empaquetado como `u32` (0xRRGGBBAA en memoria LE).
const PIXEL_BLANCO: u32 = 0xFFFF_FFFF;

/// Buffer de píxeles `u32` RGBA sobre una `Image` de raylib, más su
/// `Texture2D` asociada en GPU (creada perezosamente en `update_texture`).
pub struct RenderBuffer {
    pub image: Image,
    pub texture: Option<Texture2D>,
}

impl RenderBuffer {
    /// Crea un buffer de `width × height` píxeles inicialmente negros.
    /// Solo toca memoria CPU: no requiere ventana ni contexto OpenGL.
    pub fn new(width: i32, height: i32) -> Self {
        let image = Image::gen_image_color(width, height, Color::BLACK);
        // Los datos de `Image` provienen del allocador de raylib, alineados al
        // menos a 8 bytes en Windows/x86_64: suficiente para indexarlos como
        // `u32` (alineación 4) sin ningún padding.
        unsafe {
            assert_eq!(image.data() as usize % 4, 0, "Image data must be 4-byte aligned for u32 pixel access");
        }
        Self {
            image,
            texture: None,
        }
    }

    pub fn width(&self) -> i32 {
        self.image.width()
    }

    pub fn height(&self) -> i32 {
        self.image.height()
    }

    /// Reasigna la `Image` a nuevas dimensiones e invalida la textura GPU
    /// (`None`) para que se re-cree perezosamente en `update_texture`.
    /// Early-return si las dimensiones ya son las pedidas (sin asignaciones).
    pub fn resize(&mut self, width: i32, height: i32) {
        if self.image.width() == width && self.image.height() == height {
            return;
        }
        self.image = Image::gen_image_color(width, height, Color::BLACK);
        // Los datos de `Image` provienen del allocador de raylib, alineados al
        // menos a 8 bytes en Windows/x86_64: suficiente para indexarlos como
        // `u32` (alineación 4) sin ningún padding.
        unsafe {
            assert_eq!(
                self.image.data() as usize % 4,
                0,
                "Image data must be 4-byte aligned for u32 pixel access"
            );
        }
        self.texture = None;
    }

    /// Devuelve los datos de `image` como `&mut [u32]` RGBA8888.
    pub(crate) fn pixels_mut(&mut self) -> &mut [u32] {
        let len = (self.image.width() as usize) * (self.image.height() as usize);
        unsafe {
            std::slice::from_raw_parts_mut(self.image.data() as *mut u32, len)
        }
    }

    /// Devuelve los datos de `image` como `&[u8]` crudos (bytes RGBA).
    fn image_bytes(image: &Image) -> &[u8] {
        let len = (image.width() as usize) * (image.height() as usize) * 4;
        unsafe {
            std::slice::from_raw_parts(image.data() as *const u8, len)
        }
    }

    /// Rasteriza los cuerpos vivos de `store` como píxeles blancos.
    ///
    /// Se recorren solo los `live_count` primeros índices de `live_indices`.
    /// Las coordenadas se validan antes del cast (`0 <= x < width`,
    /// `0 <= y < height`, equivalente al check sin signo de C#), por lo que los
    /// valores negativos nunca saturan a la columna/fila 0 y nada se pinta fuera
    /// del buffer.
    pub fn draw_all(&mut self, store: &NativeStore) {
        let width = self.image.width() as usize;
        let height = self.image.height() as usize;
        let pixels = self.pixels_mut();
        let live_indices = store.live_indices_slice();
        let live = &live_indices[..store.live_count];
        let xs = store.x_slice();
        let ys = store.y_slice();
        for &idx in live {
            let i = idx as usize;
            let x = xs[i];
            let y = ys[i];
            if x < 0.0 || y < 0.0 || x >= width as f32 || y >= height as f32 {
                continue;
            }
            let ux = x as usize;
            let uy = y as usize;
            pixels[uy * width + ux] = PIXEL_BLANCO;
        }
    }

    /// Dibuja cada atrayente como un círculo amarillo de radio
    /// `sqrt(mass) * 0.05`.
    pub fn draw_attractors(&mut self, attractors: &[StaticAttractor]) {
        for a in attractors {
            let radius = (a.mass.sqrt() * 0.05) as i32;
            self.image
                .draw_circle(a.x as i32, a.y as i32, radius, Color::YELLOW);
        }
    }

    /// Sube el buffer entero a la textura GPU en una sola llamada a
    /// `update_texture`, creando la textura perezosamente si aún no existe.
    ///
    /// Contrato de re-intento: ante `Err` (fallo de `load_texture_from_image`
    /// o de subida de p\u00edxeles), el caller debe invalidar `texture`
    /// (`None`); la siguiente llamada con `texture.is_none()` vuelve a llamar
    /// a `load_texture_from_image`, por lo que un fallo transitorio no deja
    /// la app sin textura de forma permanente. Firma sin cambios.
    pub fn update_texture(
        &mut self,
        rl: &mut RaylibHandle,
        thread: &RaylibThread,
    ) -> Result<(), Error> {
        if self.texture.is_none() {
            self.texture = Some(rl.load_texture_from_image(thread, &self.image)?);
        }
        let bytes = Self::image_bytes(&self.image);
        self.texture.as_mut().unwrap().update_texture(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_buffer_pixels_mut_alineado_a_4() {
        let mut rb = RenderBuffer::new(8, 8);
        let pixels = rb.pixels_mut();
        let ptr = pixels.as_ptr() as usize;
        assert_eq!(ptr % 4, 0, "dirección base de pixels_mut debe ser múltiplo de 4");
    }

    /// AC3: `pixels.fill(0)` deja todos los pixeles en negro.
    /// `fill(0)` cero los 4 bytes de cada pixel (alpha incluida), equivalente
    /// al bucle anterior `*p = 0`: negro transparente `0x00000000`.
    /// No requiere ventana (`pixels_mut` es `pub(crate)`).
    #[test]
    fn render_buffer_limpieza_fill_0_deja_buffer_negro() {
        let mut rb = RenderBuffer::new(4, 4);
        rb.pixels_mut().fill(0);
        let pixels = rb.pixels_mut();
        assert!(
            pixels.iter().all(|&p| p == 0),
            "fill(0) debe poner todos los pixeles a 0x00000000"
        );
        for y in 0..4 {
            for x in 0..4 {
                let c = rb.image.get_color(x, y);
                assert_eq!(
                    (c.r, c.g, c.b, c.a),
                    (0, 0, 0, 0),
                    "pixel ({x}, {y}) debe ser negro (alpha incluida) tras fill(0)"
                );
            }
        }
    }
}


