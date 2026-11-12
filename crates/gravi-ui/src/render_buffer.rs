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
    image: Image,
    pub(crate) texture: Option<Texture2D>,
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
            debug_assert_eq!(image.data() as usize % 4, 0);
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
    /// Cada coordenada se castea a `usize` antes de indexar y se valida con una
    /// comprobación sin signo contra el tamaño del buffer.
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
            let ux = xs[i] as usize;
            let uy = ys[i] as usize;
            if ux < width && uy < height {
                pixels[uy * width + ux] = PIXEL_BLANCO;
            }
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

    fn make_store(count: usize) -> NativeStore {
        let mut store = NativeStore::new(count).expect("NativeStore::new");
        store.init_lifecycle();
        store
    }

    #[test]
    fn draw_all_pinta_cuerpos_vivos_blanco() {
        let mut rb = RenderBuffer::new(10, 10);
        let mut store = make_store(5);
        store.x_slice()[0] = 3.0;
        store.y_slice()[0] = 4.0;
        store.x_slice()[1] = 7.0;
        store.y_slice()[1] = 1.0;
        // Solo los dos primeros cuerpos están en la partición viva.
        store.live_count = 2;
        rb.draw_all(&store);

        assert_eq!(rb.image.get_color(3, 4), Color::WHITE);
        assert_eq!(rb.image.get_color(7, 1), Color::WHITE);
        // El resto del buffer sigue negro.
        assert_eq!(rb.image.get_color(0, 0), Color::BLACK);
        assert_eq!(rb.image.get_color(5, 5), Color::BLACK);
    }

    #[test]
    fn draw_all_descarta_fuera_de_rango_sin_panic() {
        let mut rb = RenderBuffer::new(4, 4);
        let store = make_store(3);
        store.x_slice()[0] = 100.0;
        store.y_slice()[0] = 2.0;
        store.x_slice()[1] = 2.0;
        store.y_slice()[1] = 100.0;
        store.x_slice()[2] = 2.0;
        store.y_slice()[2] = 2.0;
        rb.draw_all(&store);

        assert_eq!(rb.image.get_color(2, 2), Color::WHITE);
        assert_eq!(rb.image.get_color(0, 2), Color::BLACK);
        assert_eq!(rb.image.get_color(2, 0), Color::BLACK);
    }

    #[test]
    fn draw_attractors_dibuja_circulos_amarillos_con_radio_esperado() {
        let mut rb = RenderBuffer::new(10, 10);
        // sqrt(400.0) * 0.05 = 20.0 * 0.05 = 1.0 -> radio 1.
        let a = StaticAttractor::new(5.0, 5.0, 400.0);
        rb.draw_attractors(&[a]);

        assert_eq!(rb.image.get_color(5, 5), Color::YELLOW);
        assert_eq!(rb.image.get_color(0, 0), Color::BLACK);
    }

    #[test]
    fn imagen_es_rgba_con_tamanio_de_una_subida_a_textura() {
        let rb = RenderBuffer::new(7, 9);
        // Un frame de 7x9 píxeles RGBA8888 cabe en una sola llamada a
        // update_texture (w * h * 4 bytes).
        assert_eq!(rb.image.get_pixel_data_size() as usize, 7 * 9 * 4);
    }
}
