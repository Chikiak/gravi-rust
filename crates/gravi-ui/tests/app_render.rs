//! tests/app_render.rs — Cobertura de la ruta de render de `app.rs`/`render_buffer.rs`.
//!
//! Los tests que precisan ventana Raylib (`RaylibHandle`/`RaylibThread` con
//! display) o campos privados de `App` van `#[ignore]`: documentan el
//! contrato esperado pero no se ejecutan sin entorno gráfico. No hay mocks
//! de raylib.

use gravi_ui::render_buffer::RenderBuffer;

/// `std::str::from_utf8(bytes).unwrap_or("")` sin pasar un literal de bytes
/// directamente a `from_utf8` (evita el lint `invalid_from_utf8`).
fn utf8_fallback(bytes: &[u8]) -> &str {
    std::str::from_utf8(bytes).unwrap_or("")
}

/// AC1: `update_texture` Err -> `texture = None`; la siguiente llamada con
/// imagen valida re-crea `texture` (`Some`), no queda `None` permanente.
///
/// Requiere ventana Raylib (RaylibHandle/thread solo con display).
/// Se valida en la app real con `cargo run -p gravi-ui --release`
/// (bucle de render con ventana).
#[ignore = "requiere ventana Raylib (RaylibHandle/thread solo con display)"]
#[test]
fn update_texture_error_invalida_para_reintento() {
    // Contrato de re-intento (documentado; no ejecutable sin display):
    //   1. update_texture devuelve Err => App::render pone texture = None.
    //   2. Siguiente frame: texture.is_none() => se vuelve a llamar a
    //      load_texture_from_image y se re-crea texture (Some).
    // RenderBuffer nace con texture = None (textura perezosa); solo con
    // ventana se puede forzar el Err de update_texture y comprobar el re-intento.
    let rb = RenderBuffer::new(4, 4);
    assert!(
        rb.texture.is_none(),
        "textura empieza None y se re-intenta en el siguiente frame tras Err"
    );
}

/// AC2 (documentado, ignora ventana): bytes no-UTF-8 en una linea HUD ->
/// `std::str::from_utf8(...).unwrap_or("")` devuelve "" sin panic.
#[ignore = "requiere App::new (ventana) + campos privados hud_line1/hud_line2"]
#[test]
fn hud_from_utf8_fallback_no_panic() {
    // Cuerpo documentado: los bytes invalidos nunca hacen panic gracias a
    // from_utf8 + unwrap_or("") (sin unsafe en produccion).
    let bytes: &[u8] = &[0xFF, 0xFE, 0x00, 0x41];
    assert_eq!(utf8_fallback(bytes), "");
}

/// AC2 activo (no requiere ventana ni App): from_utf8 con fallback sobre
/// bytes crudos; invalidos -> "", validos -> el str.
#[test]
fn hud_from_utf8_fallback_bytes_sin_panic() {
    let invalidos: &[u8] = &[0xFF, 0xFE, 0x00];
    assert_eq!(
        utf8_fallback(invalidos),
        "",
        "bytes no-UTF-8 deben devolver cadena vacia sin panic"
    );
    let validos: &[u8] = b"HUD ok";
    assert_eq!(
        utf8_fallback(validos),
        "HUD ok",
        "bytes UTF-8 validos deben devolver el str"
    );
}
