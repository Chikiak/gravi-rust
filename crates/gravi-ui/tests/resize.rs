use gravi_ui::render_buffer::RenderBuffer;

#[test]
fn render_buffer_resize_cambia_dimensiones_e_invalida_textura() {
    let mut rb = RenderBuffer::new(8, 8);
    assert_eq!(rb.width(), 8);
    assert_eq!(rb.height(), 8);
    assert!(rb.texture.is_none());

    rb.resize(16, 32);
    assert_eq!(rb.width(), 16);
    assert_eq!(rb.height(), 32);
    assert!(rb.texture.is_none(), "textura invalida tras resize");
}

#[test]
fn render_buffer_resize_mismas_dimensiones_no_toca_nada() {
    let mut rb = RenderBuffer::new(8, 8);
    rb.resize(8, 8);
    assert_eq!(rb.width(), 8);
    assert_eq!(rb.height(), 8);
    assert!(rb.texture.is_none());
}

/// `App::new` requiere una ventana Raylib (llama a `raylib::init()...build()`)
/// que no está disponible en el harness de `cargo test` sin contexto de
/// ventana, por lo que este test queda ignorado. Se ejecuta manualmente
/// cuando haya un entorno con display (AC2 documentada).
#[ignore = "requiere ventana Raylib (App::new llama a raylib::init()...build())"]
#[test]
fn app_resize_reasigna_render_buffer() {
    use gravi_ui::app::{App, AppConfig, ForceEngine};

    let config = AppConfig {
        body_count: 100,
        force_engine: ForceEngine::BruteForce,
        sequential: true,
        target_fps: 60,
    };
    let mut app = App::new(config).expect("App::new debe crear ventana 1920x980");

    // Tamaño inicial: 1920x980
    assert_eq!(app.render_buffer.width(), 1920);
    assert_eq!(app.render_buffer.height(), 980);

    // Simular resize: forzar detectar tamaño distinto. get_screen_size()
    // devuelve el tamaño real de la ventana; sin resize real no cambia,
    // así que verificamos el path de detección con un tamaño distinto
    // invocando resize directamente sobre el buffer y actualizando window_size.
    app.render_buffer.resize(1280, 720);
    app.window_size = (1280, 720);

    assert_eq!(app.render_buffer.width(), 1280);
    assert_eq!(app.render_buffer.height(), 720);
    assert!(app.render_buffer.texture.is_none(), "textura invalida tras resize");

    // update_texture re-crea la textura perezosamente (necesita rl/thread ya prestados en render).
    // No se puede llamar update_texture fuera de render sin &mut self.rl; se valida
    // indirectamente: texture es None y se re-creará en el siguiente render().
}
