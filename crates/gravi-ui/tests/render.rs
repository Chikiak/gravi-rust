use gravi_core::store::NativeStore;
use gravi_ui::render_buffer::RenderBuffer;
use gravi_core::static_attractor::StaticAttractor;
use raylib::prelude::Color;

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
    store.live_count = 2;
    rb.draw_all(&store);

    assert_eq!(rb.image.get_color(3, 4), Color::WHITE);
    assert_eq!(rb.image.get_color(7, 1), Color::WHITE);
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
    let a = StaticAttractor::new(5.0, 5.0, 400.0);
    rb.draw_attractors(&[a]);

    assert_eq!(rb.image.get_color(5, 5), Color::YELLOW);
    assert_eq!(rb.image.get_color(0, 0), Color::BLACK);
}

#[test]
fn imagen_es_rgba_con_tamanio_de_una_subida_a_textura() {
    let rb = RenderBuffer::new(7, 9);
    assert_eq!(rb.image.get_pixel_data_size() as usize, 7 * 9 * 4);
}

#[test]
fn draw_all_descarta_negativos_no_pinta_columna_cero() {
    let mut rb = RenderBuffer::new(4, 4);
    let mut store = make_store(1);
    store.x_slice()[0] = -5.0;
    store.y_slice()[0] = 3.0;
    store.live_count = 1;
    rb.draw_all(&store);

    assert_eq!(rb.image.get_color(0, 3), Color::BLACK);
    for y in 0..4 {
        for x in 0..4 {
            assert_eq!(
                rb.image.get_color(x, y),
                Color::BLACK,
                "pixel ({x}, {y}) debe quedar negro (negativo no pinta)"
            );
        }
    }
}

#[test]
fn draw_all_descarta_fuera_de_rango_derecha_abajo() {
    let mut rb = RenderBuffer::new(4, 4);
    let mut store = make_store(2);
    store.x_slice()[0] = 9.0;
    store.y_slice()[0] = 9.0;
    store.x_slice()[1] = 3.0;
    store.y_slice()[1] = 3.0;
    store.live_count = 2;
    rb.draw_all(&store);

    assert_eq!(rb.image.get_color(3, 3), Color::WHITE);
    assert_eq!(rb.image.get_color(0, 0), Color::BLACK);
}

#[test]
fn draw_all_pinta_cuerpo_en_limite_cero() {
    let mut rb = RenderBuffer::new(4, 4);
    let mut store = make_store(1);
    store.x_slice()[0] = 0.0;
    store.y_slice()[0] = 0.0;
    store.live_count = 1;
    rb.draw_all(&store);

    assert_eq!(rb.image.get_color(0, 0), Color::WHITE);
}
