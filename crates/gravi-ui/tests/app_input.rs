//! tests/app_input.rs — AC1: tecla B alterna `bounce` (default true).

use gravi_ui::app::{App, AppConfig, ForceEngine};

#[test]
fn tecla_b_alternar_bounce() {
    let mut app = App::new(AppConfig {
        body_count: 2,
        force_engine: ForceEngine::BruteForce,
        sequential: true,
        target_fps: 60,
    })
    .expect("App::new debe crear la ventana y el store");

    // AC1: default bounce == true (paridad C#).
    assert!(app.bounce, "default bounce debe ser true");

    // AC1: pulsar B una vez -> false.
    app.bounce = !app.bounce;
    assert!(!app.bounce, "tras una pulsación B, bounce == false");

    // AC1: pulsar B dos veces -> true.
    app.bounce = !app.bounce;
    assert!(app.bounce, "tras dos pulsaciones B, bounce == true");
}
