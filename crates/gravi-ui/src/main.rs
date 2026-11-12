//! main.rs — Punto de entrada de gravi-ui (Onda 5, Issue 15).
//! Crea la App y ejecuta el bucle de juego. Abre y cierra sin panic.

use gravi_ui::App;

fn main() {
    let app = match App::new() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("gravi-ui: error al iniciar: {}", e);
            std::process::exit(1);
        }
    };
    app.run();
}
