//! main.rs — Punto de entrada de gravi-ui (Onda 5, Issue 15).
//! Crea la App y ejecuta el bucle de juego. Abre y cierra sin panic.
//! Soporta flags de CLI (parseo manual de std::env::args, sin crates):
//!   --verify-physics  verificación headless (VerifierPreset::F2), imprime
//!                     energía/momento/CoM y sale sin abrir ventana
//!   --body-count=N    número de cuerpos iniciales (default 2000)
//!   --engine=parallel|sequential   dispatcher (default parallel)
//!   --force=brute|barnes           motor de fuerza (default barnes)
//! Acepta `--flag value` y `--flag=value`. Flags desconocidos: aviso por
//! stderr y se ignoran.

use gravi_core::verifier::{verify, VerifierPreset};
use gravi_ui::{App};
use gravi_ui::app::{AppConfig, ForceEngine};

/// Resultado del parseo de argumentos de CLI.
struct CliConfig {
    verify_physics: bool,
    body_count: usize,
    engine: Engine,
    force: ForceEngine,
}

/// Motor de dispatcher solicitado por CLI.
#[derive(Clone, Copy)]
enum Engine {
    Parallel,
    Sequential,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cli = match parse_args(&args) {
        Ok(cli) => cli,
        Err(err) => {
            eprintln!("gravi-ui: error de argumentos: {}", err);
            std::process::exit(2);
        }
    };

    // Modo headless: verificación de física sin abrir ventana.
    if cli.verify_physics {
        let result = verify(VerifierPreset::F2);
        println!(
            "gravi-ui: verify-physics (preset F2, {} ticks, {} cuerpos)",
            result.ticks, result.preset.body_count()
        );
        println!(
            "  energia: inicial={:.6} final={:.6} drift={:.4}% (umbral 0.5)",
            result.energy_initial, result.energy_final, result.energy_drift_pct
        );
        println!(
            "  momento: inicial=({:.6},{:.6}) final=({:.6},{:.6}) drift={:.4}% (umbral 1.0)",
            result.momentum_initial.0, result.momentum_initial.1,
            result.momentum_final.0, result.momentum_final.1,
            result.momentum_drift_pct
        );
        println!(
            "  CoM: inicial=({:.6},{:.6}) final=({:.6},{:.6}) drift={:.4}px (umbral 5.0)",
            result.com_initial.0, result.com_initial.1,
            result.com_final.0, result.com_final.1,
            result.com_drift_px
        );
        println!("  max_heap_delta={}", result.max_heap_delta);
        println!("  passed={}", result.passed);
        std::process::exit(if result.passed { 0 } else { 1 });
    }

    let config = AppConfig {
        body_count: cli.body_count,
        force_engine: cli.force,
        sequential: matches!(cli.engine, Engine::Sequential),
    };

    let app = match App::new(config) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("gravi-ui: error al iniciar: {}", e);
            std::process::exit(1);
        }
    };
    app.run();
}

/// Parsea `std::env::args()` manualmente (sin crates externos).
/// Acepta `--flag value` y `--flag=value`. Flags desconocidos: aviso por
/// stderr y se ignoran. Errores de valor devuelven `Err`.
fn parse_args(args: &[String]) -> Result<CliConfig, String> {
    let mut cli = CliConfig {
        verify_physics: false,
        body_count: 2000,
        engine: Engine::Parallel,
        force: ForceEngine::BarnesHut,
    };

    let mut i = 1;
    while i < args.len() {
        let arg = &args[i];
        if !arg.starts_with("--") {
            i += 1;
            continue;
        }

        let (flag, value_opt) = split_flag(arg);

        let take_value = |i: &mut usize, name: &str| -> Result<String, String> {
            if let Some(v) = value_opt {
                Ok(v.to_string())
            } else if *i + 1 < args.len() {
                *i += 1;
                Ok(args[*i].clone())
            } else {
                Err(format!("falta valor para {}", name))
            }
        };

        match flag {
            "verify-physics" => {
                if value_opt.is_some() {
                    eprintln!("gravi-ui: aviso: --verify-physics ignora su valor");
                }
                cli.verify_physics = true;
            }
            "body-count" => {
                let v = take_value(&mut i, "--body-count")?;
                cli.body_count = parse_usize(&v, "--body-count")?;
            }
            "engine" => {
                let v = take_value(&mut i, "--engine")?;
                cli.engine = match v.as_str() {
                    "parallel" => Engine::Parallel,
                    "sequential" => Engine::Sequential,
                    _ => return Err(format!("valor de --engine no válido: '{}'", v)),
                };
            }
            "force" => {
                let v = take_value(&mut i, "--force")?;
                cli.force = match v.as_str() {
                    "brute" => ForceEngine::BruteForce,
                    "barnes" => ForceEngine::BarnesHut,
                    _ => return Err(format!("valor de --force no válido: '{}'", v)),
                };
            }
            _unknown => {
                eprintln!("gravi-ui: aviso: bandera desconocida ignorada: {}", arg);
            }
        }
        i += 1;
    }

    Ok(cli)
}

/// Divide `--flag=value` en `(flag, Some(value))` o `(flag, None)`.
fn split_flag(arg: &str) -> (&str, Option<&str>) {
    let rest = &arg[2..];
    match rest.find('=') {
        Some(pos) => (&rest[..pos], Some(&rest[pos + 1..])),
        None => (rest, None),
    }
}

fn parse_usize(text: &str, name: &str) -> Result<usize, String> {
    text.parse::<usize>()
        .map_err(|_| format!("valor de {} no es un entero válido: '{}'", name, text))
}
