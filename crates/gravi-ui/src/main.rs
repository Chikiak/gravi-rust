//! main.rs — Punto de entrada de gravi-ui (Onda 5, Issue 15).
//! Crea la App y ejecuta el bucle de juego. Abre y cierra sin panic.
//! Soporta flags de CLI (parseo manual de std::env::args, sin crates):
//!   --verify-physics  verificación headless (VerifierPreset::F2), imprime
//!                     energía/momento/CoM y sale sin abrir ventana
//!   --body-count=N    número de cuerpos iniciales (default 5000, paridad C#)
//!   --engine=parallel|sequential   dispatcher (default parallel)
//!   --force=brute|barnes           motor de fuerza (default barnes)
//! Acepta `--flag value` y `--flag=value`. Flags desconocidos: aviso por
//! stderr y se ignoran.

use gravi_core::constants::DEFAULT_BARNES_HUT_THETA;
use gravi_core::parallel_dispatcher::ParallelForceDispatcher;
use gravi_core::physics_step::PhysicsStep;
use gravi_core::quadtree::{build_quadtree, compute_forces_barnes_hut_range, compute_quadtree_masses, root_from_live_bodies, QuadtreeArena};
use gravi_core::simulation_store::{allocate, clear_forces, compute_forces_brute_range, integrate_symplectic_euler, seed_orbital_disk, SimConfig};
use gravi_core::store::NativeStore;
use gravi_core::verifier::{verify, VerifierPreset};
use gravi_ui::{App};
use gravi_ui::app::{AppConfig, ForceEngine};
use std::time::Instant;

/// Resultado del parseo de argumentos de CLI.
struct CliConfig {
    verify_physics: bool,
    bench_physics: bool,
    bench_sizes: Option<Vec<usize>>,
    body_count: usize,
    engine: Engine,
    force: ForceEngine,
    target_fps: u32,
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

    // Modo headless: harness de medición de física sin abrir ventana (Issue R4-07).
    if cli.bench_physics {
        match run_bench_physics(&cli) {
            Ok(()) => std::process::exit(0),
            Err(e) => {
                eprintln!("gravi-ui: error en --bench-physics: {}", e);
                std::process::exit(1);
            }
        }
    }

    let config = AppConfig {
        body_count: cli.body_count,
        force_engine: cli.force,
        sequential: matches!(cli.engine, Engine::Sequential),
        target_fps: cli.target_fps,
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
        bench_physics: false,
        bench_sizes: None,
        body_count: 5000,
        engine: Engine::Parallel,
        force: ForceEngine::BarnesHut,
        target_fps: 60,
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
            "target-fps" => {
                let v = take_value(&mut i, "--target-fps")?;
                cli.target_fps = v.parse::<u32>().map_err(|_| format!("valor de --target-fps no es un entero válido: '{}'", v))?;
            }
            "bench-physics" => {
                if value_opt.is_some() {
                    eprintln!("gravi-ui: aviso: --bench-physics ignora su valor");
                }
                cli.bench_physics = true;
            }
            "bench-sizes" => {
                let v = take_value(&mut i, "--bench-sizes")?;
                cli.bench_sizes = Some(
                    v.split(',')
                        .map(|t| {
                            t.trim()
                                .parse::<usize>()
                                .map_err(|_| format!("valor de --bench-sizes no es un entero válido: '{}'", t))
                        })
                        .collect::<Result<Vec<usize>, String>>()?,
                );
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

// ============================================================================
// Issue R4-07 — Harness headless --bench-physics
// ============================================================================

/// Harness de medición de física headless (sin ventana Raylib).
///
/// Para cada `N` ejecuta 3 modos sobre stores frescos con el mismo estado
/// inicial (`seed_orbital_disk(1920, 980, 50, 300, 10000, softening, 1337)`):
///   - brute_seq: clear_forces → compute_forces_brute_range → integrate
///   - bh_seq:    quadtree por tick + compute_forces_barnes_hut_range
///   - bh_par:    igual que bh_seq pero fuerzas vía ParallelForceDispatcher
///
/// Imprime `N | brute_seq ms | bh_seq ms | bh_par ms | speedup_bh_par_vs_bh_seq`
/// y devuelve `Ok(())` si todos los modos completan sus ticks.
fn run_bench_physics(cli: &CliConfig) -> Result<(), String> {
    let sizes: &[usize] = match &cli.bench_sizes {
        Some(sizes) if !sizes.is_empty() => sizes,
        _ => &[1000, 2000, 4000, 8000, 12000],
    };

    let step = PhysicsStep::create(1.0 / 60.0);
    let theta = DEFAULT_BARNES_HUT_THETA;
    let g = step.g;
    let softening = step.softening;
    let damping = step.damping;
    let dt = step.dt;
    let warmup: usize = 2;
    let measured: usize = 10;
    let width: f32 = 1920.0;
    let height: f32 = 980.0;

    let n_threads = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4)
        .max(1);

    println!("N | brute_seq ms | bh_seq ms | bh_par ms | speedup_bh_par_vs_bh_seq");

    for &n in sizes {
        let brute_ms = bench_brute_seq(n, warmup, measured, dt, damping, g, width, height, softening)?;
        let bh_seq_ms = bench_bh_seq(n, warmup, measured, dt, damping, g, softening, theta, width, height)?;

        let (bh_par_ms, speedup) = if n_threads >= 2 {
            let ms = bench_bh_par(n, warmup, measured, dt, damping, g, softening, theta, n_threads - 1, width, height)?;
            (Some(ms), Some(if ms > 0.0 { bh_seq_ms / ms } else { 0.0 }))
        } else {
            (None, None)
        };

        let bh_par_text = bh_par_ms.map_or_else(|| "n/a".to_string(), |ms| format!("{:.3}", ms));
        let speedup_text = speedup.map_or_else(|| "n/a".to_string(), |s| format!("{:.2}x", s));
        println!("{} | {:.3} | {:.3} | {} | {}", n, brute_ms, bh_seq_ms, bh_par_text, speedup_text);
    }

    Ok(())
}

/// Crea un store fresco con el estado inicial estándar del banco de pruebas.
fn fresh_store(n: usize, width: f32, height: f32, softening: f32) -> Result<NativeStore, String> {
    let mut store = allocate(SimConfig { body_count: n, width, height })?;
    store.init_lifecycle();
    seed_orbital_disk(&mut store, width, height, 50.0, 300.0, 10000.0, softening, 1337);
    Ok(store)
}

/// Ejecuta `warmup` ticks sin medir y `measured` ticks cronometrados; devuelve
/// el promedio de ms por tick.
fn measure_ms<F: FnMut()>(warmup: usize, measured: usize, mut tick: F) -> f32 {
    for _ in 0..warmup {
        tick();
    }
    let start = Instant::now();
    for _ in 0..measured {
        tick();
    }
    start.elapsed().as_secs_f32() * 1000.0 / measured as f32
}

fn bench_brute_seq(
    n: usize,
    warmup: usize,
    measured: usize,
    dt: f32,
    damping: f32,
    g: f32,
    width: f32,
    height: f32,
    softening: f32,
) -> Result<f32, String> {
    let mut store = fresh_store(n, width, height, softening)?;
    let count = n;
    let ms = measure_ms(warmup, measured, || {
        clear_forces(&mut store);
        compute_forces_brute_range(&store, 0, count, count, g);
        integrate_symplectic_euler(&mut store, dt, damping);
    });
    Ok(ms)
}

/// Un tick Barnes-Hut secuencial sobre un arena reutilizado entre ticks.
fn tick_bh_seq(
    store: &mut NativeStore,
    arena: &mut QuadtreeArena,
    n: usize,
    g: f32,
    softening: f32,
    theta: f32,
    dt: f32,
    damping: f32,
) -> Result<(), String> {
    clear_forces(store);
    let (cx, cy, half) = root_from_live_bodies(store);
    build_quadtree(store, arena, cx, cy, half).map_err(|e| format!("{:?}", e))?;
    compute_quadtree_masses(arena, store).map_err(|e| format!("{:?}", e))?;
    compute_forces_barnes_hut_range(arena, store, 0, n, g, softening, theta);
    integrate_symplectic_euler(store, dt, damping);
    Ok(())
}

fn bench_bh_seq(
    n: usize,
    warmup: usize,
    measured: usize,
    dt: f32,
    damping: f32,
    g: f32,
    softening: f32,
    theta: f32,
    width: f32,
    height: f32,
) -> Result<f32, String> {
    let mut store = fresh_store(n, width, height, softening)?;
    let mut arena = QuadtreeArena::allocate(n).map_err(|e| format!("{:?}", e))?;
    let count = n;
    for _ in 0..warmup {
        tick_bh_seq(&mut store, &mut arena, count, g, softening, theta, dt, damping)?;
    }
    let start = Instant::now();
    for _ in 0..measured {
        tick_bh_seq(&mut store, &mut arena, count, g, softening, theta, dt, damping)?;
    }
    Ok(start.elapsed().as_secs_f32() * 1000.0 / measured as f32)
}

/// Un tick Barnes-Hut paralelo (fuerzas vía dispatcher) sobre arena reutilizada.
fn tick_bh_par(
    store: &mut NativeStore,
    arena: &mut QuadtreeArena,
    dispatcher: &ParallelForceDispatcher,
    n: usize,
    g: f32,
    softening: f32,
    theta: f32,
    dt: f32,
    damping: f32,
) -> Result<(), String> {
    clear_forces(store);
    let (cx, cy, half) = root_from_live_bodies(store);
    build_quadtree(store, arena, cx, cy, half).map_err(|e| format!("{:?}", e))?;
    compute_quadtree_masses(arena, store).map_err(|e| format!("{:?}", e))?;
    dispatcher.compute_forces_barnes_hut_parallel(arena, store, 0, n, g, softening, theta);
    integrate_symplectic_euler(store, dt, damping);
    Ok(())
}

fn bench_bh_par(
    n: usize,
    warmup: usize,
    measured: usize,
    dt: f32,
    damping: f32,
    g: f32,
    softening: f32,
    theta: f32,
    dispatcher_threads: usize,
    width: f32,
    height: f32,
) -> Result<f32, String> {
    let mut store = fresh_store(n, width, height, softening)?;
    let mut arena = QuadtreeArena::allocate(n).map_err(|e| format!("{:?}", e))?;
    let dispatcher = ParallelForceDispatcher::new(dispatcher_threads);
    let count = n;
    for _ in 0..warmup {
        tick_bh_par(&mut store, &mut arena, &dispatcher, count, g, softening, theta, dt, damping)?;
    }
    let start = Instant::now();
    for _ in 0..measured {
        tick_bh_par(&mut store, &mut arena, &dispatcher, count, g, softening, theta, dt, damping)?;
    }
    dispatcher.shutdown();
    Ok(start.elapsed().as_secs_f32() * 1000.0 / measured as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bench_physics_headless_no_ventana() {
        let cli = CliConfig {
            verify_physics: false,
            bench_physics: true,
            bench_sizes: Some(vec![1000, 2000]),
            body_count: 5000,
            engine: Engine::Parallel,
            force: ForceEngine::BarnesHut,
            target_fps: 60,
        };
        let result = run_bench_physics(&cli);
        assert!(
            result.is_ok(),
            "run_bench_physics debe completar sin ventana de Raylib: {:?}",
            result.err()
        );
    }
}
