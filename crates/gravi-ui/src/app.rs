//! app.rs — Bucle de juego del motor de gravitación 2D (Onda 5, Issue 15).
//! Orquesta ventana Raylib, input, física (gravi-core) y render (gravi-ui).
//! Zero-allocation en el hot path: sin Vec/String/format!/closures en
//! PhysicsStep ni en el loop de render.

#![deny(warnings)]

use gravi_core::collisions::{CollisionConfig, resolve_collisions};
use gravi_core::constants::PhysicsConstants;
use gravi_core::diagnostics::calculate_flops_per_frame;
use gravi_core::parallel_dispatcher::ParallelForceDispatcher;
use gravi_core::scenario::{load_scenario, save_scenario};
use gravi_core::simulation_store::{
    allocate, clear_forces, compute_forces_brute_range, integrate_symplectic_euler,
    seed_random_cloud, seed_uniform_field, seed_orbital_disk, SimConfig,
};
use gravi_core::spatial_grid::SpatialGrid;
use gravi_core::static_attractor::{apply_attractor_forces, StaticAttractor};
use gravi_core::store::NativeStore;
use gravi_core::quadtree::{
    build_quadtree, compute_forces_barnes_hut_range, compute_quadtree_masses, root_from_live_bodies,
    QuadtreeArena,
};
use crate::physics_hud::{write_line1, write_line2};
use crate::render_buffer::RenderBuffer;
use raylib::consts::KeyboardKey;
use raylib::prelude::*;
use std::time::{SystemTime, UNIX_EPOCH};

/// Escenario por defecto (fichero .grav para save/load).
const SCENARIO_PATH: &str = "scenario.grav";

/// Teclas de escenario 1..=4.
const SCENARIO_KEYS: [KeyboardKey; 4] = [
    KeyboardKey::KEY_ONE,
    KeyboardKey::KEY_TWO,
    KeyboardKey::KEY_THREE,
    KeyboardKey::KEY_FOUR,
];

/// Modo de motor de fuerza.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForceEngine {
    BruteForce,
    BarnesHut,
}

/// Banderas de save/load pendientes (procesadas en frame, no en input).
struct IoFlags {
    save_pending: bool,
    load_pending: bool,
}

/// Configuración de la App proveniente de los argumentos de la línea de
/// comandos (ver `main.rs`).
#[derive(Debug, Clone, Copy)]
pub struct AppConfig {
    /// Número de cuerpos iniciales (default 2000).
    pub body_count: usize,
    /// Motor de fuerza (default Barnes-Hut).
    pub force_engine: ForceEngine,
    /// Usar ruta secuencial en lugar del dispatcher paralelo.
    pub sequential: bool,
    /// FPS objetivo de la ventana (default 60).
    pub target_fps: u32,
}

/// Estado de la aplicación.
pub struct App {
    rl: RaylibHandle,
    thread: RaylibThread,
    store: NativeStore,
    render_buffer: RenderBuffer,
    quad_arena: QuadtreeArena,
    grid: SpatialGrid,
    dispatcher: ParallelForceDispatcher,
    attractors: Vec<StaticAttractor>,
    physics_step: gravi_core::physics_step::PhysicsStep,
    force_engine: ForceEngine,
    sequential: bool,
    manual_simd: bool,
    collisions_enabled: bool,
    paused: bool,
    io: IoFlags,
    scenario: usize,
    active_g: f32,
    interactive: Option<StaticAttractor>,
    hud_line1: [u8; 128],
    hud_line2: [u8; 128],
    frame_count: u64,
    kt_ms: u64,
}

impl App {
    /// Crea la ventana 1920×980 a 60 FPS, asigna store, quadtree, grid,
    /// dispatcher y RenderBuffer según `config`. Abre y cierra sin panic.
    pub fn new(config: AppConfig) -> Result<Self, String> {
        let (mut rl, thread) = raylib::init()
            .size(1920, 980)
            .title("GraviSharp — N-body 2D (gravi-ui)")
            .build();

        rl.set_target_fps(config.target_fps);

        let sim_config = SimConfig {
            body_count: config.body_count,
            width: 1920.0,
            height: 980.0,
        };
        let mut store = allocate(sim_config)?;
        store.init_lifecycle();

        let physics_step = gravi_core::physics_step::PhysicsStep::create(1.0 / 60.0);

        seed_orbital_disk(
            &mut store,
            1920.0,
            980.0,
            50.0,
            300.0,
            10000.0,
            physics_step.softening,
            1337,
        );

        let render_buffer = RenderBuffer::new(1920, 980);

        let quad_arena = QuadtreeArena::allocate(config.body_count)
            .map_err(|e| format!("quadtree arena: {:?}", e))?;

        let mut grid = SpatialGrid::allocate(30, 16, config.body_count as i32)
            .map_err(|e| format!("spatial grid: {:?}", e))?;
        grid.set_cell_size(64.0);

        let n_threads = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4)
            .max(1);
        let dispatcher = ParallelForceDispatcher::new(n_threads);

        Ok(Self {
            rl,
            thread,
            store,
            render_buffer,
            quad_arena,
            grid,
            dispatcher,
            attractors: Vec::new(),
            scenario: 2,
            active_g: PhysicsConstants::default().g,
            interactive: None,
            physics_step,
            force_engine: config.force_engine,
            sequential: config.sequential,
            manual_simd: true,
            collisions_enabled: false,
            paused: false,
            io: IoFlags {
                save_pending: false,
                load_pending: false,
            },
            hud_line1: [0u8; 128],
            hud_line2: [0u8; 128],
            frame_count: 0,
            kt_ms: 0,
        })
    }

    /// Ejecuta el bucle principal hasta que se cierre la ventana.
    pub fn run(mut self) {
        while !self.rl.window_should_close() {
            self.process_input();
            self.process_io_pending();

            if !self.paused {
                let t0 = std::time::Instant::now();
                self.update_physics();
                self.kt_ms = t0.elapsed().as_millis() as u64;
            }

            self.render();
            self.frame_count = self.frame_count.wrapping_add(1);
        }
    }

    /// Lee input. Solo establece flags; nunca ejecuta save/load directamente.
    /// Las lecturas de `rl` ocurren en bloques cortos; el borrow de `self.rl`
    /// siempre se libera antes de mutar cualquier otro campo de `self`.
    fn process_input(&mut self) {
        // Escenarios 1-4: captura de tecla antes de mutar self.
        let scenario: Option<usize> = {
            let rl = &mut self.rl;
            SCENARIO_KEYS
                .iter()
                .enumerate()
                .find_map(|(i, &key)| rl.is_key_pressed(key).then_some(i + 1))
        }; // borrow de `self.rl` liberado aquí
        if let Some(s) = scenario {
            self.load_scenario_seeded(s);
        }

        // Atractor interactivo con el ratón (escenarios 1 y 2).
        let interactive: Option<StaticAttractor> = {
            let rl = &mut self.rl;
            if rl.is_mouse_button_down(MouseButton::MOUSE_BUTTON_LEFT)
                && (self.scenario == 1 || self.scenario == 2)
            {
                let m = rl.get_mouse_position();
                Some(StaticAttractor::new(m.x, m.y, 80000.0))
            } else {
                None
            }
        };
        self.interactive = interactive;

        // M = SIMD toggle, E = engine toggle, C = colisiones, B = bounce.
        // Lectura en bloque; la aplicación de cada toggle ocurre después.
        let (toggle_m, toggle_e, toggle_c, toggle_b, f5, f9, space) = {
            let rl = &self.rl;
            (
                rl.is_key_pressed(KeyboardKey::KEY_M),
                rl.is_key_pressed(KeyboardKey::KEY_E),
                rl.is_key_pressed(KeyboardKey::KEY_C),
                rl.is_key_pressed(KeyboardKey::KEY_B),
                rl.is_key_pressed(KeyboardKey::KEY_F5),
                rl.is_key_pressed(KeyboardKey::KEY_F9),
                rl.is_key_pressed(KeyboardKey::KEY_SPACE),
            )
        }; // borrow de `self.rl` liberado aquí

        if toggle_m {
            self.manual_simd = !self.manual_simd;
        }
        if toggle_e {
            self.force_engine = match self.force_engine {
                ForceEngine::BruteForce => ForceEngine::BarnesHut,
                ForceEngine::BarnesHut => ForceEngine::BruteForce,
            };
        }
        if toggle_c {
            self.collisions_enabled = !self.collisions_enabled;
        }
        if toggle_b {
            self.apply_bounce();
        }
        if f5 {
            self.io.save_pending = true;
        }
        if f9 {
            self.io.load_pending = true;
        }
        if space {
            self.paused = !self.paused;
        }
    }

    /// Procesa save/load pendientes fuera del handler de input (no bloquean render).
    fn process_io_pending(&mut self) {
        if self.io.save_pending {
            self.io.save_pending = false;
            let path = std::path::Path::new(SCENARIO_PATH);
            let _ = save_scenario(path, &self.store, self.attractors.len() as u32);
        }
        if self.io.load_pending {
            self.io.load_pending = false;
            let path = std::path::Path::new(SCENARIO_PATH);
            if let Ok(loaded) = load_scenario(path) {
                if loaded.count == self.store.count {
                    self.store = loaded;
                    self.store.init_lifecycle();
                }
            }
        }
    }

    /// Carga un escenario predefinido (1..=4) semeando el store.
    fn load_scenario_seeded(&mut self, scenario: usize) {
        let rseed: u32 = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0)
            % 31_333_333;
        self.scenario = scenario;
        self.active_g = if scenario == 4 {
            3000.0
        } else {
            PhysicsConstants::default().g
        };
        self.interactive = None;
        self.store.init_lifecycle();
        self.attractors.clear();

        let softening = self.physics_step.softening;
        match scenario {
            1 => {
                let spread = (rseed % 380) as f32 + 30.0;
                let thermal = (rseed % 20) as f32 + 1.0;
                seed_random_cloud(&mut self.store, 1920.0, 980.0, spread, thermal, rseed);
            }
            2 => {
                let inner = (rseed % 30) as f32 + 1.0;
                let outer = (inner * 2.0).max((rseed % 100) as f32 + 5.0);
                let central = (rseed % 3000) as f32 + 300.0;
                seed_orbital_disk(
                    &mut self.store,
                    1920.0,
                    980.0,
                    inner,
                    outer,
                    central,
                    softening,
                    rseed,
                );
            }
            3 => {
                let inner = (rseed % 15) as f32 + 1.0;
                let outer = (inner * 2.0).max((rseed % 500) as f32 + 20.0);
                let central = 25000.0 + (rseed % 10000) as f32;
                seed_orbital_disk(
                    &mut self.store,
                    1920.0,
                    980.0,
                    inner,
                    outer,
                    central,
                    softening,
                    rseed,
                );
                self.attractors.push(StaticAttractor {
                    x: 960.0,
                    y: 490.0,
                    mass: 50000.0,
                });
            }
            _ => {
                let max_speed = (rseed % 120) as f32 + 40.0;
                seed_uniform_field(&mut self.store, 1920.0, 980.0, max_speed, rseed);
            }
        }
    }

    /// Aplica rebote (bounce): invierte Vy de todos los cuerpos.
    fn apply_bounce(&mut self) {
        unsafe {
            let vy = self.store.vy;
            let count = self.store.count;
            let mut i = 0usize;
            while i < count {
                *vy.add(i) = -*vy.add(i);
                i += 1;
            }
        }
    }

    /// Paso de física: clear_forces → fuerzas (brute o Barnes-Hut) → integrar → colisiones.
    fn update_physics(&mut self) {
        let g = self.active_g;
        let dt = self.physics_step.dt;
        let damping = self.physics_step.damping;
        let softening = self.physics_step.softening;

        clear_forces(&mut self.store);

        match self.force_engine {
            ForceEngine::BruteForce => {
                let parallel = !self.sequential && self.store.live_count == self.store.count;
                if parallel {
                    self.dispatcher.compute_forces_parallel_range(
                        &self.store,
                        0,
                        self.store.count,
                        g,
                    );
                } else {
                    compute_forces_brute_range(
                        &self.store,
                        0,
                        self.store.count,
                        self.store.count,
                        g,
                    );
                }
            }
            ForceEngine::BarnesHut => {
                let (cx, cy, half) = root_from_live_bodies(&self.store);
                if half > 0.0 {
                    let _ = build_quadtree(&self.store, &mut self.quad_arena, cx, cy, half);
                    let _ = compute_quadtree_masses(&mut self.quad_arena, &self.store);
                    compute_forces_barnes_hut_range(
                        &self.quad_arena,
                        &self.store,
                        0,
                        self.store.live_count,
                        g,
                        softening,
                        PhysicsConstants::default().barnes_hut_theta,
                    );
                }
            }
        }

        apply_attractor_forces(&mut self.store, &self.attractors, g, softening);
        if let Some(a) = self.interactive {
            apply_attractor_forces(&mut self.store, &[a], g, softening);
        }

        integrate_symplectic_euler(&mut self.store, dt, damping);

        if self.collisions_enabled {
            gravi_core::spatial_grid::rebuild_grid(&self.grid, &self.store);
            let cfg = CollisionConfig::default();
            unsafe {
                resolve_collisions(&mut self.store, &self.grid, cfg);
            }
        }
    }

    /// Render: limpia buffer, dibuja cuerpos y atrayentes, sube textura,
    /// dibuja HUD y frame.
    fn render(&mut self) {
        // Limpia el buffer de píxeles a negro (frame a frame).
        let pixels = self.render_buffer.pixels_mut();
        for p in pixels.iter_mut() {
            *p = 0;
        }

        self.render_buffer.draw_all(&self.store);
        self.render_buffer.draw_attractors(&self.attractors);
        if let Some(a) = self.interactive {
            self.render_buffer.draw_attractors(&[a]);
        }

        // Sube textura antes de begin_drawing (ambos necesitan &mut self.rl).
        let _ = self.render_buffer.update_texture(&mut self.rl, &self.thread);

        // HUD líneas (PhysicsHud, zero-allocation): se escriben antes de
        // begin_drawing, porque `d` toma prestado `self.rl` como mutable.
        let simd_label = if self.manual_simd { "portable" } else { "scalar" };
        let engine_label = match self.force_engine {
            ForceEngine::BruteForce => "brute",
            ForceEngine::BarnesHut => "bhut",
        };

        let mflops = self.compute_mflops();
        let n1 = write_line1(
            &mut self.hud_line1,
            simd_label,
            self.kt_ms,
            mflops,
        );
        let n2 = write_line2(
            &mut self.hud_line2,
            engine_label,
            self.quad_arena.next_free as u64,
            0,
            self.kt_ms,
        );

        let s1 = unsafe { std::str::from_utf8_unchecked(&self.hud_line1[..n1]) };
        let s2 = unsafe { std::str::from_utf8_unchecked(&self.hud_line2[..n2]) };

        let mut d = self.rl.begin_drawing(&self.thread);
        d.clear_background(Color::BLACK);

        // draw_texture toma &Texture2D: se dibuja sin vaciar la caché, así la
        // textura se conserva entre frames sin clonar el handle.
        if let Some(tex) = &self.render_buffer.texture {
            d.draw_texture(tex, 0, 0, Color::WHITE);
        }

        d.draw_text(s1, 10, 10, 20, Color::LIME);
        d.draw_text(s2, 10, 34, 20, Color::CYAN);

        d.draw_fps(10, 82);
    }

    /// MFLOPS estimados del frame actual (para HUD).
    fn compute_mflops(&self) -> f32 {
        let flops = calculate_flops_per_frame(self.store.count);
        let sec = if self.kt_ms == 0 { 0.001 } else { self.kt_ms as f32 / 1000.0 };
        flops as f32 / sec / 1_000_000.0
    }
}

impl Drop for App {
    fn drop(&mut self) {
        self.render_buffer.texture = None;
        let disp = std::mem::replace(
            &mut self.dispatcher,
            ParallelForceDispatcher::new(0),
        );
        disp.shutdown();
    }
}
