//! app.rs — Bucle de juego del motor de gravitación 2D (Onda 5, Issue 15).
//! Orquesta ventana Raylib, input, física (gravi-core) y render (gravi-ui).
//! Zero-allocation en el hot path: sin Vec/String/format!/closures en
//! PhysicsStep ni en el loop de render.

#![deny(warnings)]

use gravi_core::collisions::{CollisionConfig, resolve_collisions};
use gravi_core::diagnostics::calculate_flops_per_frame;
use gravi_core::parallel_dispatcher::ParallelForceDispatcher;
use gravi_core::scenario::{load_scenario, save_scenario};
use gravi_core::simulation_store::{
    allocate, clear_forces, compute_forces_brute_range, integrate_symplectic_euler, seed_linear,
    SimConfig,
};
use gravi_core::spatial_grid::SpatialGrid;
use gravi_core::static_attractor::StaticAttractor;
use gravi_core::store::NativeStore;
use gravi_core::quadtree::{
    build_quadtree, compute_forces_barnes_hut_range, compute_quadtree_masses, root_from_live_bodies,
    QuadtreeArena,
};
use crate::physics_hud::{write_line1, write_line2};
use crate::render_buffer::RenderBuffer;
use raylib::consts::KeyboardKey;
use raylib::prelude::*;

/// Escenario por defecto (fichero .grav para save/load).
const SCENARIO_PATH: &str = "scenario.grav";

/// Teclas de escenario 1..=4.
const SCENARIO_KEYS: [KeyboardKey; 4] = [
    KeyboardKey::KEY_ONE,
    KeyboardKey::KEY_TWO,
    KeyboardKey::KEY_THREE,
    KeyboardKey::KEY_FOUR,
];

/// Atrayentes estáticos (4 max según formato .grav).
const ATTRACTORS: [StaticAttractor; 4] = [
    StaticAttractor {
        x: 960.0,
        y: 490.0,
        mass: 400.0,
    },
    StaticAttractor {
        x: 300.0,
        y: 300.0,
        mass: 200.0,
    },
    StaticAttractor {
        x: 1620.0,
        y: 680.0,
        mass: 300.0,
    },
    StaticAttractor {
        x: 600.0,
        y: 800.0,
        mass: 150.0,
    },
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

        rl.set_target_fps(60);

        let sim_config = SimConfig {
            body_count: config.body_count,
            width: 1920.0,
            height: 980.0,
        };
        let mut store = allocate(sim_config)?;
        seed_linear(&mut store, 1920.0, 980.0, 40.0);

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

        let physics_step = gravi_core::physics_step::PhysicsStep::create(1.0 / 60.0);

        Ok(Self {
            rl,
            thread,
            store,
            render_buffer,
            quad_arena,
            grid,
            dispatcher,
            attractors: ATTRACTORS.to_vec(),
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
        let speed = match scenario {
            1 => 40.0,
            2 => 80.0,
            3 => 15.0,
            _ => 60.0,
        };
        seed_linear(&mut self.store, 1920.0, 980.0, speed);
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
        let g = self.physics_step.g;
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
                        0.5,
                    );
                }
            }
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

        // La textura se mueve para dibujar y se vuelve a guardar: se conserva
        // la caché entre frames sin clonar el handle de textura.
        // `Texture2D` no es `Copy`: `draw_texture` se queda con el handle y
        // lo descarga al retorno, por tanto se saca de la caché y se vuelve a
        // crear perezosamente en el siguiente `update_texture`.
        if let Some(tex) = self.render_buffer.texture.take() {
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
