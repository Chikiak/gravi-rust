//! Test de invariante end-to-end "cuerpo muerto inerte" (R1-03, hallazgo 45).
//! Solo API pública de gravi_core. Sin `rand`.

use gravi_core::simulation_store::{
    clear_forces, compute_forces_brute, integrate_symplectic_euler,
};
use gravi_core::diagnostics::compute_diagnostics;
use gravi_core::store::NativeStore;

const G: f32 = 1500.0f32;
const DT: f32 = 0.01f32;
const DAMPING: f32 = 1.0f32;
const K: usize = 100;
const DEAD_MASS: f32 = -1.0f32;

/// Crea store con `n_vivos` vivos + 1 muerto (índice n_vivos, mass=-1, basura en x/y/vx/vy).
/// live_count = n_vivos; live_indices[0..n_vivos] = [0..n_vivos-1] (set by init_lifecycle).
fn make_store_con_muerto(n_vivos: usize) -> NativeStore {
    let count = n_vivos + 1;
    let mut store = NativeStore::new(count).unwrap();
    store.init_lifecycle();

    // Canales de vivos y basura del muerto. Los borrados de slices se descartan
    // al cerrar el bloque antes de tocar `live_count` (campo directo del store).
    {
        let x = store.x_slice(); let y = store.y_slice();
        let vx = store.vx_slice(); let vy = store.vy_slice();
        let mass = store.mass_slice(); let radius = store.radius_slice();

        for i in 0..n_vivos {
            x[i] = (i as f32) * 3.0 + 1.0;
            y[i] = (i as f32) * 2.0 + 1.0;
            vx[i] = (i as f32) * 0.5 - 1.0;
            vy[i] = (i as f32) * 0.3 + 0.5;
            mass[i] = (i as f32) + 1.0;
            radius[i] = 0.5;
        }

        // Cuerpo muerto: basura en canales, mass = -1.0.
        mass[n_vivos] = DEAD_MASS;
        x[n_vivos] = 9999.0; y[n_vivos] = -9999.0;
        vx[n_vivos] = 12345.0; vy[n_vivos] = -54321.0;
    }

    store.live_count = n_vivos; // excluye el muerto del rango vivo
    store
}

/// Crea store de control con solo `n_vivos` vivos (misma siembra, sin slot muerto).
fn make_control_store(n_vivos: usize) -> NativeStore {
    let mut store = NativeStore::new(n_vivos).unwrap();
    store.init_lifecycle();

    let x = store.x_slice(); let y = store.y_slice();
    let vx = store.vx_slice(); let vy = store.vy_slice();
    let mass = store.mass_slice(); let radius = store.radius_slice();

    for i in 0..n_vivos {
        x[i] = (i as f32) * 3.0 + 1.0;
        y[i] = (i as f32) * 2.0 + 1.0;
        vx[i] = (i as f32) * 0.5 - 1.0;
        vy[i] = (i as f32) * 0.3 + 0.5;
        mass[i] = (i as f32) + 1.0;
        radius[i] = 0.5;
    }
    store
}

/// Ejecuta un tick completo: clear_forces → compute_forces_brute → integrate_symplectic_euler.
fn step(store: &mut NativeStore) {
    clear_forces(store);
    compute_forces_brute(store, G);
    integrate_symplectic_euler(store, DT, DAMPING);
}

#[test]
fn cuerpo_muerto_inerte_pipeline_end_to_end() {
    let n_vivos = 4usize;
    let dead_idx = n_vivos;

    let mut dead_store = make_store_con_muerto(n_vivos);
    let mut ctrl_store = make_control_store(n_vivos);

    // Estado inicial del muerto (para AC4). Copias por valor (f32 es Copy);
    // los borrados de slices se descartan al cerrar el bloque.
    let (dead_x0, dead_y0, dead_vx0, dead_vy0, dead_m0) = {
        let x = dead_store.x_slice(); let y = dead_store.y_slice();
        let vx = dead_store.vx_slice(); let vy = dead_store.vy_slice();
        let mass = dead_store.mass_slice();
        (
            x[dead_idx],
            y[dead_idx],
            vx[dead_idx],
            vy[dead_idx],
            mass[dead_idx],
        )
    };

    // K ticks sobre ambos stores.
    for _ in 0..K {
        step(&mut dead_store);
        step(&mut ctrl_store);
    }

    // --- AC2: fuerzas de vivos bit-iguales ---
    let dfx = dead_store.fx_slice(); let dfy = dead_store.fy_slice();
    let cfx = ctrl_store.fx_slice(); let cfy = ctrl_store.fy_slice();
    for i in 0..n_vivos {
        assert_eq!(dfx[i], cfx[i], "fuerza x viva {} difiere con muerto", i);
        assert_eq!(dfy[i], cfy[i], "fuerza y viva {} difiere con muerto", i);
    }
    // Fuerza del muerto debe ser cero (filtrado como objetivo).
    assert_eq!(dfx[dead_idx], 0.0);
    assert_eq!(dfy[dead_idx], 0.0);

    // --- AC3: posiciones/velocidades de vivos bit-iguales ---
    let dx = dead_store.x_slice(); let dy = dead_store.y_slice();
    let dvx = dead_store.vx_slice(); let dvy = dead_store.vy_slice();
    let cx = ctrl_store.x_slice(); let cy = ctrl_store.y_slice();
    let cvx = ctrl_store.vx_slice(); let cvy = ctrl_store.vy_slice();
    for i in 0..n_vivos {
        assert_eq!(dx[i], cx[i], "posición x viva {} difiere", i);
        assert_eq!(dy[i], cy[i], "posición y viva {} difiere", i);
        assert_eq!(dvx[i], cvx[i], "velocidad x viva {} difiere", i);
        assert_eq!(dvy[i], cvy[i], "velocidad y viva {} difiere", i);
    }

    // --- AC3: diagnósticos bit-iguales ---
    let d_diag = compute_diagnostics(&dead_store).unwrap();
    let c_diag = compute_diagnostics(&ctrl_store).unwrap();
    assert_eq!(d_diag.kinetic_energy, c_diag.kinetic_energy);
    assert_eq!(d_diag.potential_energy, c_diag.potential_energy);
    assert_eq!(d_diag.total_energy, c_diag.total_energy);
    assert_eq!(d_diag.linear_momentum_x, c_diag.linear_momentum_x);
    assert_eq!(d_diag.linear_momentum_y, c_diag.linear_momentum_y);
    assert_eq!(d_diag.center_of_mass_x, c_diag.center_of_mass_x);
    assert_eq!(d_diag.center_of_mass_y, c_diag.center_of_mass_y);

    // --- AC4: cuerpo muerto inerte tras K ticks ---
    assert_eq!(dead_store.mass_slice()[dead_idx], dead_m0, "masa del muerto cambió");
    assert_eq!(dead_store.x_slice()[dead_idx], dead_x0, "posición x del muerto cambió");
    assert_eq!(dead_store.y_slice()[dead_idx], dead_y0, "posición y del muerto cambió");
    assert_eq!(dead_store.vx_slice()[dead_idx], dead_vx0, "velocidad x del muerto cambió");
    assert_eq!(dead_store.vy_slice()[dead_idx], dead_vy0, "velocidad y del muerto cambió");
}

#[test]
fn diagnosticos_con_muerto_equivales_control() {
    let n_vivos = 3usize;

    let dead_store = make_store_con_muerto(n_vivos);
    let ctrl_store = make_control_store(n_vivos);

    let d = compute_diagnostics(&dead_store).unwrap();
    let c = compute_diagnostics(&ctrl_store).unwrap();

    assert_eq!(d.kinetic_energy, c.kinetic_energy);
    assert_eq!(d.potential_energy, c.potential_energy);
    assert_eq!(d.total_energy, c.total_energy);
    assert_eq!(d.linear_momentum_x, c.linear_momentum_x);
    assert_eq!(d.linear_momentum_y, c.linear_momentum_y);
    assert_eq!(d.center_of_mass_x, c.center_of_mass_x);
    assert_eq!(d.center_of_mass_y, c.center_of_mass_y);
}
