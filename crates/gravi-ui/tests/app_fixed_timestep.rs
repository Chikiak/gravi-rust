//! Tests del acumulador de fixed timestep (R1-08).
//! Lógica pura: no requiere ventana Raylib.

use gravi_ui::app::fixed_timestep_substeps;

const DT: f32 = 1.0 / 60.0;

#[test]
fn fixed_timestep_substeps_count() {
    let mut acc = 0.0f32;

    // frame pequeño: ningún subpaso completo, residuo arrastrado
    let n = fixed_timestep_substeps(&mut acc, DT, 0.4 * DT, usize::MAX);
    assert_eq!(n, 0);
    assert!((acc - 0.4 * DT).abs() < f32::EPSILON * 8.0);

    // frame que completa exactamente dos subpasos
    let n = fixed_timestep_substeps(&mut acc, DT, 2.0 * DT, usize::MAX);
    assert_eq!(n, 2);
    assert!((acc - 0.4 * DT).abs() < f32::EPSILON * 8.0);

    // residuo arrastrado al frame siguiente: 0.4dt + 0.8dt = 1.2dt -> 1 subpaso, residuo 0.2dt
    let n = fixed_timestep_substeps(&mut acc, DT, 0.8 * DT, usize::MAX);
    assert_eq!(n, 1);
    assert!((acc - 0.2 * DT).abs() < f32::EPSILON * 8.0);
}

#[test]
fn fixed_timestep_equiv_pasos_fijos() {
    let mut acc_a = 0.0f32;
    let mut total_a = 0usize;
    for _ in 0..7 {
        total_a += fixed_timestep_substeps(&mut acc_a, DT, DT, usize::MAX);
    }

    let mut acc_b = 0.0f32;
    let total_b = fixed_timestep_substeps(&mut acc_b, DT, 7.0 * DT, usize::MAX);

    assert_eq!(total_a, total_b, "el acumulador es determinista");
    assert!((acc_a - acc_b).abs() < f32::EPSILON * 16.0, "residuo equivalente");
}

#[test]
fn fixed_timestep_respeta_max_steps() {
    let mut acc = 0.0f32;
    let n = fixed_timestep_substeps(&mut acc, DT, 10.0 * DT, 2);
    assert_eq!(n, 2, "tope de max_steps: debe ejecutar exactamente 2 subpasos");
    assert!((acc - 8.0 * DT).abs() < f32::EPSILON * 8.0, "residuo debe ser 8*dt");
}

#[test]
fn fixed_timestep_drena_recorte_descontrolado() {
    let mut acc = 0.0f32;
    // delta = 30.0 * DT ≈ 500 ms, muy por encima del umbral de recorte (~0.25 s)
    let n = fixed_timestep_substeps(&mut acc, DT, 30.0 * DT, 2);
    assert_eq!(n, 2, "tope de max_steps aunque el delta sea descontrolado");
    assert!((acc - 0.25).abs() < f32::EPSILON * 8.0, "acumulador recortado a 0.25 s, no desbocado");
}
