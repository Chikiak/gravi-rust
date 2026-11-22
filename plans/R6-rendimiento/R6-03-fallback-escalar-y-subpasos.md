---
id: R6-03
title: Visibilidad del fallback escalar y subpasos sin realimentación
phase: R6
status: done
depends_on: []
size: M
---

# R6-03 · Visibilidad del fallback escalar y subpasos sin realimentación

Pequeño a propósito: ambos tocan `app.rs` y su lógica de bucle, así que comparten contexto y se agrupan en una sesión.

## Objetivo

Dos mejoras menores del bucle principal que dejan de ser invisibles o que evitan que un frame caro se convierta en caída sostenida:
1. La etiqueta del HUD muestra cuando el motor de fuerza bruta cae al camino escalar por slots muertos.
2. Si la física se pasa del presupuesto del frame, solo se ejecuta un subpaso y el acumulador drena (simulación en cámara lenta, aplicación fluida).

## Alcance

**Haz**
- En `crates/gravi-ui/src/app.rs`: función pura que decide la etiqueta de la línea 1 del HUD (incluyendo el nuevo estado de fallback escalar cuando motor = fuerza bruta y `live_count != count`); función pura que decide el tope efectivo de subpasos y el drenaje a partir del `kt` del frame anterior y el presupuesto del frame; `physics_substeps_for_frame` usa esa función.
- En `crates/gravi-core/src/simulation_store.rs`: documentar la condición y el coste del camino escalar.
- Tests: etiqueta del HUD sin ventana; política de subpasos pura; caso de fuerza bruta con slots muertos (ampliación de `dead_body_invariant.rs`).

**No hagas**
- No reescribir el kernel escalar a SIMD enmascarado (queda en Pendientes).
- No cambiar la firma ni el comportamiento de `fixed_timestep_substeps` (sus tests existentes se fijan y no se tocan).
- No cambiar `dt`, el recorte del acumulador a 0.25 s, ni el formato de las líneas 1/2 del HUD para las etiquetas actuales.

### Archivos previstos
- `crates/gravi-ui/src/app.rs` — modificar — etiqueta HUD, política de subpasos, uso en bucle.
- `crates/gravi-core/src/simulation_store.rs` — modificar — documentación del fallback escalar.
- `crates/gravi-ui/tests/hud.rs` — modificar — test de la etiqueta nueva.
- `tests/gravi-core-tests/src/dead_body_invariant.rs` — modificar — caso con slots muertos.
- `crates/gravi-ui/tests/app_fixed_timestep.rs` — modificar — tests de la política de subpasos.

## Criterios de aceptación

- [ ] **AC1** Con motor de fuerza bruta y `live_count != count`, la etiqueta del HUD indica el fallback escalar; con `live_count == count`, la etiqueta es la de siempre. Verif: test de la función pura de etiqueta en `hud.rs`.
- [ ] **AC2** Con `kt` anterior ≤ presupuesto del frame, el número de subpasos y el residuo coinciden con `fixed_timestep_substeps` a tope 2. Verif: test nuevo en `app_fixed_timestep.rs`.
- [ ] **AC3** Con `kt` anterior > presupuesto y acumulador > `dt`, se ejecuta exactamente 1 subpaso y el acumulador queda ≤ `dt`. Verif: test nuevo.
- [ ] **AC4** Con delta desbocado y `kt` anterior > presupuesto, se ejecuta 1 subpaso y el acumulador queda ≤ `dt`. Verif: test nuevo.
- [ ] **AC5** Los cuatro tests existentes del acumulador siguen verdes sin modificarse. Verif: `cargo test --release app_fixed_timestep`.
- [ ] **AC6** El camino escalar con slots muertos da la misma fuerza que la suma sobre cuerpos vivos. Verif: caso en `dead_body_invariant.rs`.
- [ ] **AC7** `cargo check --workspace --all-targets` → 0 errores, 0 warnings; `cargo test --release` → suite completa en verde.

## Contexto

**Ya existe**
- `compute_forces_brute_range` decide entre camino vectorial y escalar comparando `live_count` con `count`; el escalar salta slots muertos comprobando la masa.
- La etiqueta de la línea 1 del HUD se compone en `app.rs` y se pasa como texto libre a `write_line1`.
- `fixed_timestep_substeps` es pura y está probada: acumula, consume subpasos hasta el tope, recorta a 0.25 s si se desboca.
- `dead_body_invariant.rs` ya compara un store con un cuerpo muerto contra uno de control.

**Contratos que debes respetar**: formato del HUD, presupuesto de frame (duración del frame anterior en ms, comparación estricta), cero asignaciones en hot path — definidos en `../PHASE.md#contratos-compartidos`.

**Ojo**
- `kt_ms` mezcla todos los subpasos del frame; la política compara ese total contra el presupuesto.
- El drenaje se aplica después de consumir subpasos, nunca antes.
- Con colisiones desactivadas `live_count == count`, así que el fallback escalar solo se ve si se activan fusiones.

### Leer primero
- `crates/gravi-core/src/simulation_store.rs` — `compute_forces_brute_range`.
- `crates/gravi-ui/src/app.rs` — composición del HUD, `physics_substeps_for_frame`, `fixed_timestep_substeps`.
- `crates/gravi-ui/tests/hud.rs` y `crates/gravi-ui/tests/app_fixed_timestep.rs` — aserciones existentes que no deben cambiar.
- `tests/gravi-core-tests/src/dead_body_invariant.rs` — patrón de store con muerto vs control.

## Especificación

Estados de la etiqueta de la línea 1 del HUD (bruta únicamente):
| Motor | `live_count` vs `count` | Etiqueta |
| fuerza bruta | iguales | Etiqueta actual de SIMD |
| fuerza bruta | distintos | Etiqueta nueva que nombra el fallback escalar |

Política de subpasos:
| `kt` anterior vs presupuesto | Tope | Drenaje |
| ≤ presupuesto | 2 | Ninguno adicional |
| > presupuesto | 1 | Acumulador limitado a `dt` tras consumir |

### Pasos sugeridos
1. Test de etiqueta nueva y de política de subpasos (fallan antes).
2. Implementar las dos funciones puras en `app.rs`.
3. Documentar el fallback en `simulation_store.rs`.
4. Ampliar `dead_body_invariant.rs` y ejecutar suite completa.

## Tests

- Unidad pura en `crates/gravi-ui/tests/hud.rs`: tabla estados → etiqueta.
- Unidad pura en `crates/gravi-ui/tests/app_fixed_timestep.rs`: tabla (kt, delta, acumulador) → (subpasos, acumulador final).
- Integración en `tests/gravi-core-tests/src/dead_body_invariant.rs`: fuerza bruta con slots muertos vs referencia.

## Verificación

- `cargo test --release hud` → esperado: en verde.
- `cargo test --release app_fixed_timestep` → esperado: nuevos y existentes en verde.
- `cargo test --release dead_body_invariant` → esperado: en verde.
- `cargo test --release` → esperado: suite completa en verde.
- `cargo check --workspace --all-targets` → esperado: 0 errores, 0 warnings.

**Terminado** = AC1..AC7 cumplidos con evidencia + comandos en verde + sin cambios fuera de alcance.

## Si te bloqueas

Si alguna aserción de formato existente del HUD o del acumulador queda en rojo, para y repórtalo: esos contratos están fijados por tests. Decisiones que no son tuyas: cambiar el formato del HUD, el tope base de 2 subpasos, el umbral de recorte.

## Cierre

- Actualiza `status` aquí y en la tabla de `PHASE.md`.
- No edites Alcance ni Criterios de aceptación (si están mal, repórtalo abajo).
- Termina con un resumen: comandos ejecutados y su resultado, AC por AC.

### Notas de implementación

- Rama: `feat/r6-03-fallback-escalar-y-subpasos` (cortada de `feat/r6-02-kernel-bits-dispatch-bruto`).
- Commits (conventional, uno por cambio coherente):
  - `5f78dcb test(gravi-ui): tabla hud_line1_label para fallback escalar (RED)`
  - `eb6eb84 feat(gravi-ui): hud_line1_label + etiqueta HUD en render (fallback escalar solo en brute)`
  - `e64b73a test(gravi-ui): politica de subpasos con presupuesto kt_ms (RED)`
  - `8ebe907 feat(gravi-ui): physics_substeps_policy en physics_substeps_for_frame (presupuesto kt_ms)`
  - `98cccd2 docs(gravi-core): comentar coste O(N²) del fallback escalar en brute_range`
  - `a2bde65 test(gravi-core): invariante dead-body en fuerza bruta (AC6)`
  - `chore(plan): R6-03 done` (este commit: issue frontmatter, tabla PHASE.md y notas).
- Cambios de firma autorizados (ninguna firma existente se modificó):
  - `pub fn hud_line1_label(engine: &str, live_count: usize, count: usize) -> &'static str` (nueva, app.rs).
  - `pub fn physics_substeps_policy(kt_ms: u64, budget_ms: u64, accumulator: f32, dt: f32) -> (usize, f32)` (nueva, app.rs).
  - `physics_substeps_for_frame` ahora suma `frame_dt` al acumulador, llama a `physics_substeps_policy(self.kt_ms, PHYSICS_BUDGET_MS, ...)` y asigna el residuo a `self.accumulator`.
  - `fixed_timestep_substeps` y sus 4 tests existentes: sin cambios.
  - `dt`, recorte 0.25 s y formato de líneas 1/2 del HUD: sin cambios.
- Archivos modificados:
  - `crates/gravi-ui/src/app.rs` (hud_line1_label, physics_substeps_policy, PHYSICS_BUDGET_MS, render + physics_substeps_for_frame).
  - `crates/gravi-ui/tests/hud.rs` (test hud_line1_label_table).
  - `crates/gravi-ui/tests/app_fixed_timestep.rs` (3 tests nuevos de política; 4 existentes intactos).
  - `crates/gravi-core/src/simulation_store.rs` (doc de compute_forces_brute_range: camino escalar O(N²) cuando live_count != count, salta mass<0).
  - `tests/gravi-core-tests/src/dead_body_invariant.rs` (test brute_ignora_slots_muertos).
  - `plans/R6-rendimiento/R6-03-fallback-escalar-y-subpasos.md` (status: done + notas).
  - `plans/R6-rendimiento/PHASE.md` (fila R6-03 a done).
- Desviaciones: ninguna. `manual_simd` sigue leído para Barnes-Hut en render (sin warnings de código muerto); el motor bruto es el único afectado por la nueva etiqueta, como pide el alcance.
- Evidencia AC (cargo test --release, rama feat/r6-03-fallback-escalar-y-subpasos):
  - AC1 (PASS): `cargo test --release hud` → hud_line1_label_table ok ("brute"/100/100→scalar, "brute"/90/100→scalar-fallback, "bhut"/90/100→scalar).
  - AC2 (PASS): `cargo test --release app_fixed_timestep` → policy_presupuesto_ok ok (kt<=budget, n=2, residuo natural).
  - AC3 (PASS): `cargo test --release app_fixed_timestep` → policy_presupuesto_excedido ok (kt>budget, acc>dt, n=1, acc<=dt).
  - AC4 (PASS): `cargo test --release app_fixed_timestep` → policy_delta_desbocado ok (delta desbocado, kt>budget, n=1, acc<=dt).
  - AC5 (PASS): los 4 tests existentes de app_fixed_timestep siguen verdes.
  - AC6 (PASS): `cargo test --release dead_body_invariant` → brute_ignora_slots_muertos ok (fuerza en vivos == suma sobre vivos; muerto sin fuerza).
  - AC7 (PASS): `cargo check --workspace --all-targets` → 0 errores, 0 warnings; `cargo test --release` → suite completa verde.