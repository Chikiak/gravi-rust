---
id: R6-02
title: El dispatch de fuerza bruta publica su kernel
phase: R6
status: done
depends_on: []
size: S
---

# R6-02 · El dispatch de fuerza bruta publica su kernel

Pequeño a propósito: dos líneas de producción merecen sesión propia por ser un bug de
corrección silencioso cuyo test de regresión falla hoy.

## Objetivo

Que alternar el motor de fuerza con la tecla E no deje a los workers calculando Barnes-Hut
con un árbol obsoleto. Hoy `compute_forces_parallel_range` no publica `kernel_bits`, así que
tras cualquier frame de Barnes-Hut todos los dispatches de fuerza bruta usan el kernel
Barnes-Hut: física incorrecta en silencio.

## Alcance

**Haz**
- En `compute_forces_parallel_range` (`crates/gravi-core/src/parallel_dispatcher.rs`),
  publicar el kernel del frame (fuerza bruta) junto al resto de parámetros, antes de
  incrementar la época y notificar.
- Test de regresión en `tests/gravi-core-tests/src/parallel.rs`: mismo dispatcher, primero
  Barnes-Hut y después fuerza bruta, comparado bit-idéntico contra bruta secuencial.

**No hagas**
- No cambies el orden de publicación de `compute_forces_barnes_hut_parallel` (ya publica su
  kernel al final, por una razón documentada en su comentario).
- No toques la lógica de workers, epochs ni el apagado.
- No modifiques tests existentes.

### Archivos previstos
- `crates/gravi-core/src/parallel_dispatcher.rs` — modificar — publicación del kernel en el
  dispatch de bruta.
- `tests/gravi-core-tests/src/parallel.rs` — modificar — test de regresión nuevo.

## Criterios de aceptación

- [ ] **AC1** Un dispatch de Barnes-Hut seguido de un dispatch de fuerza bruta sobre el
  mismo dispatcher produce fuerzas bit-idénticas a `compute_forces_brute_range` secuencial.
  Verif: test nuevo `ac_brute_tras_barnes_hut_usa_kernel_bruto` (falla antes del fix).
- [ ] **AC2** Sin dispatch previo de Barnes-Hut, la bruta paralela sigue siendo bit-idéntica
  a la secuencial. Verif: `ac_parallel_bit_identical_to_sequential`.
- [ ] **AC3** Barnes-Hut paralelo tras bruta paralela sigue siendo bit-idéntico a
  secuencial. Verif: `ac_barnes_hut_parallel_bit_identical_to_sequential`.
- [ ] **AC4** `cargo check --workspace --all-targets` → 0 errores, 0 warnings;
  `cargo test --release parallel` y `cargo test --release` → en verde.

## Contexto

**Ya existe**
- El estado compartido tiene un campo de kernel con dos valores (0 = fuerza bruta,
  1 = Barnes-Hut); cada worker lo lee una sola vez al despertar y con él decide qué cálculo
  ejecutar sobre su bloque.
- `compute_forces_barnes_hut_parallel` publica el kernel en último lugar, después del
  puntero al arena y los parámetros, por sincronización con los workers.

**Contratos que debes respetar**: publicación del kernel por dispatch y cero asignaciones,
definidos en `../PHASE.md#contratos-compartidos`.

**Ojo**
- El worker sólo se despierta cuando cambia la época. Un dispatch que no publica el kernel
  reutiliza el valor del dispatch anterior: ese es exactamente el fallo. Al publicarlo, hazlo
  con el mismo orden de memoria (`Release`) que el resto de parámetros del frame y antes del
  bloque que incrementa la época.

### Leer primero
- `crates/gravi-core/src/parallel_dispatcher.rs` — la estructura de parámetros compartidos y
  los dos métodos de dispatch, con sus comentarios de sincronización.
- `tests/gravi-core-tests/src/parallel.rs` — pruebas de bit-identidad existentes y sus
  fixtures (`make_store_bh`).

## Especificación

Tras el fix, cada dispatch publica los tres elementos del frame de forma consistente:
rango, store con sus parámetros (`g`, y en su caso arena, `softening` y `theta`) e
identidad del kernel, todo antes del incremento de época que despierta a los workers.

| Dispatch | Kernel publicado | Efecto en los workers |
|-----------|------------------|-----------------------|
| Fuerza bruta | Fuerza bruta | Calculan bruta sobre su bloque |
| Barnes-Hut | Barnes-Hut | Calculan traversal sobre el arena del frame |

### Pasos sugeridos
1. Escribe el test AC1 y comprueba que falla con el código actual.
2. Publica el kernel en el dispatch de bruta.
3. Ejecuta los tests de `parallel` y la suite completa.

## Tests

Integración en `tests/gravi-core-tests/src/parallel.rs`. El caso nuevo encadena dos dispatches
de kernels distintos sobre un mismo dispatcher con workers reales (sin mocks) y compara con
el resultado secuencial de referencia. Reutiliza las fixtures existentes; posiciones y masas
fijas.

## Verificación

- `cargo test --release ac_brute_tras_barnes_hut_usa_kernel_bruto` → esperado: en verde
  (antes del fix, en rojo).
- `cargo test --release parallel` → esperado: todos en verde.
- `cargo test --release` → esperado: suite completa en verde.
- `cargo check --workspace --all-targets` → esperado: 0 errores, 0 warnings.

**Terminado** = AC1..AC4 cumplidos con evidencia + comandos en verde + sin cambios fuera de
alcance.

## Si te bloqueas

Si el test de bit-identidad no puede expresarse sin cambiar la firma pública del dispatcher,
para y repórtalo: no es una decisión de implementación. Decisiones que no son tuyas: cambiar
el mecanismo de sincronización, fusionar los dos caminos de dispatch.

## Cierre

- Actualiza `status` aquí y en la tabla de `PHASE.md`.
- No edites Alcance ni Criterios de aceptación (si están mal, repórtalo abajo).
- Termina con un resumen: comandos ejecutados y su resultado, AC por AC.

### Notas de implementación

Cambio: `kernel_bits.store(0, Ordering::Release)` publicado con Release antes del incremento de época en `compute_forces_parallel_range` (`crates/gravi-core/src/parallel_dispatcher.rs`), para que un dispatch de fuerza bruta no reuse el kernel=1 (Barnes-Hut) de un frame previo.

Archivos tocados:
- `crates/gravi-core/src/parallel_dispatcher.rs` — 1 línea + comentario (publicación del kernel en dispatch de bruta).
- `tests/gravi-core-tests/src/parallel.rs` — test AC1 nuevo `ac_brute_tras_barnes_hut_usa_kernel_bruto`.
- `plans/R6-rendimiento/R6-02-kernel-bits-dispatch-bruto.md` — status done + notas de implementación.
- `plans/R6-rendimiento/PHASE.md` — tabla de issues, fila R6-02: Estado todo → done.

Comandos ejecutados y resultado:
- `cargo test --release ac_brute_tras_barnes_hut_usa_kernel_bruto` → RED (test colgado/deadlock: workers reusan kernel=1 y entran en bucle infinito en traversal BH sobre `body_next` no inicializado).
- `cargo test --release ac_brute_tras_barnes_hut_usa_kernel_bruto` → GREEN (1 passed).
- `cargo test --release parallel` → todos los tests de parallel.rs en verde (AC1 nuevo + AC2 + AC3 sin modificar).
- `cargo test --release` → suite completa en verde (0 failed en todas las suites).
- `cargo check --workspace --all-targets` → 0 errores, 0 warnings.

AC1 (PASS): `ac_brute_tras_barnes_hut_usa_kernel_bruto` — bit-identidad bruta tras Barnes-Hut.
AC2 (PASS): `ac_parallel_bit_identical_to_sequential` — sin cambios.
AC3 (PASS): `ac_barnes_hut_parallel_bit_identical_to_sequential` — sin cambios.
AC4 (PASS): `cargo check --workspace --all-targets` 0 errores/warnings; `cargo test --release parallel` y `cargo test --release` en verde.