---
phase: R6
title: FPS estables con núcleos colapsados (hot path Barnes-Hut)
status: todo            # todo | in_progress | done
master: —
---

# R6 · FPS estables con núcleos colapsados

## Objetivo y entregable

Con 8000 cuerpos, el coste por frame deja de depender de cuánto se haya apretado el núcleo
gravitacional: los FPS se mantienen aunque la simulación colapse, y el alternado de motores
de fuerza (tecla E) vuelve a ser correcto. Demostrable con tests deterministas más una
medición manual de `kt` en el HUD antes y después de colapsar el disco.

## Estado de partida

Hechos verificados leyendo el repo y con la medición del usuario (release, escenario 3,
Barnes-Hut + rebote, sin colisiones):

- El HUD muestra `kt` en subida mientras `NODES` (línea 2, valor `quad_arena.next_free`) se
  mantiene: el coste crece en el **traversal**, no en la construcción del árbol.
- `crates/gravi-core/src/quadtree.rs`: la rama `kind == 3` (hoja multi-cuerpo) de
  `compute_forces_barnes_hut_range` recorre la cadena `body_next` **siempre** cuerpo a
  cuerpo, sin criterio de apertura; la rama `kind == 2` (Internal) sí acepta monopolo con
  `s = 2·half_size` y `s² < θ²·d²`. Las hojas multi nacen cuando una celda no puede
  subdividirse más (`half < MIN_CELL_SIZE` o `depth >= MAX_QUADTREE_DEPTH`) y ya no se
  subdividen nunca.
- `crates/gravi-core/src/parallel_dispatcher.rs`: `compute_forces_parallel_range` (bruta)
  no publica `kernel_bits`; `compute_forces_barnes_hut_parallel` lo publica en 1. El worker
  lo lee una vez al despertar.
- `crates/gravi-core/src/simulation_store.rs`: `compute_forces_brute_range` cae a escalar
  O(N²) cuando `live_count != count`.
- `crates/gravi-ui/src/app.rs`: `fixed_timestep_substeps` con tope de 2 subpasos y recorte
  del acumulador a 0.25 s; `kt_ms` mide todo el bloque de física del frame.
- Comandos vigentes en el repo (vistos en `plans/R4-rendimiento/R4-05-guardas-quadtree.md`):
  `cargo check --workspace --all-targets` y `cargo test --release` desde la raíz del
  workspace; los tests de integración viven en `tests/gravi-core-tests/src/` y los de UI en
  `crates/gravi-ui/tests/`.
- No hay `MASTER.md` en el repo: este `PHASE.md` es autocontenido (supuesto registrado).

## Contratos compartidos

- **Criterio de apertura BH** (ya existe para nodos Internal): monopolo aceptado si
  `s² < θ²·d²`, con `s = 2·half_size` del nodo y `d` = distancia del cuerpo al centro
  geométrico del nodo. Decisión de fase: las hojas multi-cuerpo usan **el mismo** criterio.
- **Exactidad con θ = 0**: `θ² = 0` hace el criterio insatisfacible, la hoja multi se
  recorre exactamente y ningún test de exactitud existente cambia. Es la garantía de que el
  fix no rompe la paridad de comportamiento exacto.
- **Masa de hoja multi**: la que deja `compute_quadtree_masses` (suma de la cadena, sólo
  masas positivas). El monopolo usa esa masa y el centro geométrico de la celda.
- **Kernel del dispatcher**: `kernel_bits` = 0 fuerza bruta, 1 Barnes-Hut; se publica en
  cada dispatch antes de incrementar la época, junto al resto de parámetros del frame.
- **Hot path**: cero asignaciones; sin `Vec`, `String`, `format!`, `Box` ni closures que
  capturen; sólo punteros crudos, aritmética de enteros y bucles.
- **Presupuesto de frame de R6-03**: duración del frame anterior en ms; la política de
  subpasos se activa cuando el `kt` medido del frame anterior lo supera (comparación
  estricta).

## Decisiones de la fase

- IDs `R6-nn` y carpeta `R6-rendimiento/` — manda la convención del repo
  (`plans/R4-rendimiento/`), no la plantilla genérica.
- R6-01 reutiliza el criterio θ existente en vez de añadir un parámetro nuevo — porque evita
  introducir perillas invisibles y mantiene el mismo comportamiento cerca/lejos que ya
  tienen los nodos internos.
- R6-03: si la física se pasa del presupuesto, 1 subpaso y drenaje del acumulador a `dt`
  (simulación en cámara lenta con app fluida) — elegido por el usuario frente a mantener
  hasta 2 subpasos. El fallback escalar visible y el test de corrección viajan en el mismo
  issue porque comparten `app.rs`.
- Reescribir el kernel escalar enmascarado queda en Pendientes (no es el síntoma actual y
  cambiaría resultados bit a bit).

## Issues

| ID | Título | Depende de | Tamaño | Paralelizable con | Estado |
|----|--------|------------|--------|-------------------|--------|
| R6-01 | Monopolo para hojas multi-cuerpo en el traversal Barnes-Hut | — | M | R6-02, R6-03 | done |
| R6-02 | El dispatch de fuerza bruta publica su kernel | — | S | R6-01, R6-03 | todo |
| R6-03 | Visibilidad del fallback escalar y subpasos sin realimentación | — | M | R6-01, R6-02 | todo |

### Orden y dependencias

```mermaid
graph LR
```

Ninguna dependencia técnica real: los tres issues tocan archivos disjuntos y se pueden ejecutar
en cualquier orden. Orden recomendado R6-02 → R6-01 → R6-03: R6-02 es el cambio más pequeño
y quita un bug de corrección silencioso; R6-01 es el fix del síntoma reportado y el de mayor
valor; R6-03 agrupa dos asuntos menores que comparten `app.rs`.

## Cobertura

| Entregable / criterio de salida de la fase | Issues que lo cubren |
|--------------------------------------------|----------------------|
| `kt` acotado con el núcleo colapsado (FPS estables) | R6-01 |
| Correcto al alternar motor bruta/Barnes-Hut con E | R6-02 |
| El modo bruta con slots muertos es visible y correcto | R6-03 |
| Un frame caro no provoca caída sostenida de FPS | R6-03 |

## Verificación de salida de la fase

- `cargo check --workspace --all-targets` → esperado: 0 errores, 0 warnings.
- `cargo test --release` → esperado: suite completa en verde.
- Medición manual (pasos en R6-01): escenario 3, 8000 cuerpos, dejar colapsar el disco 2-3
  minutos; el `kt` del HUD se mantiene acotado en vez de crecer con el tiempo.
- [ ] Los FPS se mantienen estables tras el colapso (sin la caída 60 → ~20 observada).
- [ ] Sesión de verificación independiente recomendada: un agente distinto comprueba cada
  criterio contra el código y los comandos antes de dar la fase por cerrada.

## Prompt de sesión (copiar/pegar para cada issue)

```text
Implementa el issue `plan/R6-rendimiento/<ID>.md`.
Léelo entero antes de tocar código y empieza por su sección "Leer primero".
Ejecuta los comandos de "Verificación" y no lo des por terminado hasta cubrir cada
criterio de aceptación con evidencia. Al acabar: actualiza `status` y rellena
"Notas de implementación". Si el issue contradice el repo o te bloquea, para y repórtalo
en lugar de improvisar.
```

## Registro de ejecución

_(vacío: se rellena al avanzar)_

## Pendientes (fuera de esta fase)

- Kernel de fuerza bruta enmascarado para `live_count != count` (hoy cae a escalar O(N²)):
  cambiaría resultados bit a bit y no es el síntoma reportado; replanificar si se activan
  colisiones con fusión.
- Política de subpasos configurable (hoy umbral fijo) si se quiere ajustar sin recompilar.