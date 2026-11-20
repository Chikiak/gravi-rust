# PHASE R4 — Rendimiento

Estado del plan de optimizaciones de rendimiento (ruta R4).

## Issues

| ID | Título | Estado |
|----|--------|--------|
| R4-01 | Tecla B toggle bounce + `reflect_bounds` en 4 paredes | done |
| R4-02 | Render descarta coordenadas negativas y fuera de rango | done |
| R4-03 | Tope de subpasos y drenaje del acumulador | done |
| R4-04 | Paralelizar traversal Barnes-Hut con el dispatcher de fuerza | done |
| R4-05 | Guardas del quadtree: MinCellSize/MaxQuadtreeDepth, clamps de raíz, clamp inserción, reportar overflow | done |

## Execution Log

- R4-01: merge a `main` (commit 609ac01).
- R4-02: merge a `main` (commit 75b0404).
- R4-03: merge a `main` (commit eba940e).
- R4-04: implementado en rama `feat/r4-04-paralelizar-traversal-barnes-hut`,
  merge `--no-ff` a `main` tras verificación verde.
- R4-05: implementado en rama `feat/r4-05-guardas-quadtree`
  (commit 5526f04), verificación AC1..AC5 verde.
