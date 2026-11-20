---
id: R4-05
title: Guardas del quadtree: MinCellSize/MaxQuadtreeDepth, clamps de raíz, clamp inserción, reportar overflow
phase: R4
status: done
branch: feat/r4-05-guardas-quadtree
depends_on: [R4-03]
size: M
findings: []
---

# R4-05 — Guardas del quadtree (paridad con GraviSharp C#)

## Objetivo

Acorrar el quadtree de Rust al de C# (`GraviSharp.Core.Quadtree`) para que el árbol esté **siempre acotado**: profundidad limitada, celdas mínimas, raíz clampada, inserción clampada al cuadrado raíz, y **reportar** el overflow del arena en vez de ignorarlo en silencio.

Hoy Rust no tiene `MinCellSize=2`, `MaxQuadtreeDepth=32`, `MinRootHalf=64`, `MaxRootHalf=4096` ni degradación controlada; y `app.rs:408` hace `let _ = build_quadtree(...)` ignorando `Err(Overflow)`, dejando el árbol a medio construir (físicas incorrectas, no solo lentas). Además, como no hay rebote (R4-01 lo añade), los cuerpos escapan y el AABB de `root_from_live_bodies` crece sin límite, haciendo el árbol cada vez más profundo y costoso.

## Alcance

**Haz**
- `crates/gravi-core/src/constants.rs`:
  - Añadir `pub const MIN_CELL_SIZE: f32 = 2.0;`, `pub const MAX_QUADTREE_DEPTH: i32 = 32;`, `pub const MIN_ROOT_HALF: f32 = 64.0;`, `pub const MAX_ROOT_HALF: f32 = 4096.0;` (paridad con `PhysicsConstants.cs:11-14` de C#).
- `crates/gravi-core/src/quadtree.rs`:
  - `root_from_live_bodies` (línea 558): clamp `half` a `[MIN_ROOT_HALF, MAX_ROOT_HALF]`; si `live_count == 0` retornar `half = MIN_ROOT_HALF` (no 0.0) para que la raíz tenga tamaño mínimo.
  - `build_quadtree` (línea 218): pasar `root_half` ya clampeado; `insert_body_iterative` recibe `depth` y detiene la subdivisión cuando `child_half < MIN_CELL_SIZE * 0.5` o `depth >= MAX_QUADTREE_DEPTH` — en vez de seguir subdividiendo hasta desbordar el arena, **degradar la hoja a contenedor multi-cuerpo** (LeafBucket-like) o simplemente parar de subdividir (la hoja acumula varios cuerpos en el mismo nodo; el traversal BH los trata individualmente si son hojas, o como monopolo si son internos). Opción más simple y segura: al llegar al límite de profundidad/tamaño, no subdividir más; el nodo actual se convierte en "hoja con varios cuerpos" y se guarda `body_index = -1, count > 1` con un marcador. Dado que `QuadNode.kind` tiene Empty=0, Leaf=1, Internal=2, añadir `LeafMulti = 3` (paridad con `LeafBucket` de C#) y un canal `body_next` (o reusar `child_offset` como cabeza de cadena) para encadenar varios cuerpos en una hoja.
  - `insert_body_iterative`: cuando una hoja recibe un segundo cuerpo y no puede subdividirse (límite de profundidad o `child_half` muy pequeño), degradar a `LeafMulti` encadenando el cuerpo existente y el nuevo (cadena ligera por índice de cuerpo, reusando un campo existente o añadiendo `body_next: i32` a `QuadNode` — evaluar impacto de 32 bytes; si se añade un campo, romper `repr(C)` 32 bytes. Alternativa: reusar `child_offset` como `body_next` cuando `kind == LeafMulti`, y `body_index` como cabeza. Eso mantiene 32 bytes).
  - `compute_quadtree_masses`: manejar `LeafMulti` sumando la cadena de cuerpos.
  - `compute_forces_barnes_hut_range`: manejar `LeafMulti` recorriendo la cadena de cuerpos como hojas (sin aceptación monopolo del nodo, igual que LeafBucket de C#).
  - `build_quadtree`: retornar `Err(Overflow)` solo si realmente no hay nodos; el llamante (`app.rs`) debe **reportar** el error (log/consola) en vez de `let _ =`.
- `crates/gravi-ui/src/app.rs` (línea 408): cambiar `let _ = build_quadtree(...)` a manejar el `Result`: si `Err(Overflow)`, imprimir advertencia por consola una vez (no por frame) y continuar con el árbol parcial (o saltar BH este frame).

**No hagas**
- No romper `QuadNode` = 32 bytes (`repr(C)`). Si se añade estado, reusar campos existentes.
- No cambiar la firma pública de `build_quadtree`/`compute_quadtree_masses`/`compute_forces_barnes_hut_range`.
- No añadir asignaciones por frame.

### Archivos previstos
- `crates/gravi-core/src/constants.rs`
- `crates/gravi-core/src/quadtree.rs`
- `crates/gravi-core/src/simulation_store.rs` (si se toca `reflect_bounds` aquí — no, es R4-01; dejar sin cambios)
- `crates/gravi-ui/src/app.rs`
- `tests/gravi-core-tests/src/quadtree_build.rs`, `tests/gravi-core-tests/src/quadtree_bh.rs`

## Criterios de aceptación

- **AC1:** Test `quadtree_root_clamp_min_max_half`: `root_from_live_bodies` con cuerpos en una esquina (extent pequeño) retorna `half >= MIN_ROOT_HALF`; con cuerpos en extremos opuestos de pantalla (1920×980) retorna `half <= MAX_ROOT_HALF`.
- **AC2:** Test `quadtree_no_desborda_arena_con_cuerpos_coincidentes`: siembra N cuerpos en la misma coordenada exacta (o dentro de MIN_CELL_SIZE), construye quadtree con arena pequeña; `build_quadtree` retorna `Ok` (no `Err(Overflow)`) y el árbol no desborda; `compute_forces_barnes_hut_range` no hace panic ni escribe fuera de rango.
- **AC3:** Test `quadtree_leaf_multi_masa_acumulada`: hoja con 3 cuerpos en la misma celda (no subdivide por límite de profundidad); `compute_quadtree_masses` da `mass = suma` y `count = 3`; `compute_forces_barnes_hut_range` aplica fuerza de los 3 cuerpos individualmente a un cuerpo externo (resultados coinciden con brute force).
- **AC4:** `app.rs` ya no ignora `Err(Overflow)` en silencio: test de integración o build con arena justa provoca log de advertencia (verificar que compila y no usa `let _ =` para el Result de `build_quadtree`).
- **AC5:** `cargo check --workspace --all-targets` → 0 errores, 0 warnings; `cargo test --release quadtree_build quadtree_bh` → todo verde; `cargo test --release` → suite completa verde.

## Contexto

**Ya existe**
- `QuadNode` 32 bytes `repr(C)` con `kind: u32` (0=Empty, 1=Leaf, 2=Internal) en quadtree.rs:33-47.
- `QuadtreeArena::allocate` capacidad `4N + 16` potencia de 2 (línea 79-113).
- `build_quadtree` (218), `insert_body_iterative` (272), `compute_quadtree_masses` (464), `compute_forces_barnes_hut_range` (667), `root_from_live_bodies` (558).
- `app.rs:408`: `let _ = build_quadtree(&self.store, &mut self.quad_arena, cx, cy, half);` ignora el error.
- C# `Quadtree.cs:216`: `canSubdivide = childHalf >= MinCellSize * 0.5f && curDepth < MaxQuadtreeDepth`; degrada a `LeafBucket` si no; `RootFromLiveBodies` clampa `half` a `[MinRootHalf, MaxRootHalf]` (447-475); inserción clampa al cuadrado raíz (156-157).

**Contratos** (PHASE.md): árbol siempre acotado; overflow reportado, no ignorado; 32 bytes por nodo preservados.

**Ojo:** `QuadNode` tiene 4 campos de 4 bytes después de `mass` (count, child_offset, body_index, kind). Para `LeafMulti` se necesita una cadena de cuerpos. Opción A: añadir `body_next: i32` → rompe 32 bytes (5 campos × 4 = 20 + center_x/y/half/mass = 16 → 36 bytes). Opción B: reusar `child_offset` como `body_next` (cabeza de cadena) cuando `kind == LeafMulti`, y `body_index` como primer cuerpo. Pero `child_offset` es i32 y `body_index` es i32, ambos ya existen; solo hay que interpretar distinto según `kind`. Eso mantiene 32 bytes. Elegir Opción B.

## Tests

Archivos: `tests/gravi-core-tests/src/quadtree_build.rs`, `tests/gravi-core-tests/src/quadtree_bh.rs`.
Nuevos tests: `quadtree_root_clamp_min_max_half`, `quadtree_no_desborda_arena_con_cuerpos_coincidentes`, `quadtree_leaf_multi_masa_acumulada`.
Setup: `QuadtreeArena::allocate`, `NativeStore::new` + `init_lifecycle`, setear masas/posiciones, `build_quadtree`, `compute_quadtree_masses`, `compute_forces_barnes_hut_range`, verificar por nodo/cuerpo. Sin asignaciones en el bucle de aserción.

## Verificación

Ejecutar desde `D:\Proyectos\GS-Folder\gravi-rust\`:
1. `cargo check --workspace --all-targets` → Finished, 0 errores, 0 warnings.
2. `cargo test --release quadtree_build quadtree_bh` → verde.
3. `cargo test --release` → suite completa verde.

## Cierre

Pendiente al cerrar en verde.

### Notas de implementación
- Rama: `feat/r4-05-guardas-quadtree`
- Commit: 5526f04a189de1058ea6bb3c494065fe2dff6258 (feat/r4-05: guardas del quadtree MinCellSize/MaxQuadtreeDepth/clamps/overflow)
- Archivos modificados: crates/gravi-core/src/constants.rs, crates/gravi-core/src/quadtree.rs, crates/gravi-ui/src/app.rs, tests/gravi-core-tests/src/quadtree_build.rs, tests/gravi-core-tests/src/quadtree_bh.rs, plans/R4-rendimiento/R4-05-guardas-quadtree.md, plans/R4-rendimiento/PHASE.md
- Desviaciones: Ninguna (se siguió el plan táctico; LeafMulti kind=3 reusa body_index/child_offset; 32 bytes preservados; cero asignaciones hot path).

**AC1 — PASS:**
  - Comando: cargo test --release quadtree_root_clamp_min_max_half
  - Resultado: PASS (root_from_live_bodies clampa half a [64,4096]; live_count==0 → MIN_ROOT_HALF)

**AC2 — PASS:**
  - Comando: cargo test --release quadtree_no_desborda_arena_con_cuerpos_coincidentes
  - Resultado: PASS (build_quadtree Ok con cuerpos coincidentes; no Overflow; forces sin panic)

**AC3 — PASS:**
  - Comando: cargo test --release quadtree_leaf_multi_masa_acumulada quadtree_leaf_multi_fuerza_igual_bruto
  - Resultado: PASS (mass=suma count=3; fuerza externo coincide brute force theta=0)

**AC4 — PASS:**
  - Comando: cargo check --workspace --all-targets + inspección app.rs (no `let _ = build_quadtree`)
  - Resultado: PASS (app.rs maneja Err(Overflow) con eprintln! una vez y continúa)

**AC5 — PASS:**
  - Comando: cargo check --workspace --all-targets && cargo test --release quadtree_build quadtree_bh && cargo test --release
  - Resultado: PASS (0 errores 0 warnings; suite verde)
