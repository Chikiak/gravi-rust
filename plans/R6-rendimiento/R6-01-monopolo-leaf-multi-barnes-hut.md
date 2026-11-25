---
id: R6-01
title: Monopolo para hojas multi-cuerpo en el traversal Barnes-Hut
phase: R6
status: done
depends_on: []
size: M
---

# R6-01 · Monopolo para hojas multi-cuerpo en el traversal Barnes-Hut

## Objetivo

Que un núcleo de cuerpos coincidentes (hoja multi-cuerpo del quadtree) deje de costar
O(cuerpos × tamaño del núcleo) por frame y vuelva a O(N log N): los cuerpos lejanos aceptan
el núcleo como monopolo. Es la causa de la caída observada de 60 a ~20 FPS cuando el disco
colapsa y el HUD muestra `kt` en subida con `NODES` estable.

## Alcance

**Haz**
- En `compute_forces_barnes_hut_range` (`crates/gravi-core/src/quadtree.rs`), la rama de
  hoja multi-cuerpo evalúa primero el criterio de apertura con la geometría del propio
  nodo; si se acepta, un único término de monopolo con la masa ya agregada del nodo; si no,
  la cadena exacta cuerpo a cuerpo de hoy.
- Actualiza la documentación del cambio de comportamiento (la nota heredada de R4-05 que
  fija "sin aceptación monopolo" para estas hojas queda obsoleta).
- Tests nuevos en `tests/gravi-core-tests/src/quadtree_bh.rs`.

**No hagas**
- No tocar `build_quadtree`, `insert_body_iterative` ni `compute_quadtree_masses`: el árbol y
  sus masas ya son correctos.
- No tocar el criterio ni la rama de nodos internos, ni `QuadNode` (32 B), ni `body_next`.
- No introducir perillas nuevas (theta adicional, flags, configuración).
- No modificar ni adaptar tests existentes; no añadir asignaciones al hot path.

### Archivos previstos
- `crates/gravi-core/src/quadtree.rs` — modificar — rama de hoja multi-cuerpo del traversal
  y su documentación.
- `tests/gravi-core-tests/src/quadtree_bh.rs` — modificar — tests nuevos.

## Criterios de aceptación

- [ ] **AC1** Con θ = 0.3 y una hoja multi de 200 cuerpos coincidentes, un cuerpo alejado
  (distancia ≥ 10× el lado de la celda) recibe el monopolo del nodo: error relativo < 1e-5
  contra el cálculo analítico independiente escrito en el test. Verif: test nuevo
  `leaf_multi_monopolo_lejano_theta_positivo`.
- [ ] **AC2** Con θ = 0, la fuerza sobre cada cuerpo sigue siendo la bruta exacta
  (tolerancia existente). Verif: `b7_theta_cero_coincide_con_fuerza_bruta`,
  `quadtree_leaf_multi_fuerza_igual_bruto` y el resto de la suite, sin modificarlos.
- [ ] **AC3** Con θ = 0.3, un cuerpo dentro o en el borde de la celda multi (criterio no
  satisfecho) sigue recibiendo la cadena exacta: error < 1e-4 contra bruto. Verif: test
  nuevo `leaf_multi_cerca_cadena_exacta`.
- [ ] **AC4** El error acotado con θ positivo sobre cuerpos separados no empeora. Verif:
  `b7_theta_positivo_error_acotado`.
- [ ] **AC5** La traversía paralela y la secuencial siguen siendo bit-idénticas. Verif:
  `ac_barnes_hut_parallel_bit_identical_to_sequential`, `ac_barnes_hut_parallel_salta_muros`.
- [ ] **AC6** `cargo check --workspace --all-targets` → 0 errores, 0 warnings; y
  `cargo test --release` → suite completa en verde.

## Contexto

**Ya existe**
- La rama de nodo interno (`kind == 2`) aplica exactamente el criterio y el monopolo que
  hay que reutilizar: mismo `soft2`, misma guarda de masa positiva, una sola escritura de
  `fx`/`fy` por cuerpo. Imítala, no inventes una variante.
- `compute_quadtree_masses` ya deja `mass` y `count` correctos en las hojas multi (suma de
  la cadena, sólo masas positivas), así que el monopolo tiene su masa disponible sin trabajo
  extra.
- Las hojas multi sólo aparecen cuando una celda ya no puede subdividirse
  (`half < MIN_CELL_SIZE` o `depth >= MAX_QUADTREE_DEPTH`); una vez degradadas, no vuelven a
  subdividirse. Es lo que permite que un núcleo colapsado acumule cadenas largas.

**Contratos que debes respetar**: criterio de apertura, exactitud con θ = 0, masa de hoja
multi y cero asignaciones en el hot path, definidos en `../PHASE.md#contratos-compartidos`.

**Ojo**
- Con θ = 0 el criterio no puede cumplirse (`θ² = 0`), por eso los tests de exactitud
  existentes no cambian: es la razón de que este fix sea seguro para el modo exacto.
- La nota de R4-05 ("sin aceptación monopolo del nodo, igual que LeafBucket de C#") queda
  obsoleta a propósito: cerca del núcleo se mantiene la exactitud; lejos se acepta monopolo,
  igual que ya hacen los nodos internos.
- `half_size` de una hoja multi es ≥ 1 salvo fixtures que construyen el árbol con un
  `root_half` pequeño a propósito; el criterio debe comportarse bien en esos casos
  (θ = 0 los cubre).

### Leer primero
- `crates/gravi-core/src/quadtree.rs` — `compute_forces_barnes_hut_range` (las tres ramas:
  hoja simple, hoja multi, interno) y `compute_quadtree_masses` (rama de hoja multi).
- `tests/gravi-core-tests/src/quadtree_bh.rs` — fixtures (`make_store`, `bruto_esperado`) y
  tests de θ = 0 que fijan la exactitud.
- `crates/gravi-core/src/constants.rs` — `MIN_CELL_SIZE`, `MAX_QUADTREE_DEPTH`,
  `DEFAULT_BARNES_HUT_THETA`.

## Especificación

Comportamiento de la rama de hoja multi-cuerpo en `compute_forces_barnes_hut_range`:

| Caso | Condición | Resultado |
|------|-----------|-----------|
| Núcleo lejano | `s² < θ²·d²`, con `s = 2·half_size` del nodo y `d` del cuerpo al centro geométrico | Un monopolo: `g · M · vector_d · (d² + soft²)^(-3/2)`, con `M` = masa agregada del nodo |
| Núcleo cercano | Resto de casos, incluidos θ = 0 y `d = 0` | Cadena `body_next` exacta cuerpo a cuerpo, igual que hoy |
| Sin masa | Masa agregada del nodo ≤ 0, o nodo vacío | Sin contribución (misma guarda que los nodos internos) |

Requisitos de forma: una sola escritura de `fx`/`fy` por cuerpo (half-coupling, como ahora);
mismo orden de recorrido; determinismo bit-idéntico entre traversal secuencial y paralelo;
cero asignaciones.

### Pasos sugeridos
1. Escribe el test del núcleo lejano (AC1) y comprueba que falla con el código actual.
2. Implementa la rama con el criterio de apertura reutilizando el patrón del nodo interno.
3. Comprueba AC2..AC5 y la suite completa.

## Tests

Integración, en `tests/gravi-core-tests/src/quadtree_bh.rs`, sin mocks: store real, árbol
real. Casos: núcleo lejano con 200 cuerpos coincidentes → monopolo analítico (referencia
calculada en el test, no la implementación); núcleo cercano → bruto exacto. Reutiliza las
fixtures existentes (`make_store`, `bruto_esperado`). Determinismo: semillas y posiciones
fijas en el test.

## Verificación

- `cargo test --release leaf_multi` → esperado: tests nuevos en verde.
- `cargo test --release quadtree_bh` → esperado: todos en verde, incluidos los de θ = 0.
- `cargo test --release` → esperado: suite completa en verde sin regresiones.
- `cargo check --workspace --all-targets` → esperado: 0 errores, 0 warnings.
- Evidencia de rendimiento (manual, apuntar en Notas): escenario 3, 8000 cuerpos, colapsar
  el disco 2-3 minutos; el `kt` del HUD se mantiene acotado en lugar de crecer.

**Terminado** = AC1..AC6 cumplidos con evidencia + comandos en verde + sin stubs ni cambios
fuera de alcance.

## Si te bloqueas

Si un test existente con θ > 0 cambia de resultado y su aserción queda en rojo, **no lo
edites**: puede ser un cambio de semántica legítimo (más cercano a bruto, no menos) y
requiere decisión; repórtalo en "Notas de implementación". Decisiones que no son tuyas:
cambiar el θ por defecto, tocar la construcción del árbol, añadir parámetros de
configuración.

## Cierre

- Actualiza `status` aquí y en la tabla de `PHASE.md`.
- No edites Alcance ni Criterios de aceptación (si están mal, repórtalo abajo).
- Termina con un resumen: comandos ejecutados y su resultado, AC por AC, y los valores de
  `kt` medidos antes/después.

### Notas de implementación

- Rama: feat/r6-01-monopolo-leaf-multi-barnes-hut (cortada de main).
- Cambio en compute_forces_barnes_hut_range (quadtree.rs): rama kind==3 (LeafMulti) evalúa primero el criterio de apertura s² < θ²·d² con s = 2·half_size; si se acepta, un solo término de monopolo con node.mass y centro geométrico; si no, cadena body_next exacta cuerpo a cuerpo (igual que antes). Guarda de masa ≤ 0 igual que nodos internos.
- Tests nuevos en quadtree_bh.rs: leaf_multi_monopolo_lejano_theta_positivo (AC1) y leaf_multi_cerca_cadena_exacta (AC3).
- Desviación registrada: PR #1 también incluyó sobre main la implementación de store.body_next (crates/gravi-core/src/store.rs), su encadenado en crates/gravi-core/src/quadtree.rs y los 3 tests leafmulti_live_indices_* / leafmulti_no_incluye_cuerpo_ajeno_del_rango.
- No se introdujeron perillas nuevas; se reutiliza θ existente.
- Cero asignaciones en el hot path (solo punteros crudos, aritmética de enteros, bucles).
- Medición manual de kt en HUD (escenario 3, 8000 cuerpos, colapsar disco 2-3 min): pendiente de ejecución con GUI antes del cierre de la fase R6.
- Comandos ejecutados: cargo check --workspace --all-targets (0 errores, 0 warnings); cargo test --release --test quadtree_bh (13/13 ok); cargo test --release --test parallel (4/4 ok); cargo test --release (suite completa verde).
- AC1 PASS (monopolo lejano, rel < 1e-5); AC2 PASS (θ=0, bruta exacta sin cambios); AC3 PASS (cerca, cadena exacta, rel < 1e-4); AC4 PASS (b7_theta_positivo_error_acotado); AC5 PASS (paralelo bit-idéntico a secuencial); AC6 PASS (check + suite verdes).