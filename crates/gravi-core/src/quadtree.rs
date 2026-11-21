//! quadtree.rs — Quadtree plano para Barnes-Hut (Issue 6, parte A.1).
//!
//! Estructuras de datos:
//! - `QuadNode`: nodo de 32 bytes, `repr(C)`, disposición plana (POD).
//! - `QuadtreeArena`: arena preasignada (capacidad = potencia de 2 >= 4N+16,
//!   alineada 64 B), bump-allocator O(1), sin recursión, zero-allocation en
//!   el hot path.
//!
//! Invariantes del hot path:
//! - Sin `Vec`, `String`, `format!`, `Box`, closures que capturen ni LINQ.
//! - Sólo punteros crudos, aritmética de enteros y bucles `while`/`for`.
//! - Construcción iterativa (sin recursión).

use crate::constants::{MAX_QUADTREE_DEPTH, MAX_ROOT_HALF, MIN_CELL_SIZE, MIN_ROOT_HALF};
use crate::store::NativeStore;
use std::alloc::{alloc, dealloc, Layout};
use std::ptr::NonNull;

// ============================================================================
// QuadNode — 32 bytes, repr(C)
// ============================================================================

/// Nodo de quadtree plano: exactamente 32 bytes, disposition `repr(C)`.
///
/// Offsets (cada campo 4 B):
///   0  center_x   f32  centro X de la celda
///   4  center_y   f32  centro Y de la celda
///   8  half_size  f32  semilado de la celda
///   12 mass       f32  masa acumulada de la celda (cuerpos + descendencia)
///   16 count      i32  número de cuerpos en la celda (0 si vacía)
///   20 child_offset i32 — índice del primer hijo si `kind == Internal` (-1 si hoja)
///   24 body_index  i32 — índice del cuerpo si `kind == Leaf` y un solo cuerpo (-1 si no)
///   28 kind       u32  0=Empty, 1=Leaf, 2=Internal, 3=LeafMulti (cabeza de cadena:
///                      body_index = cabeza de la cadena cuerpo-a-cuerpo,
///                      child_offset = cola informativa de la cadena,
///                      count = nº de cuerpos en la hoja multi; los enlaces
///                      cuerpo-a-cuerpo viven en store.body_next, no en slots
///                      consecutivos del arena).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuadNode {
    pub center_x: f32,
    pub center_y: f32,
    pub half_size: f32,
    pub mass: f32,
    pub count: i32,
    pub child_offset: i32,
    pub body_index: i32,
    pub kind: u32,
}

/// Constante de compilación: `QuadNode` debe ocupar exactamente 32 bytes.
const _: () = assert!(core::mem::size_of::<QuadNode>() == 32, "QuadNode debe ser 32 bytes");

// ============================================================================
// QuadtreeArena
// ============================================================================

/// Arena preasignada de nodos de quadtree.
///
/// Capacidad = potencia de 2 >= `4 * n_bodies + 16`. Asignación única (bump
/// allocator): `next_free` avanza monotonicamente; reiniciar a 0 es O(1) por
/// frame. No hay recursión: build y traversal usan bucles explícitos.
#[repr(C)]
#[derive(Debug)]
pub struct QuadtreeArena {
    pub base: *mut QuadNode,
    pub capacity: usize,
    pub next_free: usize,
    pub root_index: i32,
}

/// Errores del arena de quadtree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuadtreeError {
    ZeroCapacity,
    AllocationFailed,
    Overflow, // next_free >= capacity
}

impl QuadtreeArena {
    /// Reserva un arena con capacidad >= 4*N + 16 (redondeada a potencia de 2),
    /// alineada a 64 B. Todos los nodos inician en `kind = Empty`.
    /// No es hot path (se llama una vez por escenario).
    pub fn allocate(n_bodies: usize) -> Result<Self, QuadtreeError> {
        if n_bodies == 0 {
            return Err(QuadtreeError::ZeroCapacity);
        }
        let needed = 4usize.saturating_mul(n_bodies).saturating_add(16);
        let capacity = next_power_of_two(needed.max(16));
        let byte_size = capacity * core::mem::size_of::<QuadNode>();
        let layout = match Layout::from_size_align(byte_size, 64) {
            Ok(l) => l,
            Err(_) => return Err(QuadtreeError::AllocationFailed),
        };
        let ptr = unsafe { alloc(layout) };
        let base = match NonNull::new(ptr as *mut QuadNode) {
            Some(p) => p,
            None => return Err(QuadtreeError::AllocationFailed),
        };

        // Inicializar nodos a Empty (no es hot path).
        unsafe {
            std::ptr::write_bytes(base.as_ptr(), 0, capacity);
            for i in 0..capacity {
                *base.as_ptr().add(i) = QuadNode {
                    kind: 0, // Empty
                    ..core::mem::zeroed()
                };
            }
        }

        Ok(Self {
            base: base.as_ptr(),
            capacity,
            next_free: 0,
            root_index: -1,
        })
    }

    /// Libera el arena. No es hot path (se llama al salir).
    pub unsafe fn dispose(arena: &mut Self) {
        if arena.base.is_null() {
            return;
        }
        let byte_size = arena.capacity * core::mem::size_of::<QuadNode>();
        let layout = match Layout::from_size_align(byte_size, 64) {
            Ok(l) => l,
            Err(_) => return,
        };
        unsafe {
            dealloc(arena.base as *mut u8, layout);
        }
        arena.base = core::ptr::null_mut();
        arena.capacity = 0;
        arena.next_free = 0;
        arena.root_index = -1;
    }

    /// Reserva un nodo (bump allocator, O(1)). Retorna índice o `Err(Overflow)`.
    #[inline(always)]
    pub fn alloc_node(&mut self) -> Result<usize, QuadtreeError> {
        if self.next_free >= self.capacity {
            return Err(QuadtreeError::Overflow);
        }
        let idx = self.next_free;
        self.next_free += 1;
        Ok(idx)
    }

    /// Referencia mutable al nodo `idx` (índice válido, caller garantiza).
    #[inline(always)]
    pub unsafe fn node_mut(&mut self, idx: usize) -> &mut QuadNode {
        unsafe { &mut *self.base.add(idx) }
    }

    /// Referencia al nodo `idx` (índice válido, caller garantiza).
    #[inline(always)]
    pub unsafe fn node(&self, idx: usize) -> &QuadNode {
        unsafe { &*self.base.add(idx) }
    }

    /// Escribe un nodo plano en el slot `idx` usando puntero crudo (evita
    /// conflictos del borrow checker al combinar con `alloc_node`).
    #[inline(always)]
    pub unsafe fn write_node(&mut self, idx: usize, node: QuadNode) {
        unsafe {
            std::ptr::write(self.base.add(idx), node);
        }
    }

    /// Lee un nodo plano del slot `idx` usando puntero crudo.
    #[inline(always)]
    pub unsafe fn read_node(&self, idx: usize) -> QuadNode {
        unsafe { std::ptr::read(self.base.add(idx)) }
    }
}

impl Drop for QuadtreeArena {
    fn drop(&mut self) {
        if !self.base.is_null() {
            unsafe { Self::dispose(self) };
        }
    }
}

// ============================================================================
// Construcción iterativa del quadtree (sin recursión)
// ============================================================================

/// Cuadrante en que cae un punto respecto al centro de la celda.
/// 0=NO (x<cx, y<cy), 1=NE (x>=cx, y<cy), 2=SO (x<cx, y>=cy), 3=SE (x>=cx, y>=cy).
#[inline(always)]
fn quadrant_of(px: f32, py: f32, cx: f32, cy: f32) -> usize {
    let mut q: usize = 0;
    if px >= cx {
        q |= 1;
    }
    if py >= cy {
        q |= 2;
    }
    q
}

/// Centro y semilado del hijo `quadrant` de una celda.
#[inline(always)]
fn child_cell(cx: f32, cy: f32, half: f32, quadrant: usize) -> (f32, f32, f32) {
    let hw = half / 2.0;
    let dx = if quadrant & 1 == 1 { hw } else { -hw };
    let dy = if quadrant & 2 == 2 { hw } else { -hw };
    (cx + dx, cy + dy, hw)
}

/// Construye el quadtree sobre los cuerpos vivos del `store`.
///
/// Reinicia el arena (`next_free = 0`, `root_index = -1`), crea la raíz e
/// inserta cada cuerpo vivo iterativamente (sin recursión). Cuando una hoja
/// recibe un segundo cuerpo, se subdivide en 4 hijos y se redistribuyen.
///
/// Zero-allocation en hot path: bump allocator, ningún `Vec`/`String`/`Box`.
///
/// Retorna `Err(Overflow)` si el arena se llena (debería no ocurrir con
/// capacidad 4N+16 salvo patologías de muchas celdas unitarias).
pub fn build_quadtree(
    store: &NativeStore,
    arena: &mut QuadtreeArena,
    root_cx: f32,
    root_cy: f32,
    root_half: f32,
) -> Result<(), QuadtreeError> {
    unsafe {
        arena.next_free = 0;
        arena.root_index = -1;

        // Reset de la cadena body_next (reusa el buffer existente,
        // zero-allocation: no asigna, solo escribe -1 en cada slot).
        {
            let bn = store.body_next;
            let mut k = 0usize;
            while k < store.count {
                *bn.add(k) = -1i32;
                k += 1;
            }
        }

        let live_count = store.live_count;
        if live_count == 0 {
            return Ok(());
        }

        let x = store.x;
        let y = store.y;
        let mass = store.mass;
        let body_next = store.body_next;
        let live_indices = store.live_indices;

        // Nodo raíz.
        let root_idx = arena.alloc_node()?;
        arena.root_index = root_idx as i32;
        arena.write_node(
            root_idx,
            QuadNode {
                center_x: root_cx,
                center_y: root_cy,
                half_size: root_half,
                mass: 0.0,
                count: 0,
                child_offset: -1,
                body_index: -1,
                kind: 0, // Empty (la primera inserción la promueve a Leaf)
            },
        );

        for p in 0..live_count {
            let body_idx = *live_indices.add(p) as usize;
            let bx = *x.add(body_idx);
            let by = *y.add(body_idx);
            let bm = *mass.add(body_idx);
            insert_body_iterative(
                arena, root_idx, body_idx, bx, by, bm,
                root_cx, root_cy, root_half, 0, x, y, mass, body_next,
            )?;
        }
        Ok(())
    }
}

/// Inserción iterativa de un cuerpo descendiendo desde `node_idx` hasta una hoja.
///
/// `depth` es la profundidad actual del nodo (raíz = 0). Cuando una hoja recibe
/// un segundo cuerpo distinto pero no puede subdividirse
/// (`child_half < MIN_CELL_SIZE * 0.5` o `depth >= MAX_QUADTREE_DEPTH`), se
/// degrada a `LeafMulti` (kind=3): `body_index` es la cabeza de la cadena de
/// cuerpos, `child_offset` es la cola informativa de la cadena, `count` es el
/// número de cuerpos y `mass` la suma. Los enlaces cuerpo-a-cuerpo viven en
/// `store.body_next` (no se asumen slots consecutivos). No se consumen nodos
/// extra, por lo que no genera `Overflow` por cuerpos coincidentes.
#[inline(always)]
fn insert_body_iterative(
    arena: &mut QuadtreeArena,
    mut node_idx: usize,
    body_idx: usize,
    bx: f32,
    by: f32,
    bm: f32,
    mut cx: f32,
    mut cy: f32,
    mut half: f32,
    depth: i32,
    x: *mut f32,
    y: *mut f32,
    mass: *mut f32,
    body_next: *mut i32,
) -> Result<(), QuadtreeError> {
    loop {
        let (node, kind) = unsafe {
            let n = arena.read_node(node_idx);
            (n, n.kind)
        };
        let new_mass = node.mass + bm;
        let new_count = node.count + 1;

        match kind {
            0 => { // Empty -> Leaf con este cuerpo
                unsafe {
                    arena.write_node(
                        node_idx,
                        QuadNode {
                            center_x: cx,
                            center_y: cy,
                            half_size: half,
                            mass: new_mass,
                            count: new_count,
                            child_offset: -1,
                            body_index: body_idx as i32,
                            kind: 1, // Leaf
                        },
                    );
                }
                break;
            }
            1 => { // Leaf -> subdividir o degradar a LeafMulti
                let existing = node.body_index as usize;
                if existing == body_idx {
                    // mismo cuerpo ya está aquí; solo actualizar masa/count
                    unsafe {
                        arena.write_node(
                            node_idx,
                            QuadNode {
                                center_x: cx,
                                center_y: cy,
                                half_size: half,
                                mass: new_mass,
                                count: new_count,
                                child_offset: -1,
                                body_index: node.body_index,
                                kind: 1,
                            },
                        );
                    }
                    break;
                }
                // ¿Puede subdividir? child_half = half/2; exige child_half >=
                // MIN_CELL_SIZE*0.5 (equiv. half >= MIN_CELL_SIZE) y profundidad
                // por debajo del tope.
                let can_subdivide = half >= MIN_CELL_SIZE && depth < MAX_QUADTREE_DEPTH;
                if !can_subdivide {
                    // Degradar a LeafMulti: encadenar el cuerpo existente y el
                    // nuevo en este mismo nodo (no consume nodos extra).
                    // body_index = cabeza (cuerpo previo), child_offset = cola
                    // informativa (cuerpo nuevo); la cadena cuerpo-a-cuerpo vive
                    // en store.body_next.
                    unsafe {
                        arena.write_node(
                            node_idx,
                            QuadNode {
                                center_x: cx,
                                center_y: cy,
                                half_size: half,
                                mass: new_mass,
                                count: new_count,
                                child_offset: body_idx as i32, // cola informativa
                                body_index: existing as i32,   // cabeza de cadena
                                kind: 3,                       // LeafMulti
                            },
                        );
                        *body_next.add(existing) = body_idx as i32;
                        *body_next.add(body_idx) = -1i32;
                    }
                    break;
                }
                // Subdividir en 4 hijos contiguos (reservar slots primero, sin
                // mantener ninguna referencia mutable al arena).
                // Reservar 4 slots contiguos para los hijos (child_base..child_base+4).
                // alloc_node avanza 1; se avanzan los 3 restantes para que la
                // siguiente alloc_node no sobrescriba los hijos ya escritos.
                let child_base = arena.alloc_node()?;
                if arena.next_free + 3 > arena.capacity {
                    arena.next_free = child_base; // deshacer el alloc_node ya consumido
                    return Err(QuadtreeError::Overflow);
                }
                arena.next_free = child_base + 4;
                for k in 0..4usize {
                    unsafe {
                        arena.write_node(
                            child_base + k,
                            QuadNode {
                                kind: 0, // Empty
                                ..core::mem::zeroed()
                            },
                        );
                    }
                }

                // Marcar actual como interno.
                unsafe {
                    arena.write_node(
                        node_idx,
                        QuadNode {
                            center_x: cx,
                            center_y: cy,
                            half_size: half,
                            mass: new_mass,
                            count: new_count,
                            child_offset: child_base as i32,
                            body_index: -1,
                            kind: 2, // Internal
                        },
                    );
                }

                // Re-insertar el cuerpo existente y el nuevo, descendiendo a
                // sus respectivos hijos (insert_body_iterative maneja
                // Empty→Leaf y Leaf→subdivide si chocan en el mismo cuadrante).
                let (ex_x, ex_y, ex_m) = unsafe {
                    (*x.add(existing), *y.add(existing), *mass.add(existing))
                };

                let q_ex = quadrant_of(ex_x, ex_y, cx, cy);
                let (ecx, ecy, ehalf) = child_cell(cx, cy, half, q_ex);
                insert_body_iterative(
                    arena,
                    child_base + q_ex,
                    existing,
                    ex_x,
                    ex_y,
                    ex_m,
                    ecx,
                    ecy,
                    ehalf,
                    depth + 1,
                    x,
                    y,
                    mass,
                    body_next,
                )?;

                let q_new = quadrant_of(bx, by, cx, cy);
                let (ncx, ncy, nhalf) = child_cell(cx, cy, half, q_new);
                insert_body_iterative(
                    arena, child_base + q_new, body_idx, bx, by, bm,
                    ncx, ncy, nhalf, depth + 1, x, y, mass, body_next,
                )?;
                break;
            }
            3 => { // LeafMulti -> añadir cuerpo al final de la cadena cuerpo-a-cuerpo
                // body_index = cabeza de la cadena, child_offset = cola
                // informativa. La cadena vive en store.body_next (los slots de
                // cuerpo NO son consecutivos: live_indices es una permutación).
                unsafe {
                    let tail = node.child_offset as usize;
                    *body_next.add(tail) = body_idx as i32;
                    *body_next.add(body_idx) = -1i32;
                    arena.write_node(
                        node_idx,
                        QuadNode {
                            mass: new_mass,
                            count: new_count,
                            child_offset: body_idx as i32, // nueva cola informativa
                            ..node
                        },
                    );
                }
                break;
            }
            2 => { // Internal -> acumular masa/count y descender al hijo del cuadrante
                unsafe {
                    arena.write_node(
                        node_idx,
                        QuadNode {
                            center_x: cx,
                            center_y: cy,
                            half_size: half,
                            mass: new_mass,
                            count: new_count,
                            child_offset: node.child_offset,
                            body_index: node.body_index,
                            kind: 2,
                        },
                    );
                }
                let q = quadrant_of(bx, by, cx, cy);
                let child_idx = (node.child_offset as usize) + q;
                let (ncx, ncy, nhalf) = child_cell(cx, cy, half, q);
                cx = ncx;
                cy = ncy;
                half = nhalf;
                node_idx = child_idx;
            }
            _ => break, // kind inválido, no avanzar
        }
    }
    Ok(())
}

// ============================================================================
// Masas en post-orden y raiz desde cuerpos vivos (Issue 6, parte A.2)
// ============================================================================

/// Frame de la pila de post-orden: nodo a procesar y siguiente hijo (0..4).
#[derive(Clone, Copy)]
struct PostOrderFrame {
    node_idx: usize,
    next_child: u8,
}

/// Capacidad de la pila de post-orden en stack (cubre la profundidad práctica
/// del quadtree; sin recursion ni Vec).
const POST_ORDER_STACK_CAP: usize = 256;

/// Calcula masa y count de todos los nodos en post-orden (iterativo, pila en
/// stack, sin recursion).
///
/// - `Empty`: mass = 0, count = 0 (sin cambios).
/// - `Leaf` (cuerpo): mass = masa del cuerpo, count = 1.
/// - `Internal`: mass = suma de masas de los 4 hijos; count = suma de counts.
///
/// Escribe `mass` y `count` en cada nodo del arena. No toca `center_x` /
/// `center_y` (centro geometrico de la celda, no centro de masa).
///
/// Zero-allocation: pila fija en stack, sin `Vec`/`String`/`format!`/`Box` ni
/// closures. Retorna `Ok(())` incluso si `root_index < 0` (arbol vacio).
#[inline(always)]
pub fn compute_quadtree_masses(
    arena: &mut QuadtreeArena,
    store: &NativeStore,
) -> Result<(), QuadtreeError> {
    unsafe {
        let root = arena.root_index;
        if root < 0 {
            return Ok(());
        }
        let x = store.x;
        let y = store.y;
        let mass = store.mass;
        let body_next = store.body_next;

        let mut stack: [PostOrderFrame; POST_ORDER_STACK_CAP] =
            [PostOrderFrame { node_idx: 0, next_child: 0 }; POST_ORDER_STACK_CAP];
        let mut sp: usize = 0;
        stack[sp] = PostOrderFrame {
            node_idx: root as usize,
            next_child: 0,
        };
        sp += 1;

        while sp > 0 {
            sp -= 1;
            let frame = stack[sp];
            let node = arena.read_node(frame.node_idx);

            match node.kind {
                0 => { // Empty: nada que acumular (mass/count ya son 0)
                }
                1 => { // Leaf con un cuerpo: masa y count del cuerpo
                    let body_idx = node.body_index as usize;
                    let m = *mass.add(body_idx);
                    arena.write_node(
                        frame.node_idx,
                        QuadNode {
                            mass: m,
                            count: 1,
                            ..node
                        },
                    );
                }
                3 => { // LeafMulti: recorre la cadena cuerpo a cuerpo (body_next)
                    let mut acc_mass: f32 = 0.0;
                    let mut acc_count: i32 = 0;
                    let mut j = node.body_index as usize;
                    loop {
                        let m = *mass.add(j);
                        if m > 0.0 {
                            acc_mass = acc_mass + m;
                            acc_count = acc_count + 1;
                        }
                        let nxt = *body_next.add(j);
                        if nxt == -1i32 {
                            break;
                        }
                        j = nxt as usize;
                    }
                    arena.write_node(
                        frame.node_idx,
                        QuadNode {
                            mass: acc_mass,
                            count: acc_count,
                            ..node
                        },
                    );
                }
                2 => { // Internal: acumular de hijos (post-orden)
                    if frame.next_child < 4 {
                        // Volver a este nodo con el siguiente hijo, y procesar hijo actual.
                        stack[sp] = PostOrderFrame {
                            node_idx: frame.node_idx,
                            next_child: frame.next_child + 1,
                        };
                        sp += 1;
                        let child_idx = (node.child_offset as usize) + frame.next_child as usize;
                        stack[sp] = PostOrderFrame {
                            node_idx: child_idx,
                            next_child: 0,
                        };
                        sp += 1;
                    } else {
                        // Todos los hijos ya procesados: acumular.
                        let child_offset = node.child_offset as usize;
                        let mut acc_mass: f32 = 0.0;
                        let mut acc_count: i32 = 0;
                        let mut k: usize = 0;
                        while k < 4usize {
                            let ch = arena.read_node(child_offset + k);
                            acc_mass = acc_mass + ch.mass;
                            acc_count = acc_count + ch.count;
                            k += 1;
                        }
                        arena.write_node(
                            frame.node_idx,
                            QuadNode {
                                mass: acc_mass,
                                count: acc_count,
                                ..node
                            },
                        );
                    }
                }
                _ => {} // kind invalido, ignorar
            }
        }
        Ok(())
    }
}

/// Calcula el cuadrado raiz que cubre el AABB de los cuerpos vivos.
///
/// Retorna `(cx, cy, half)` donde `(cx, cy)` es el punto medio del AABB (por
/// tanto dentro del rectangulo de vivos) y `half = max(extent_x, extent_y) / 2`.
///
/// Si no hay cuerpos vivos, retorna `(0.0, 0.0, 0.0)`.
///
/// Zero-allocation: solo punteros crudos, aritmética de enteros y bucle `while`.
#[inline(always)]
pub fn root_from_live_bodies(store: &NativeStore) -> (f32, f32, f32) {
    unsafe {
        let live_count = store.live_count;
        if live_count == 0 {
            return (0.0, 0.0, MIN_ROOT_HALF);
        }
        let x = store.x;
        let y = store.y;
        let live_indices = store.live_indices;

        let first = *live_indices.add(0) as usize;
        let mut min_x = *x.add(first);
        let mut max_x = min_x;
        let mut min_y = *y.add(first);
        let mut max_y = min_y;

        let mut p: usize = 1;
        while p < live_count {
            let idx = *live_indices.add(p) as usize;
            let px = *x.add(idx);
            let py = *y.add(idx);
            if px < min_x {
                min_x = px;
            }
            if px > max_x {
                max_x = px;
            }
            if py < min_y {
                min_y = py;
            }
            if py > max_y {
                max_y = py;
            }
            p += 1;
        }

        let cx = (min_x + max_x) * 0.5;
        let cy = (min_y + max_y) * 0.5;
        let extent_x = max_x - min_x;
        let extent_y = max_y - min_y;
        let mut half = if extent_x > extent_y {
            extent_x * 0.5
        } else {
            extent_y * 0.5
        };
        if half < MIN_ROOT_HALF {
            half = MIN_ROOT_HALF;
        } else if half > MAX_ROOT_HALF {
            half = MAX_ROOT_HALF;
        }
        (cx, cy, half)
    }
}

/// Inserta un cuerpo en el hijo adecuado de `child_base` (hoja vacía).
#[inline(always)]
fn insert_into_child(
    arena: &mut QuadtreeArena,
    child_base: usize,
    cx: f32,
    cy: f32,
    half: f32,
    body_idx: usize,
    bx: f32,
    by: f32,
    bm: f32,
    x: *mut f32,
    y: *mut f32,
    mass: *mut f32,
) -> Result<(), QuadtreeError> {
    let q = quadrant_of(bx, by, cx, cy);
    let (ncx, ncy, nhalf) = child_cell(cx, cy, half, q);
    let child_idx = child_base + q;
    unsafe {
        arena.write_node(
            child_idx,
            QuadNode {
                center_x: ncx,
                center_y: ncy,
                half_size: nhalf,
                mass: bm,
                count: 1,
                child_offset: -1,
                body_index: body_idx as i32,
                kind: 1, // Leaf
            },
        );
    }
    Ok(())
}

// ============================================================================
// Fuerzas Barnes-Hut (Issue 7, parte B)
// ============================================================================

/// Capacidad de la pila de traversal en stack (cubre la profundidad práctica
/// del quadtree; sin recursión, sin `Vec`).
const BH_STACK_CAP: usize = 128;

/// Acelera los cuerpos en posiciones vivas `[start, end)` sobre el quadtree
/// ya construido (masas en `node.mass`).
///
/// Traversal ITERATIVO con pila fija en stack `[usize; BH_STACK_CAP]`, sin
/// recursión, sin `Vec`/`String`/`format!`/`Box` ni closures que capturen.
///
/// Criterio de apertura: aceptar el nodo como monopolo si
///   s² < θ²·d²   con s = 2·half_size (lado completo de la celda)
///   y d = distancia del cuerpo `i` al centro del nodo.
///
/// Softening en el término del monopolo (se suma al dist²), sin división
/// flotante explícita (usa `.recip()`).
///
/// Escribe `Fx[i]`/`Fy[i]` una vez por cuerpo (half-coupling).
#[inline(always)]
pub fn compute_forces_barnes_hut_range(
    arena: &QuadtreeArena,
    store: &NativeStore,
    start: usize,
    end: usize,
    g: f32,
    softening: f32,
    theta: f32,
) {
    unsafe {
        let root = arena.root_index;
        if root < 0 {
            return;
        }
        let live_count = store.live_count;
        let mut end = end;
        if end > live_count {
            end = live_count;
        }
        if start >= end {
            return;
        }

        let x = store.x;
        let y = store.y;
        let mass = store.mass;
        let fx = store.fx;
        let fy = store.fy;
        let body_next = store.body_next;

        let theta2 = theta * theta;
        let soft2 = softening * softening;

        let mut stack: [usize; BH_STACK_CAP] = [0; BH_STACK_CAP];

        let mut p = start;
        while p < end {
            let i = *store.live_indices.add(p) as usize;
            let ix = *x.add(i);
            let iy = *y.add(i);

            let mut ax: f32 = 0.0;
            let mut ay: f32 = 0.0;

            let mut sp: usize = 0;
            stack[sp] = root as usize;
            sp += 1;

            while sp > 0 {
                sp -= 1;
                let node_idx = stack[sp];
                let node = arena.read_node(node_idx);

                match node.kind {
                    0 => {} // Empty: no aporta
                    1 => { // Leaf con un cuerpo
                        let j = node.body_index as usize;
                        if j == i {
                            continue;
                        }
                        let mj = *mass.add(j);
                        if mj <= 0.0 {
                            continue;
                        }
                        let dx = *x.add(j) - ix;
                        let dy = *y.add(j) - iy;
                        let mut dist2 = dx * dx + dy * dy + soft2;
                        if dist2 <= 0.0 {
                            dist2 = 1e-32f32;
                        }
                        let r2 = dist2.recip();
                        let inv_dist3 = r2 * r2.sqrt();
                        let acc = g * mj * inv_dist3;
                        ax = ax + acc * dx;
                        ay = ay + acc * dy;
                    }
                    3 => { // LeafMulti: recorrer la cadena cuerpo a cuerpo (sin monopolo)
                        // Misma fórmula pairwise que Leaf (kind 1); se salta el
                        // propio cuerpo (j == i) y masas no positivas, y se
                        // sigue body_next hasta -1.
                        let mut j = node.body_index as usize;
                        loop {
                            if j != i {
                                let mj = *mass.add(j);
                                if mj > 0.0 {
                                    let dx = *x.add(j) - ix;
                                    let dy = *y.add(j) - iy;
                                    let mut dist2 = dx * dx + dy * dy + soft2;
                                    if dist2 <= 0.0 {
                                        dist2 = 1e-32f32;
                                    }
                                    let r2 = dist2.recip();
                                    let inv_dist3 = r2 * r2.sqrt();
                                    let acc = g * mj * inv_dist3;
                                    ax = ax + acc * dx;
                                    ay = ay + acc * dy;
                                }
                            }
                            let nxt = *body_next.add(j);
                            if nxt == -1i32 {
                                break;
                            }
                            j = nxt as usize;
                        }
                    }
                    2 => { // Internal: criterio de apertura
                        let s = 2.0 * node.half_size;
                        let s2 = s * s;
                        let dx = node.center_x - ix;
                        let dy = node.center_y - iy;
                        let d2 = dx * dx + dy * dy;
                        if s2 < theta2 * d2 {
                            // Aceptar monopolo del nodo
                            let mj = node.mass;
                            if mj > 0.0 {
                                let mut dist2 = d2 + soft2;
                                if dist2 <= 0.0 {
                                    dist2 = 1e-32f32;
                                }
                                let r2 = dist2.recip();
                                let inv_dist3 = r2 * r2.sqrt();
                                let acc = g * mj * inv_dist3;
                                ax = ax + acc * dx;
                                ay = ay + acc * dy;
                            }
                        } else {
                            // Abrir: apilar los 4 hijos
                            let base = node.child_offset as usize;
                            if sp + 4 <= BH_STACK_CAP {
                                stack[sp] = base;
                                sp += 1;
                                stack[sp] = base + 1;
                                sp += 1;
                                stack[sp] = base + 2;
                                sp += 1;
                                stack[sp] = base + 3;
                                sp += 1;
                            }
                        }
                    }
                    _ => {} // kind inválido, ignorar
                }
            }

            *fx.add(i) = ax;
            *fy.add(i) = ay;
            p += 1;
        }
    }
}

// ============================================================================
// Utilidades
// ============================================================================

/// Potencia de 2 >= `v` (v >= 1). Sin división flotante.
#[inline(always)]
fn next_power_of_two(mut v: usize) -> usize {
    if v <= 1 {
        return 1;
    }
    v -= 1;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    if core::mem::size_of::<usize>() == 8 {
        v |= v >> 32;
    }
    v + 1
}

