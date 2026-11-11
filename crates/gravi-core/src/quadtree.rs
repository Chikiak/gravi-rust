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
///   28 kind       u32  0=Empty, 1=Leaf, 2=Internal
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

        let live_count = store.live_count;
        if live_count == 0 {
            return Ok(());
        }

        let x = store.x;
        let y = store.y;
        let mass = store.mass;
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
                root_cx, root_cy, root_half, x, y, mass,
            )?;
        }
        Ok(())
    }
}

/// Inserción iterativa de un cuerpo descendiendo desde `node_idx` hasta una hoja.
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
    x: *mut f32,
    y: *mut f32,
    mass: *mut f32,
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
            1 => { // Leaf -> subdividir si hay otro cuerpo distinto
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
                // Subdividir en 4 hijos contiguos (reservar slots primero, sin
                // mantener ninguna referencia mutable al arena).
                // Reservar 4 slots contiguos para los hijos (child_base..child_base+4).
                // alloc_node avanza 1; se avanzan los 3 restantes para que la
                // siguiente alloc_node no sobrescriba los hijos ya escritos.
                let child_base = arena.alloc_node()?;
                arena.next_free = child_base + 4;
                if arena.next_free > arena.capacity {
                    return Err(QuadtreeError::Overflow);
                }
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
                    x,
                    y,
                    mass,
                )?;

                let q_new = quadrant_of(bx, by, cx, cy);
                let (ncx, ncy, nhalf) = child_cell(cx, cy, half, q_new);
                insert_body_iterative(
                    arena, child_base + q_new, body_idx, bx, by, bm,
                    ncx, ncy, nhalf, x, y, mass,
                )?;
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
            return (0.0, 0.0, 0.0);
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
        let half = if extent_x > extent_y {
            extent_x * 0.5
        } else {
            extent_y * 0.5
        };
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

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_store(count: usize) -> NativeStore {
        let mut store = NativeStore::new(count).expect("store creation failed");
        store.init_lifecycle();
        store
    }

    #[test]
    fn ac1_quadnode_es_32_bytes_repr_c() {
        assert_eq!(core::mem::size_of::<QuadNode>(), 32);
        assert_eq!(core::mem::align_of::<QuadNode>(), 4);
    }

    #[test]
    fn ac2_arena_capacidad_es_potencia_de_2_maxima_4n16() {
        for n in [1usize, 2, 4, 8, 16, 100, 1000, 40000] {
            let arena = QuadtreeArena::allocate(n).unwrap();
            let needed = 4 * n + 16;
            assert!(arena.capacity >= needed, "cap={} < needed={}", arena.capacity, needed);
            // potencia de 2
            assert!(arena.capacity & (arena.capacity - 1) == 0, "cap={} no es pow2", arena.capacity);
            assert_eq!(arena.next_free, 0);
            assert_eq!(arena.root_index, -1);
            assert!(!arena.base.is_null());
        }
    }

    #[test]
    fn ac3_arena_cero_cuerpos_es_error() {
        assert!(matches!(
            QuadtreeArena::allocate(0),
            Err(QuadtreeError::ZeroCapacity)
        ));
    }

    #[test]
    fn ac4_build_arbol_simple_raiz_una_hoja() {
        let mut store = make_store(1);
        store.x_slice()[0] = 10.0;
        store.y_slice()[0] = 20.0;
        store.mass_slice()[0] = 5.0;
        store.live_count = 1;
        let li = store.live_indices_slice();
        li[0] = 0;

        let mut arena = QuadtreeArena::allocate(4).unwrap();
        let res = build_quadtree(&store, &mut arena, 0.0, 0.0, 100.0);
        assert!(res.is_ok());
        assert_eq!(arena.root_index, 0);

        unsafe {
            let root = arena.node(0);
            assert_eq!(root.kind, 1); // Leaf
            assert_eq!(root.body_index, 0);
            assert_eq!(root.count, 1);
            assert_eq!(root.mass, 5.0);
        }
    }

    #[test]
    fn ac5_build_subdivide_hoja_al_segundo_cuerpo() {
        let mut store = make_store(2);
        store.x_slice()[0] = 10.0;
        store.y_slice()[0] = 10.0;
        store.mass_slice()[0] = 1.0;
        store.x_slice()[1] = 90.0;
        store.y_slice()[1] = 90.0;
        store.mass_slice()[1] = 2.0;
        store.live_count = 2;
        let li = store.live_indices_slice();
        li[0] = 0;
        li[1] = 1;

        let mut arena = QuadtreeArena::allocate(8).unwrap();
        let res = build_quadtree(&store, &mut arena, 50.0, 50.0, 100.0);
        assert!(res.is_ok());

        unsafe {
            let root = arena.node(0);
            assert_eq!(root.kind, 2); // Internal (subdividió)
            assert_eq!(root.count, 2);
            assert_eq!(root.mass, 3.0);

            // Cada cuerpo en su propia hoja (hijos del root).
            let base = root.child_offset as usize;
            let mut found = 0usize;
            for k in 0..4usize {
                let ch = arena.node(base + k);
                if ch.kind == 1 {
                    found += 1;
                    assert_eq!(ch.count, 1);
                }
            }
            assert_eq!(found, 2);
        }
    }

    #[test]
    fn ac6_alloc_node_bump_o1() {
        let mut arena = QuadtreeArena::allocate(16).unwrap();
        assert_eq!(arena.alloc_node(), Ok(0));
        assert_eq!(arena.alloc_node(), Ok(1));
        assert_eq!(arena.next_free, 2);
    }

    #[test]
    fn ac7_overflow_si_sobrepasa_capacidad() {
        let mut arena = QuadtreeArena::allocate(1).unwrap();
        let cap = arena.capacity;
        for _ in 0..cap {
            assert!(arena.alloc_node().is_ok());
        }
        assert_eq!(arena.alloc_node().unwrap_err(), QuadtreeError::Overflow);
    }

    #[test]
    fn ac8_drop_libera_sin_panic() {
        let arena = QuadtreeArena::allocate(16).unwrap();
        let _ptr = arena.base;
        assert!(!_ptr.is_null());
        drop(arena);
        // Si llegamos aquí, no hubo double-free ni panic.
    }

    #[test]
    fn ac9_build_masas_acumuladas_en_camino() {
        let mut store = make_store(3);
        store.x_slice()[0] = 10.0;
        store.y_slice()[0] = 10.0;
        store.mass_slice()[0] = 1.0;
        store.x_slice()[1] = 12.0;
        store.y_slice()[1] = 12.0;
        store.mass_slice()[1] = 2.0;
        store.x_slice()[2] = 90.0;
        store.y_slice()[2] = 90.0;
        store.mass_slice()[2] = 4.0;
        store.live_count = 3;
        let li = store.live_indices_slice();
        li[0] = 0;
        li[1] = 1;
        li[2] = 2;

        let mut arena = QuadtreeArena::allocate(16).unwrap();
        assert!(build_quadtree(&store, &mut arena, 50.0, 50.0, 100.0).is_ok());

        unsafe {
            let root = arena.node(0);
            assert_eq!(root.count, 3);
            assert_eq!(root.mass, 7.0);
        }
    }

    #[test]
    fn ac10_arena_dispose_pone_base_nulo() {
        let mut arena = QuadtreeArena::allocate(16).unwrap();
        unsafe { QuadtreeArena::dispose(&mut arena); }
        assert!(arena.base.is_null());
        assert_eq!(arena.capacity, 0);
        assert_eq!(arena.root_index, -1);
    }

    // --- Issue 6, parte A.2: compute_quadtree_masses ---

    #[test]
    fn a2_masses_arbol_vacio_root_negativo_es_ok() {
        let store = make_store(1);
        let mut arena = QuadtreeArena::allocate(4).unwrap();
        arena.root_index = -1; // sin arbol construido
        assert!(compute_quadtree_masses(&mut arena, &store).is_ok());
    }

    #[test]
    fn a2_masses_acumula_masa_y_count_post_orden() {
        // 3 cuerpos en cuadrantes distintos del root (sin subdivicion anidada).
        let mut store = make_store(3);
        store.x_slice()[0] = 10.0;
        store.y_slice()[0] = 10.0;
        store.mass_slice()[0] = 1.0;
        store.x_slice()[1] = 90.0;
        store.y_slice()[1] = 90.0;
        store.mass_slice()[1] = 2.0;
        store.x_slice()[2] = 10.0;
        store.y_slice()[2] = 90.0;
        store.mass_slice()[2] = 4.0;
        store.live_count = 3;
        let li = store.live_indices_slice();
        li[0] = 0;
        li[1] = 1;
        li[2] = 2;

        let mut arena = QuadtreeArena::allocate(32).unwrap();
        assert!(build_quadtree(&store, &mut arena, 50.0, 50.0, 100.0).is_ok());
        let root_idx = arena.root_index as usize;

        // Zero de masa/count en TODOS los nodos para forzar recomputo desde cero.
        unsafe {
            for i in 0..arena.next_free {
                let n = arena.read_node(i);
                arena.write_node(i, QuadNode { mass: 0.0, count: 0, ..n });
            }
        }

        assert!(compute_quadtree_masses(&mut arena, &store).is_ok());

        unsafe {
            let root = arena.read_node(root_idx);
            assert_eq!(root.mass, 7.0, "raiz masa");
            assert_eq!(root.count, 3, "raiz count");
            // Cada hoja conserva masa y count de su cuerpo.
            if root.kind == 2 {
                let base = root.child_offset as usize;
                let mut hojas = 0usize;
                for k in 0..4usize {
                    let ch = arena.read_node(base + k);
                    if ch.kind == 1 {
                        hojas += 1;
                        assert_eq!(ch.count, 1);
                        assert!(ch.mass > 0.0);
                    }
                }
                assert_eq!(hojas, 3);
            }
        }
    }

    #[test]
    fn a2_masses_una_hoja_sin_subdividir() {
        let mut store = make_store(1);
        store.x_slice()[0] = 10.0;
        store.y_slice()[0] = 20.0;
        store.mass_slice()[0] = 5.0;
        store.live_count = 1;
        let li = store.live_indices_slice();
        li[0] = 0;

        let mut arena = QuadtreeArena::allocate(4).unwrap();
        assert!(build_quadtree(&store, &mut arena, 0.0, 0.0, 100.0).is_ok());
        assert!(compute_quadtree_masses(&mut arena, &store).is_ok());

        unsafe {
            let root = arena.read_node(arena.root_index as usize);
            assert_eq!(root.kind, 1);
            assert_eq!(root.mass, 5.0);
            assert_eq!(root.count, 1);
        }
    }

    // --- Issue 6, parte A.2: root_from_live_bodies ---

    #[test]
    fn a2_root_cero_cuerpo_devuelve_ceros() {
        let store = NativeStore::new(4).unwrap();
        let (cx, cy, half) = root_from_live_bodies(&store);
        assert_eq!((cx, cy, half), (0.0, 0.0, 0.0));
    }

    #[test]
    fn a2_root_un_cuerpo_centro_en_el_cuerpo() {
        let mut store = make_store(1);
        store.x_slice()[0] = 100.0;
        store.y_slice()[0] = -200.0;
        store.mass_slice()[0] = 7.0;
        store.live_count = 1;
        let li = store.live_indices_slice();
        li[0] = 0;

        let (cx, cy, half) = root_from_live_bodies(&store);
        assert_eq!(cx, 100.0);
        assert_eq!(cy, -200.0);
        assert_eq!(half, 0.0); // extent nulo
    }

    #[test]
    fn a2_root_centro_dentro_del_rectangulo_de_vivos() {
        let mut store = make_store(4);
        store.x_slice()[0] = 0.0;
        store.y_slice()[0] = 0.0;
        store.x_slice()[1] = 100.0;
        store.y_slice()[1] = 0.0;
        store.x_slice()[2] = 0.0;
        store.y_slice()[2] = 100.0;
        store.x_slice()[3] = 100.0;
        store.y_slice()[3] = 100.0;
        for i in 0..4usize {
            store.mass_slice()[i] = 1.0;
        }
        store.live_count = 4;
        let li = store.live_indices_slice();
        for i in 0..4usize {
            li[i] = i as i32;
        }

        let (cx, cy, half) = root_from_live_bodies(&store);
        assert_eq!(cx, 50.0);
        assert_eq!(cy, 50.0);
        assert_eq!(half, 50.0); // max(extent_x, extent_y)/2 = 100/2

        // Centro dentro del rectangulo [0,100]x[0,100]
        assert!(cx >= 0.0 && cx <= 100.0);
        assert!(cy >= 0.0 && cy <= 100.0);
        assert!(half > 0.0);
    }

    #[test]
    fn a2_root_solo_cuerpos_vivos_ignorando_muertos() {
        let mut store = make_store(4);
        // Cuerpo 0 y 2 vivos, 1 y 3 muertos (masa < 0, pero live_count define los vivos).
        store.x_slice()[0] = 10.0;
        store.y_slice()[0] = 10.0;
        store.mass_slice()[0] = 1.0;
        store.x_slice()[1] = 1000.0;
        store.y_slice()[1] = 1000.0;
        store.mass_slice()[1] = -1.0; // muerto (no esta en live_indices)
        store.x_slice()[2] = 20.0;
        store.y_slice()[2] = 20.0;
        store.mass_slice()[2] = 1.0;
        store.x_slice()[3] = -1000.0;
        store.y_slice()[3] = -1000.0;
        store.mass_slice()[3] = -1.0; // muerto
        store.live_count = 2;
        let li = store.live_indices_slice();
        li[0] = 0;
        li[1] = 2;

        let (cx, cy, half) = root_from_live_bodies(&store);
        assert_eq!(cx, 15.0); // (10+20)/2
        assert_eq!(cy, 15.0);
        assert_eq!(half, 5.0); // extent 10 / 2
    }

    // --- Issue 7: compute_forces_barnes_hut_range ---

    /// Aceleración esperada (fuerza bruta pairwise) para el cuerpo `i`.
    fn bruto_esperado(
        store: &NativeStore,
        i: usize,
        g: f32,
        soft2: f32,
    ) -> (f32, f32) {
        unsafe {
            let x = store.x;
            let y = store.y;
            let mass = store.mass;
            let live_count = store.live_count;
            let live_indices = store.live_indices;
            let ix = *x.add(i);
            let iy = *y.add(i);
            let mut ex: f32 = 0.0;
            let mut ey: f32 = 0.0;
            let mut q: usize = 0;
            while q < live_count {
                let j = *live_indices.add(q) as usize;
                q += 1;
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
                ex = ex + g * mj * inv_dist3 * dx;
                ey = ey + g * mj * inv_dist3 * dy;
            }
            (ex, ey)
        }
    }

    #[test]
    fn b7_theta_cero_coincide_con_fuerza_bruta() {
        let mut store = make_store(5);
        store.x_slice()[0] = 10.0; store.y_slice()[0] = 10.0; store.mass_slice()[0] = 1.0;
        store.x_slice()[1] = 90.0; store.y_slice()[1] = 90.0; store.mass_slice()[1] = 2.0;
        store.x_slice()[2] = 10.0; store.y_slice()[2] = 90.0; store.mass_slice()[2] = 3.0;
        store.x_slice()[3] = 90.0; store.y_slice()[3] = 10.0; store.mass_slice()[3] = 4.0;
        store.x_slice()[4] = 50.0; store.y_slice()[4] = 50.0; store.mass_slice()[4] = 5.0;
        store.live_count = 5;
        let li = store.live_indices_slice();
        for i in 0..5usize { li[i] = i as i32; }

        let mut arena = QuadtreeArena::allocate(64).unwrap();
        assert!(build_quadtree(&store, &mut arena, 50.0, 50.0, 100.0).is_ok());
        assert!(compute_quadtree_masses(&mut arena, &store).is_ok());

        let g: f32 = 1.0;
        let softening: f32 = 0.0;
        let theta: f32 = 0.0;
        compute_forces_barnes_hut_range(&arena, &store, 0, 5, g, softening, theta);

        unsafe {
            let fx = store.fx; let fy = store.fy;
            let mut max_rel: f32 = 0.0;
            let mut i = 0usize;
            while i < 5usize {
                let (ex, ey) = bruto_esperado(&store, i, g, softening * softening);
                let ax = *fx.add(i); let ay = *fy.add(i);
                eprintln!("DEBUG body {}: ax={} ay={} ex={} ey={}", i, ax, ay, ex, ey);
                let escala = 1.0f32.max(ex.abs()).max(ey.abs());
                let rel = ((ax - ex).abs() + (ay - ey).abs()) / escala.max(1e-6);
                if rel > max_rel { max_rel = rel; }
                assert!(rel < 1e-4, "cuerpo {} theta=0 no coincide: ax={} ay={} ex={} ey={}", i, ax, ay, ex, ey);
                i += 1;
            }
            assert!(max_rel < 1e-4, "max_rel={}", max_rel);
        }
    }

    #[test]
    fn b7_theta_positivo_error_acotado() {
        let mut store = make_store(60);
        let mut i = 0usize;
        while i < 60usize {
            // Cuerpos separados en una rejilla para evitar celdas unitarias profundas.
            store.x_slice()[i] = ((i % 10) as f32) * 11.0 + 2.0;
            store.y_slice()[i] = ((i / 10) as f32) * 11.0 + 2.0;
            store.mass_slice()[i] = 1.0 + (i as f32) * 0.13;
            i += 1;
        }
        store.live_count = 60;
        let li = store.live_indices_slice();
        i = 0;
        while i < 60usize { li[i] = i as i32; i += 1; }

        let mut arena = QuadtreeArena::allocate(256).unwrap();
        assert!(build_quadtree(&store, &mut arena, 50.0, 50.0, 60.0).is_ok());
        assert!(compute_quadtree_masses(&mut arena, &store).is_ok());

        let g: f32 = 1.0;
        let softening: f32 = 1.0;
        let theta: f32 = 0.5;
        compute_forces_barnes_hut_range(&arena, &store, 0, 60, g, softening, theta);

        unsafe {
            let fx = store.fx; let fy = store.fy;
            let soft2 = softening * softening;
            let mut err_sum: f32 = 0.0;
            let mut max_rel: f32 = 0.0;
            let mut i = 0usize;
            while i < 60usize {
                let (ex, ey) = bruto_esperado(&store, i, g, soft2);
                let ax = *fx.add(i); let ay = *fy.add(i);
                let escala = 1.0f32.max(ex.abs()).max(ey.abs());
                let rel = ((ax - ex).abs() + (ay - ey).abs()) / escala;
                err_sum = err_sum + rel;
                if rel > max_rel { max_rel = rel; }
                assert!(rel < 0.50, "cuerpo {} theta=0.5 error={} (demasiado grande)", i, rel);
                i += 1;
            }
            let media = err_sum / 60.0;
            assert!(media < 0.05, "error medio={} >= 5%", media);
            assert!(max_rel < 0.20, "error max={} >= 20%", max_rel);
        }
    }

    #[test]
    fn b7_cluster_degenerado_sin_nan() {
        // Cluster muy apretado (casi un mismo punto): no debe haber NaN ni Inf.
        // Las posiciones son distintas para que el arbol se construya (sin
        // MaxQuadtreeDepth/bucket aun en esta base; eso llega en F5-02).
        let mut store = make_store(12);
        let mut i = 0usize;
        while i < 12usize {
            store.x_slice()[i] = 5.0 + (i as f32) * 0.001;
            store.y_slice()[i] = 5.0 + (i as f32) * 0.001;
            store.mass_slice()[i] = 1.0;
            i += 1;
        }
        store.live_count = 12;
        let li = store.live_indices_slice();
        i = 0;
        while i < 12usize { li[i] = i as i32; i += 1; }

        let mut arena = QuadtreeArena::allocate(256).unwrap();
        let (cx, cy, half) = root_from_live_bodies(&store);
        assert!(build_quadtree(&store, &mut arena, cx, cy, half).is_ok());
        assert!(compute_quadtree_masses(&mut arena, &store).is_ok());

        let g: f32 = 1.0;
        let softening: f32 = 8.0;
        let theta: f32 = 0.5;
        compute_forces_barnes_hut_range(&arena, &store, 0, 12, g, softening, theta);

        unsafe {
            let fx = store.fx; let fy = store.fy;
            let mut i = 0usize;
            while i < 12usize {
                let ax = *fx.add(i);
                let ay = *fy.add(i);
                assert!(ax.is_finite(), "cuerpo {} Fx no finito: {}", i, ax);
                assert!(ay.is_finite(), "cuerpo {} Fy no finito: {}", i, ay);
                i += 1;
            }
        }
    }

    #[test]
    fn b7_arbol_vacio_y_rango_vacio_no_hacen_nada() {
        let store = make_store(1);
        let mut arena = QuadtreeArena::allocate(4).unwrap();
        arena.root_index = -1;
        // Sin root: no escribe ni panica.
        compute_forces_barnes_hut_range(&arena, &store, 0, 1, 1.0, 1.0, 0.5);
        unsafe { assert!(*store.fx.add(0) == 0.0); }

        // Rango vacío: no escribe.
        let mut arena2 = QuadtreeArena::allocate(4).unwrap();
        let (cx, cy, half) = root_from_live_bodies(&store);
        assert!(build_quadtree(&store, &mut arena2, cx, cy, half).is_ok());
        assert!(compute_quadtree_masses(&mut arena2, &store).is_ok());
        compute_forces_barnes_hut_range(&arena2, &store, 0, 0, 1.0, 1.0, 0.5);
    }
}
