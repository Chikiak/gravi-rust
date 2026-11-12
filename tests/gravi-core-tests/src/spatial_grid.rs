//! spatial_grid.rs — Tests de integración: SpatialGrid y SpatialPairEnumerator.
//! Extraídos de crates/gravi-core/src/spatial_grid.rs (mod tests).

use gravi_core::spatial_grid::{rebuild_grid, SpatialGrid, SpatialPairEnumerator};
use gravi_core::store::NativeStore;

fn make_store(count: usize) -> NativeStore {
    let mut store = NativeStore::new(count).expect("store creation failed");
    store.init_lifecycle();
    store
}

#[test]
fn ac1_allocate_reserves_buckets_and_next() {
    let grid = SpatialGrid::allocate(30, 16, 40000).unwrap();
    assert_eq!(grid.grid_width, 30);
    assert_eq!(grid.grid_height, 16);
    assert_eq!(grid.bucket_count, 480);
    assert_eq!(grid.body_count_max, 40000);
    assert!(!grid.bucket_head.is_null());
    assert!(!grid.next.is_null());
    // Todas las cabeceras inicializadas a -1.
    unsafe {
        let mut b = 0usize;
        while b < 480usize {
            assert_eq!(*grid.bucket_head.add(b), -1);
            b += 1;
        }
    }
    // Dispose doble sin panic.
    let mut g2 = grid;
    unsafe { SpatialGrid::dispose(&mut g2); }
    assert!(g2.bucket_head.is_null());
    drop(g2);
}

#[test]
fn ac2_rebuild_grid_no_write_outside_bucket_head() {
    let mut store = make_store(4);
    store.x_slice()[0] = 10.0; store.y_slice()[0] = 10.0; store.mass_slice()[0] = 1.0;
    store.x_slice()[1] = 70.0; store.y_slice()[1] = 70.0; store.mass_slice()[1] = 1.0;
    store.x_slice()[2] = 130.0; store.y_slice()[2] = 130.0; store.mass_slice()[2] = 1.0;
    store.x_slice()[3] = -5.0; store.y_slice()[3] = -5.0; store.mass_slice()[3] = 1.0; // fuera, clampeado
    store.live_count = 4;
    let li = store.live_indices_slice();
    li[0] = 0; li[1] = 1; li[2] = 2; li[3] = 3;

    let mut grid = SpatialGrid::allocate(30, 16, 4).unwrap();
    grid.set_cell_size(64.0);
    rebuild_grid(&grid, &store);

    // Cuerpo 3 en (-5,-5) clampeado a celda (0,0) -> bucket 0, prepende.
    // Cuerpo 0 también en (10,10) -> bucket 0. Insertado antes que 3.
    // Lista bucket 0: 3 -> 0 -> -1.
    unsafe {
        assert_eq!(*grid.bucket_head.add(0), 3);
        assert_eq!(*grid.next.add(3), 0);
        assert_eq!(*grid.next.add(0), -1);
    }
    // Ninguna escritura fuera de bucket_head: todas las cabeceras válidas.
    unsafe {
        let mut b = 0usize;
        while b < 480usize {
            let h = *grid.bucket_head.add(b);
            assert!(h >= -1 && h < 4, "cabecera fuera de rango en bucket {}", b);
            b += 1;
        }
    }
}

#[test]
fn ac3_dead_slots_skipped() {
    let mut store = make_store(3);
    store.x_slice()[0] = 10.0; store.y_slice()[0] = 10.0; store.mass_slice()[0] = 1.0;
    store.x_slice()[1] = 20.0; store.y_slice()[1] = 20.0; store.mass_slice()[1] = -1.0; // muerto
    store.x_slice()[2] = 30.0; store.y_slice()[2] = 30.0; store.mass_slice()[2] = 1.0;
    store.live_count = 3;
    let li = store.live_indices_slice();
    li[0] = 0; li[1] = 1; li[2] = 2;

    let mut grid = SpatialGrid::allocate(30, 16, 3).unwrap();
    grid.set_cell_size(64.0);
    rebuild_grid(&grid, &store);

    let mut e = SpatialPairEnumerator::new(&grid, &store);
    let mut count = 0usize;
    while e.move_next() {
        let p = e.current();
        assert_ne!(p.i, 1, "slot muerto no debe ser I");
        assert_ne!(p.j, 1, "slot muerto no debe ser J");
        count += 1;
    }
    // Vivos: 0 y 2, buckets distintos (0 y 1) pero mismos coords no,
    // 0->(0,0), 2->(0,0) también (30/64=0). Mismo bucket -> 1 par (0,2).
    assert_eq!(count, 1);
}

#[test]
fn ac4_enumerator_emits_each_pair_once() {
    let mut store = make_store(3);
    // Todos en la misma celda (0,0).
    for i in 0..3usize {
        store.x_slice()[i] = 10.0 + (i as f32);
        store.y_slice()[i] = 10.0 + (i as f32);
        store.mass_slice()[i] = 1.0;
    }
    store.live_count = 3;
    let li = store.live_indices_slice();
    li[0] = 0; li[1] = 1; li[2] = 2;

    let mut grid = SpatialGrid::allocate(30, 16, 3).unwrap();
    grid.set_cell_size(64.0);
    rebuild_grid(&grid, &store);

    let mut e = SpatialPairEnumerator::new(&grid, &store);
    let mut found_01 = false;
    let mut found_02 = false;
    let mut found_12 = false;
    let mut total = 0usize;
    while e.move_next() {
        let p = e.current();
        assert!(p.j > p.i, "j > i: {:?}", p);
        total += 1;
        match (p.i, p.j) {
            (0, 1) => found_01 = true,
            (0, 2) => found_02 = true,
            (1, 2) => found_12 = true,
            _ => panic!("par inesperado {:?}", p),
        }
    }
    assert_eq!(total, 3, "se esperaban 3 pares");
    assert!(found_01 && found_02 && found_12, "faltan pares");
}

#[test]
fn ac5_get_cell_out_of_range_returns_minus_one() {
    let grid = SpatialGrid::allocate(30, 16, 4).unwrap();
    assert_eq!(grid.get_cell(-1, 0), -1);
    assert_eq!(grid.get_cell(30, 0), -1);
    assert_eq!(grid.get_cell(0, -1), -1);
    assert_eq!(grid.get_cell(0, 16), -1);
    assert_eq!(grid.get_cell(0, 0), -1); // vacío, no fuera de rango
}

#[test]
fn ac6_bodies_outside_window_clamp() {
    let mut store = make_store(2);
    store.x_slice()[0] = -100.0; store.y_slice()[0] = -100.0; store.mass_slice()[0] = 1.0;
    store.x_slice()[1] = 2000.0; store.y_slice()[1] = 2000.0; store.mass_slice()[1] = 1.0;
    store.live_count = 2;
    let li = store.live_indices_slice();
    li[0] = 0; li[1] = 1;

    let mut grid = SpatialGrid::allocate(30, 16, 2).unwrap();
    grid.set_cell_size(64.0);
    rebuild_grid(&grid, &store);

    // Ambos clampeados a celdas de borde, sin acceso fuera de bucket_head.
    unsafe {
        // cuerpo 0 -> cx=0, cy=0 -> bucket 0
        // cuerpo 1 -> cx=30->29, cy=16->15 -> bucket 15*30+29=479
        let h0 = *grid.bucket_head.add(0);
        let h1 = *grid.bucket_head.add(479);
        assert!(h0 >= 0 && h0 < 2);
        assert!(h1 >= 0 && h1 < 2);
    }
    let mut e = SpatialPairEnumerator::new(&grid, &store);
    let mut count = 0usize;
    while e.move_next() {
        let _ = e.current();
        count += 1;
    }
    // Ambos en celdas de borde distintas (0,0) y (29,15) -> sin par.
    assert_eq!(count, 0);
}
