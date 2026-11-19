use gravi_ui::physics_hud::{write_f32_fixed, write_line1, write_line2, write_u64};

#[test]
fn line1_format_and_bounds() {
    let mut buf = [0u8; 64];
    let n = write_line1(&mut buf, "portable", 5, 1234.5);
    let s = core::str::from_utf8(&buf[..n]).unwrap();
    assert_eq!(s, "SIMD:portable | kt:5ms | FLOPS:1234.5Mf", "line1={:?}", s);
    let mut small = [0u8; 4];
    let m = write_line1(&mut small, "x", 1, 1.0);
    assert!(m <= small.len());
}

#[test]
fn line2_format_and_bounds() {
    let mut buf = [0u8; 64];
    let n = write_line2(&mut buf, "barnes-hut", 1024, 3, 7);
    let s = core::str::from_utf8(&buf[..n]).unwrap();
    assert_eq!(s, "ENG:barnes-hut | NODES:1024 | BUILD:3ms | TRAV:7ms", "line2={:?}", s);
    let mut small = [0u8; 2];
    let m = write_line2(&mut small, "e", 1, 1, 1);
    assert!(m <= small.len());
}

#[test]
fn itoa_and_fixed() {
    let mut b = [0u8; 16];
    assert_eq!(write_u64(&mut b, 0, 0), 1);
    assert_eq!(write_u64(&mut b, 0, 42), 2);
    assert_eq!(write_u64(&mut b, 0, 123456789), 9);
    let mut bf = [0u8; 16];
    let n = write_f32_fixed(&mut bf, 0, 3.1415f32, 2);
    assert_eq!(n, 4);
    let s = core::str::from_utf8(&bf[..n]).unwrap();
    assert_eq!(s, "3.14");
}

#[test]
#[ignore = "medición de rendimiento: comparar TIMING antes/después de la tabla const"]
fn timing_write_f32_fixed_100k() {
    use std::hint::black_box;
    use std::time::Instant;
    let iters = 100_000;
    let mut buf = [0u8; 16];
    let start = Instant::now();
    for _ in 0..iters {
        let n = black_box(write_f32_fixed(&mut buf, 0, 3.1415f32, 2));
        black_box(n);
    }
    let elapsed = start.elapsed();
    println!("TIMING write_f32_fixed {} iters en {:?} ({} ns/call)", iters, elapsed, elapsed.as_nanos() / iters as u128);
}
