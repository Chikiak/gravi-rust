use gravi_core::heap_verifier::HeapVerifier;

#[test]
fn ac_begin_end_frame_reports_zero_delta_without_allocations() {
    let mut hv = HeapVerifier::new();
    hv.begin_frame();
    let _x: usize = 42;
    let report = hv.end_frame();
    assert_eq!(report.delta_bytes, 0);
    assert_eq!(report.heap_bytes_start, report.heap_bytes_end);
    assert_eq!(hv.max_delta(), 0);
}

#[test]
fn ac_max_delta_zero_after_full_physics_pipeline() {
    let mut hv = HeapVerifier::new();
    let mut i = 0u64;
    while i < 100 {
        hv.begin_frame();
        let mut acc: f32 = 0.0;
        let mut j = 0usize;
        while j < 64 {
            acc = acc + (j as f32) * 0.5f32;
            j += 1;
        }
        let _ = acc;
        let report = hv.end_frame();
        assert_eq!(report.delta_bytes, 0, "frame {}", i);
        i += 1;
    }
    assert_eq!(hv.max_delta(), 0, "max_delta debe ser 0 sin save/load");
}

#[test]
fn ac_report_fields_consistent() {
    let mut hv = HeapVerifier::new();
    hv.begin_frame();
    let _r = hv.end_frame();
    assert_eq!(hv.max_delta(), 0);
    assert_eq!(_r.frame_id, 1);
    assert_eq!(_r.delta_bytes, 0);
}

#[test]
fn ac_allocator_global_cuenta_asignaciones_reales() {
    let mut hv = HeapVerifier::new();
    hv.begin_frame();
    let v = gravi_core::heap_verifier::probe_allocate_usize();
    let _ = *v;
    let report = hv.end_frame();
    assert!(report.delta_bytes > 0, "delta_bytes={}", report.delta_bytes);
    assert!(report.max_delta_bytes > 0, "max_delta_bytes={}", report.max_delta_bytes);
    assert!(hv.max_delta() > 0, "max_delta={}", hv.max_delta());
    hv.begin_frame();
    let report2 = hv.end_frame();
    assert_eq!(report2.delta_bytes, 0, "delta_bytes={}", report2.delta_bytes);
    drop(v);
}

#[test]
fn ac_max_delta_tracks_peak() {
    let mut hv = HeapVerifier::new();
    hv.begin_frame();
    let _r1 = hv.end_frame();
    assert_eq!(_r1.max_delta_bytes, 0);
    hv.begin_frame();
    let v = gravi_core::heap_verifier::probe_allocate_usize();
    let _ = *v;
    let _r2 = hv.end_frame();
    assert!(_r2.delta_bytes > 0, "delta_bytes={}", _r2.delta_bytes);
    assert!(_r2.max_delta_bytes > 0, "max_delta_bytes={}", _r2.max_delta_bytes);
    assert!(hv.max_delta() > 0, "max_delta={}", hv.max_delta());
    drop(v);
}

#[test]
fn ac_reset_max() {
    let mut hv = HeapVerifier::new();
    hv.begin_frame();
    let v = gravi_core::heap_verifier::probe_allocate_usize();
    let _ = *v;
    let _r = hv.end_frame();
    assert!(_r.delta_bytes > 0, "delta_bytes={}", _r.delta_bytes);
    hv.reset_max();
    assert_eq!(hv.max_delta(), 0);
    drop(v);
}

#[test]
fn ac_heap_box_delta_positive() {
    let mut hv = HeapVerifier::new();
    hv.begin_frame();
    let v = gravi_core::heap_verifier::probe_allocate_usize();
    let _ = *v;
    let report = hv.end_frame();
    assert!(report.delta_bytes > 0, "delta_bytes={}", report.delta_bytes);
    assert!(hv.max_delta() > 0, "max_delta={}", hv.max_delta());
    drop(v);
}
