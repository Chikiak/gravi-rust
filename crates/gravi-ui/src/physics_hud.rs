//! physics_hud — HUD de física para el render (Onda 5).
//! Formateo manual de líneas en `&mut [u8]`: zero-allocation,
//! sin `format!`, `String` ni `Vec` en el formateo.

/// Escribe la línea 1 del HUD: `SIMD:<label> | kt:<ms>ms | FLOPS:<Mf>Mf`
/// Devuelve el número de bytes escritos (truncado al largo de `buf`).
pub fn write_line1(buf: &mut [u8], simd_label: &str, kt_ms: u64, mflops: f32) -> usize {
    let mut pos = 0;
    pos += write_str(buf, pos, b"SIMD:");
    pos += write_str(buf, pos, simd_label.as_bytes());
    pos += write_str(buf, pos, b" | kt:");
    pos += write_u64(buf, pos, kt_ms);
    pos += write_str(buf, pos, b"ms | FLOPS:");
    pos += write_f32_fixed(buf, pos, mflops, 1);
    pos += write_str(buf, pos, b"Mf");
    pos
}

/// Escribe la línea 2 del HUD: `ENG:<engine> | NODES:<n> | BUILD:<ms>ms | TRAV:<ms>ms`
/// Devuelve el número de bytes escritos (truncado al largo de `buf`).
pub fn write_line2(buf: &mut [u8], engine: &str, nodes: u64, build_ms: u64, trav_ms: u64) -> usize {
    let mut pos = 0;
    pos += write_str(buf, pos, b"ENG:");
    pos += write_str(buf, pos, engine.as_bytes());
    pos += write_str(buf, pos, b" | NODES:");
    pos += write_u64(buf, pos, nodes);
    pos += write_str(buf, pos, b" | BUILD:");
    pos += write_u64(buf, pos, build_ms);
    pos += write_str(buf, pos, b"ms | TRAV:");
    pos += write_u64(buf, pos, trav_ms);
    pos += write_str(buf, pos, b"ms");
    pos
}

// ---------------------------------------------------------------------------
// Helpers de formateo manual (zero-allocation)
// ---------------------------------------------------------------------------

/// Copia `src` en `buf` desde `pos`, respetando el límite. Devuelve bytes escritos.
fn write_str(buf: &mut [u8], mut pos: usize, src: &[u8]) -> usize {
    let mut written = 0;
    while written < src.len() && pos < buf.len() {
        buf[pos] = src[written];
        pos += 1;
        written += 1;
    }
    written
}

/// itoa manual de u64 en base 10. Devuelve bytes escritos.
fn write_u64(buf: &mut [u8], mut pos: usize, mut n: u64) -> usize {
    if n == 0 {
        return write_str(buf, pos, b"0");
    }
    // Dígitos al revés en un buffer de pila fijo.
    let mut tmp = [0u8; 20];
    let mut len = 0;
    while n > 0 {
        tmp[len] = b'0' + (n % 10) as u8;
        n /= 10;
        len += 1;
    }
    let mut written = 0;
    while len > 0 && pos < buf.len() {
        len -= 1;
        buf[pos] = tmp[len];
        pos += 1;
        written += 1;
    }
    written
}

/// f32 con dígitos fijos después del punto (sin `format!`).
/// `decimals` cifra decimal forzada (ej. 1 -> "12.3"). Devuelve bytes escritos.
fn write_f32_fixed(buf: &mut [u8], mut pos: usize, v: f32, decimals: u32) -> usize {
    let mut written = 0;
    let neg = v.is_sign_negative();
    let x = v.abs();

    if neg && pos < buf.len() {
        buf[pos] = b'-';
        pos += 1;
        written += 1;
    }

    // Multiplicador 10^decimals (sin división flotante en el cálculo).
    let mut scale: f32 = 1.0;
    let mut i = 0;
    while i < decimals {
        scale = scale * 10.0;
        i += 1;
    }
    let scaled = (x * scale).round() as i64;
    let int_part = scaled / (10i64.pow(decimals));
    let frac_part = scaled % (10i64.pow(decimals));

    written += write_i64(buf, pos, int_part as i64);
    pos += written;

    if decimals > 0 && pos < buf.len() {
        buf[pos] = b'.';
        pos += 1;
        written += 1;
    }

    // Parte fraccionaria con ceros a la izquierda.
    let mut frac_tmp = [0u8; 10];
    let mut frac_len: usize = 0;
    let mut fp = frac_part;
    while frac_len < decimals as usize {
        frac_tmp[frac_len] = b'0' + (fp % 10) as u8;
        fp /= 10;
        frac_len += 1;
    }
    let mut k: usize = 0;
    while k < frac_len && pos < buf.len() {
        buf[pos] = frac_tmp[frac_len - 1 - k];
        pos += 1;
        k += 1;
        written += 1;
    }
    written
}

/// itoa manual de i64 con signo. Devuelve bytes escritos.
fn write_i64(buf: &mut [u8], mut pos: usize, n: i64) -> usize {
    if n == 0 {
        return write_str(buf, pos, b"0");
    }
    let neg = n < 0;
    let mut m: u64 = if neg { n.unsigned_abs() } else { n as u64 };
    let mut tmp = [0u8; 24];
    let mut len = 0;
    while m > 0 {
        tmp[len] = b'0' + (m % 10) as u8;
        m /= 10;
        len += 1;
    }
    let mut written = 0;
    if neg && pos < buf.len() {
        buf[pos] = b'-';
        pos += 1;
        written += 1;
    }
    while len > 0 && pos < buf.len() {
        len -= 1;
        buf[pos] = tmp[len];
        pos += 1;
        written += 1;
    }
    written
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line1_format_and_bounds() {
        let mut buf = [0u8; 64];
        let n = write_line1(&mut buf, "portable", 5, 1234.5);
        let s = core::str::from_utf8(&buf[..n]).unwrap();
        assert_eq!(s, "SIMD:portable | kt:5ms | FLOPS:1234.5Mf", "line1={:?}", s);
        // Truncación segura: buf muy pequeño no sobrescribe.
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
}
