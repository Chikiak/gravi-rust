//! scenario.rs — ScenarioStore: serialización `.grav` v1 (Onda 4 / Issue 12).
//!
//! Formato de archivo (todo little-endian):
//!   [0..4)   magic      = b"GRV1"
//!   [4..8)   versión     = u32 (== 1)
//!   [8..12)  attractor_count = u32 (debe ser <= 4)
//!   [12..16) body_count  = u32
//!   [16..40) reserved    = [u8; 24] (ceros)
//!   [40..)   cuerpo      = gzip( body_count u32 LE + [x,y,vx,vy] f32 LE por cuerpo )
//!
//! Cabecera: 40 bytes, repr(C, pack=1), little-endian.
//! La cabecera se valida ANTES de reservar memoria.

use crate::store::NativeStore;
use std::fs;
use std::io::{Read, Write};
use std::path::Path;

const MAGIC: &[u8; 4] = b"GRV1";
const VERSION: u32 = 1;
const HEADER_SIZE: usize = 40;
const MAX_ATTRACTORS: u32 = 4;
const RESERVED_SIZE: usize = 24;

/// Cabecera del formato `.grav` v1. 40 bytes, empaquetada, little-endian.
#[repr(C, packed)]
#[derive(Copy, Clone, Debug)]
struct GravHeader {
    magic: [u8; 4],
    version: u32,
    attractor_count: u32,
    body_count: u32,
    reserved: [u8; RESERVED_SIZE],
}

const _: () = assert!(core::mem::size_of::<GravHeader>() == HEADER_SIZE);

/// Lee un u32 little-endian de un slice de exactamente 4 bytes.
fn read_u32_le(bytes: &[u8]) -> Result<u32, String> {
    if bytes.len() != 4 {
        return Err("se esperaban 4 bytes para u32 LE".to_string());
    }
    let arr = [bytes[0], bytes[1], bytes[2], bytes[3]];
    Ok(u32::from_le_bytes(arr))
}

/// Lee un f32 little-endian de un slice de exactamente 4 bytes.
fn read_f32_le(bytes: &[u8]) -> Result<f32, String> {
    if bytes.len() != 4 {
        return Err("se esperaban 4 bytes para f32 LE".to_string());
    }
    let arr = [bytes[0], bytes[1], bytes[2], bytes[3]];
    Ok(f32::from_le_bytes(arr))
}

/// Error de escenario.
#[derive(Debug)]
pub struct ScenarioError(pub String);

impl std::fmt::Display for ScenarioError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for ScenarioError {}

/// Guarda un escenario en `path`. Escribe X/Y/Vx/Vy de los `body_count`
/// primeros cuerpos del `store`.
pub fn save_scenario(path: &Path, store: &NativeStore, attractor_count: u32) -> Result<(), String> {
    if attractor_count > MAX_ATTRACTORS {
        return Err(format!(
            "attractor_count {} excede el máximo {}",
            attractor_count, MAX_ATTRACTORS
        ));
    }
    let body_count = store.count as u32;
    let x = store.x_slice();
    let y = store.y_slice();
    let vx = store.vx_slice();
    let vy = store.vy_slice();

    // Cuerpo sin comprimir: body_count + 4 canales f32 LE por cuerpo.
    let mut body = Vec::with_capacity(4 + body_count as usize * 16);
    body.extend_from_slice(&body_count.to_le_bytes());
    for i in 0..body_count as usize {
        body.extend_from_slice(&x[i].to_le_bytes());
        body.extend_from_slice(&y[i].to_le_bytes());
        body.extend_from_slice(&vx[i].to_le_bytes());
        body.extend_from_slice(&vy[i].to_le_bytes());
    }

    let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    encoder.write_all(&body).map_err(|e| e.to_string())?;
    let compressed = encoder.finish().map_err(|e| e.to_string())?;

    let header = GravHeader {
        magic: *MAGIC,
        version: VERSION,
        attractor_count,
        body_count,
        reserved: [0; RESERVED_SIZE],
    };
    let mut header_bytes = [0u8; HEADER_SIZE];
    unsafe {
        std::ptr::write(header_bytes.as_mut_ptr() as *mut GravHeader, header);
    }

    let mut out = Vec::with_capacity(HEADER_SIZE + compressed.len());
    out.extend_from_slice(&header_bytes);
    out.extend_from_slice(&compressed);

    fs::write(path, &out).map_err(|e| e.to_string())?;
    Ok(())
}

/// Carga un escenario desde `path`.
///
/// Valida la cabecera (magic, versión, attractor_count <= 4) ANTES de
/// reservar memoria para los cuerpos. Devuelve un `NativeStore` con los
/// canales X/Y/Vx/Vy poblados y `live_count == body_count`.
pub fn load_scenario(path: &Path) -> Result<NativeStore, String> {
    let data = fs::read(path).map_err(|e| e.to_string())?;
    if data.len() < HEADER_SIZE {
        return Err(format!(
            "archivo demasiado corto para la cabecera ({} bytes, se requieren {})",
            data.len(),
            HEADER_SIZE
        ));
    }

    // --- Validación de cabecera ANTES de cualquier reserva ---
    if &data[0..4] != MAGIC.as_slice() {
        return Err("magic inválido: se esperaba b\"GRV1\"".to_string());
    }
    let version = read_u32_le(&data[4..8])?;
    if version != VERSION {
        return Err(format!("versión no soportada: {} (se espera {})", version, VERSION));
    }
    let attractor_count = read_u32_le(&data[8..12])?;
    if attractor_count > MAX_ATTRACTORS {
        return Err(format!(
            "attractor_count {} excede el máximo {}",
            attractor_count, MAX_ATTRACTORS
        ));
    }
    let body_count = read_u32_le(&data[12..16])? as usize;

    // --- Cuerpo: descomprimir gzip ---
    let compressed = &data[HEADER_SIZE..];
    let mut decoder = flate2::read::GzDecoder::new(compressed);
    let mut body = Vec::new();
    decoder.read_to_end(&mut body).map_err(|e| e.to_string())?;

    if body.len() < 4 {
        return Err("cuerpo comprimido sin body_count".to_string());
    }
    let body_count_decoded = read_u32_le(&body[0..4])? as usize;
    if body_count_decoded != body_count {
        return Err(format!(
            "body_count de cabecera ({}) no coincide con el cuerpo ({})",
            body_count, body_count_decoded
        ));
    }
    let expected = 4 + body_count * 16;
    if body.len() < expected {
        return Err(format!(
            "cuerpo comprimido truncado: {} bytes, se requieren {}",
            body.len(),
            expected
        ));
    }

    // --- Reserva de memoria (solo tras validar cabecera) ---
    if body_count == 0 {
        return Err("body_count debe ser > 0".to_string());
    }
    let mut store = NativeStore::new(body_count).map_err(|e| e.to_string())?;

    let x = store.x_slice();
    let y = store.y_slice();
    let vx = store.vx_slice();
    let vy = store.vy_slice();
    let mut off = 4usize;
    for i in 0..body_count {
        x[i] = read_f32_le(&body[off..off + 4])?;
        off += 4;
        y[i] = read_f32_le(&body[off..off + 4])?;
        off += 4;
        vx[i] = read_f32_le(&body[off..off + 4])?;
        off += 4;
        vy[i] = read_f32_le(&body[off..off + 4])?;
        off += 4;
    }
    store.live_count = body_count;
    Ok(store)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation_store::{allocate, seed_linear, SimConfig};

    fn config() -> SimConfig {
        SimConfig { body_count: 8, width: 100.0, height: 100.0 }
    }

    #[test]
    fn ac1_round_trip_save_load_recovers_xyvxvy() {
        let mut store = allocate(config()).unwrap();
        let speed = 10.0f32;
        seed_linear(&mut store, 100.0, 100.0, speed);

        let path = std::env::temp_dir().join("gravi_scenario_roundtrip.grav");
        let _ = save_scenario(&path, &store, 2);
        let loaded = load_scenario(&path).unwrap();

        assert_eq!(loaded.count, store.count);
        assert_eq!(loaded.live_count, store.count);
        let x = store.x_slice();
        let y = store.y_slice();
        let vx = store.vx_slice();
        let vy = store.vy_slice();
        let lx = loaded.x_slice();
        let ly = loaded.y_slice();
        let lvx = loaded.vx_slice();
        let lvy = loaded.vy_slice();
        for i in 0..store.count {
            assert_eq!(lx[i], x[i], "x[{}]", i);
            assert_eq!(ly[i], y[i], "y[{}]", i);
            assert_eq!(lvx[i], vx[i], "vx[{}]", i);
            assert_eq!(lvy[i], vy[i], "vy[{}]", i);
        }
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn ac2_attractor_count_gt_4_errors() {
        let store = allocate(config()).unwrap();
        let path = std::env::temp_dir().join("gravi_scenario_attractors.grav");
        assert!(save_scenario(&path, &store, 5).is_err());
        assert!(save_scenario(&path, &store, 4).is_ok());
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn ac3_header_validated_before_allocate_bad_magic() {
        let path = std::env::temp_dir().join("gravi_scenario_bad.grav");
        let mut data = vec![0u8; HEADER_SIZE + 10];
        data[0..4].copy_from_slice(b"XXXX"); // magic inválido
        fs::write(&path, &data).unwrap();
        assert!(load_scenario(&path).is_err());
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn ac4_header_validated_before_allocate_bad_version() {
        let path = std::env::temp_dir().join("gravi_scenario_badver.grav");
        let mut data = vec![0u8; HEADER_SIZE + 10];
        data[0..4].copy_from_slice(b"GRV1");
        data[4..8].copy_from_slice(&2u32.to_le_bytes()); // versión inválida
        fs::write(&path, &data).unwrap();
        assert!(load_scenario(&path).is_err());
        let _ = fs::remove_file(&path);
    }
}
