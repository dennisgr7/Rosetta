//! Utilidades compartidas entre los motores ASR (`parakeet`, `whisper`): helpers
//! de extracción de tensores ONNX, mapeo de errores de `ort` y un `argmax` único
//! y robusto. Antes estaban DUPLICADOS en `parakeet.rs` y `whisper/mod.rs`
//! (simplificar-1/4 + simplificar-3 + bugs-2).

use std::path::{Path, PathBuf};

use ndarray::{Array3, Array4};
use ort::value::DynValue;

use rosetta_core::{Result, RosettaError};

/// Devuelve el primer fichero de `names` que exista en `dir`. Error de modelo si
/// no hay ninguno.
pub(crate) fn find_first(dir: &Path, names: &[&str]) -> Result<PathBuf> {
    for n in names {
        let p = dir.join(n);
        if p.exists() {
            return Ok(p);
        }
    }
    Err(RosettaError::Model(format!(
        "no se encontró ninguno de {names:?} en {}",
        dir.display()
    )))
}

/// Mapea un error de `ort` a `RosettaError::Asr`.
pub(crate) fn asr_err(e: ort::Error) -> RosettaError {
    RosettaError::Asr(e.to_string())
}

/// Extrae un tensor f32 de 3 dims de un `DynValue`.
pub(crate) fn extract_3d(value: &DynValue, name: &str) -> Result<Array3<f32>> {
    let (shape, data) = value
        .try_extract_tensor::<f32>()
        .map_err(|e| RosettaError::Asr(format!("extraer {name}: {e}")))?;
    let d = shape.as_ref();
    if d.len() != 3 {
        return Err(RosettaError::Asr(format!(
            "{name}: 3 dims esperadas, {d:?}"
        )));
    }
    Array3::from_shape_vec((d[0] as usize, d[1] as usize, d[2] as usize), data.to_vec())
        .map_err(|e| RosettaError::Asr(format!("reshape {name}: {e}")))
}

/// Extrae un tensor f32 de 4 dims de un `DynValue`.
pub(crate) fn extract_4d(value: &DynValue, name: &str) -> Result<Array4<f32>> {
    let (shape, data) = value
        .try_extract_tensor::<f32>()
        .map_err(|e| RosettaError::Asr(format!("extraer {name}: {e}")))?;
    let d = shape.as_ref();
    if d.len() != 4 {
        return Err(RosettaError::Asr(format!(
            "{name}: 4 dims esperadas, {d:?}"
        )));
    }
    Array4::from_shape_vec(
        (d[0] as usize, d[1] as usize, d[2] as usize, d[3] as usize),
        data.to_vec(),
    )
    .map_err(|e| RosettaError::Asr(format!("reshape {name}: {e}")))
}

/// `argmax` único y robusto (simplificar-3 + bugs-2). Devuelve el índice del
/// **primer** máximo **finito** del iterador, o `None` si está vacío o si todos
/// los valores son NaN/±inf (no hay máximo finito).
///
/// La semilla es `NEG_INFINITY` y la condición es `v.is_finite() && v > max` con
/// `>` ESTRICTO: así se conserva el tie-break del primer máximo (a igualdad,
/// gana el de menor índice) y nunca se devuelve un índice espurio 0 ante una
/// entrada degenerada (todo-NaN, todo -inf o slice vacío) — el caller decide qué
/// hacer ante `None` (propagar error / abortar) en vez de seguir desde el token 0.
pub(crate) fn argmax<I: IntoIterator<Item = f32>>(it: I) -> Option<usize> {
    let mut best: Option<usize> = None;
    let mut max = f32::NEG_INFINITY;
    for (i, v) in it.into_iter().enumerate() {
        if v.is_finite() && v > max {
            max = v;
            best = Some(i);
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::argmax;

    #[test]
    fn argmax_normal_primer_maximo() {
        // máximo único
        assert_eq!(argmax([0.1_f32, 0.9, 0.3]), Some(1));
        // empate: gana el de MENOR índice (tie-break del primer máximo)
        assert_eq!(argmax([0.5_f32, 0.5, 0.2]), Some(0));
        // un solo elemento finito
        assert_eq!(argmax([42.0_f32]), Some(0));
    }

    #[test]
    fn argmax_todo_nan_es_none() {
        let v = [f32::NAN, f32::NAN, f32::NAN];
        assert_eq!(argmax(v), None);
    }

    #[test]
    fn argmax_todo_neg_inf_es_none() {
        let v = [f32::NEG_INFINITY, f32::NEG_INFINITY];
        assert_eq!(argmax(v), None);
    }

    #[test]
    fn argmax_vacio_es_none() {
        let v: [f32; 0] = [];
        assert_eq!(argmax(v), None);
    }

    #[test]
    fn argmax_ignora_no_finitos_intercalados() {
        // +inf y NaN se ignoran; gana el mayor finito (0.7 en índice 2).
        assert_eq!(argmax([f32::INFINITY, f32::NAN, 0.7_f32, 0.2]), Some(2));
    }
}
