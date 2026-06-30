//! Segmentador `pyannote-segmentation-3.0` (Bloque E2): fronteras de voz y
//! regiones de solape a partir del modelo ONNX powerset.
//!
//! El modelo recibe **waveform crudo** 16 kHz `x[N,1,T]` (NO fbank) y devuelve
//! `y[N,frames,7]` en codificación **powerset**: por frame, una de 7 clases que
//! representa qué subconjunto de los 3 hablantes locales está activo
//! (0=silencio, 1/2/3=un solo hablante, 4/5/6=pares). El número de `frames` se
//! deriva de la shape de salida (NO se hardcodea).
//!
//! Se procesa el audio por ventanas fijas de 10 s con un *stride* configurable.
//! Cada ventana se decodifica por **argmax** (NO softmax+suma) a multilabel
//! `[frames,3]` (hablantes LOCALES, no comparables entre ventanas), se suaviza
//! con un filtro de mediana temporal y se acumula con **overlap-add** sobre un
//! grid global usando SOLO magnitudes invariantes a permutación: nº de hablantes
//! activos por frame. De ese conteo salen la máscara de voz (`>=1`) y la de
//! solape (`>=2`), y de ahí las fronteras (con histéresis) y los tramos de
//! solape. **No** se hace *stitching* de identidades locales entre ventanas: la
//! identidad global la resuelve CAM+++clustering aguas abajo.

use std::path::Path;

use anyhow::{Context, Result};
use ndarray::Array3;
use ort::session::Session;
use ort::value::Value;

use rosetta_accel::{Device, HwProfile};
use rosetta_audio::{AudioPcm, SpeechSegment};

/// Frecuencia de muestreo de trabajo (Hz). pyannote-segmentation-3.0 es 16 kHz.
const SR: usize = 16_000;
/// Tamaño de ventana de inferencia: 10 s @ 16 kHz.
const WIN: usize = 160_000;
/// *Stride* por defecto: 5 s @ 16 kHz (50 % de solape entre ventanas).
const DEFAULT_STRIDE: usize = 80_000;
/// Máximo de hablantes locales simultáneos que codifica el powerset.
const LOCAL_SPEAKERS: usize = 3;
/// Nº de clases del powerset (silencio + 3 solos + 3 pares).
const POWERSET_CLASSES: usize = 7;

/// Ventana del filtro de mediana temporal (frames). Impar para que la mediana
/// caiga en un elemento. ~11 frames ≈ 0,18 s con `frame_dur ≈ 16,4 ms`.
const MEDIAN_WIN: usize = 11;

/// Tabla powerset: clase (0..7) → activación de los 3 hablantes locales.
/// 0=silencio, 1/2/3=un solo hablante, 4/5/6=pares.
const POWERSET_MAPPING: [[u8; LOCAL_SPEAKERS]; POWERSET_CLASSES] = [
    [0, 0, 0], // 0: silencio
    [1, 0, 0], // 1: hablante 0
    [0, 1, 0], // 2: hablante 1
    [0, 0, 1], // 3: hablante 2
    [1, 1, 0], // 4: hablantes 0+1
    [1, 0, 1], // 5: hablantes 0+2
    [0, 1, 1], // 6: hablantes 1+2
];

/// Mínima duración de un tramo de voz (frames descartados por debajo). ~10
/// frames ≈ 170 ms con `frame_dur ≈ 16,4 ms`.
const MIN_SPEECH_FRAMES: usize = 10;
/// Mínima duración de un tramo de solape (s).
const MIN_OVERLAP_S: f32 = 0.1;

/// Holgura para el umbral de solape sobre la media ESTRICTA de hablantes
/// (`count/weight`, sin redondeo half-up). Una franja se marca como solape solo
/// si la media `>= 2.0 - OVERLAP_EPSILON`, es decir si la MAYORÍA de las ventanas
/// que la cubren ve ≥2 hablantes. Con 0.25, la franja de solape entre ventanas
/// (stride 5 s) donde una ventana ve 2 y la otra 1 da media 1.5 y NO marca (evita
/// el solape espurio sistemático cada ~5 s); media ≥1.75 sí marca.
const OVERLAP_EPSILON: f32 = 0.25;

/// Histéresis sobre la cuenta de hablantes activos para fijar fronteras de voz.
/// Entra en voz al cruzar `>= ON` y sale al caer por debajo de `OFF` (evita el
/// parpadeo en los bordes). Trabaja sobre `speakers_per_frame` (≥1 = hay voz).
const HYST_ON: f32 = 1.0;
const HYST_OFF: f32 = 0.5;

/// Resultado de [`PyannoteSegmenter::segment`]: regiones de voz (con histéresis)
/// y tramos de solape de hablantes `(inicio_s, fin_s)`.
pub type Segmentation = (Vec<SpeechSegment>, Vec<(f32, f32)>);

/// Segmentador pyannote: sesión ONNX (CPU) + parámetros de ventaneo.
pub struct PyannoteSegmenter {
    session: Session,
    /// Salto entre ventanas consecutivas (muestras). `< WIN` ⇒ solape.
    stride_samples: usize,
}

impl PyannoteSegmenter {
    /// Carga el modelo `pyannote-segmentation-3.0` en una sesión CPU (las
    /// ventanas pequeñas no rentan acelerador) con `stride_samples` muestras de
    /// salto entre ventanas (0 ⇒ por defecto 5 s).
    pub fn from_file(
        model: &Path,
        hw: &HwProfile,
        threads: usize,
        stride_samples: usize,
    ) -> Result<Self> {
        let (session, _ep) = rosetta_accel::ep::build_session(model, hw, Device::Cpu, threads)
            .map_err(|e| anyhow::anyhow!("segmenter: build_session: {e}"))?;
        let stride_samples = if stride_samples == 0 {
            DEFAULT_STRIDE
        } else {
            stride_samples.min(WIN)
        };
        Ok(Self {
            session,
            stride_samples,
        })
    }

    /// Segmenta el audio mono 16 kHz. Devuelve las regiones de voz (fronteras
    /// con histéresis) y los tramos de solape de hablantes, ambos en segundos.
    pub fn segment(&mut self, audio: &AudioPcm) -> Result<Segmentation> {
        anyhow::ensure!(
            audio.sample_rate as usize == SR,
            "segmenter: se esperaba audio a {SR} Hz, recibido {} Hz",
            audio.sample_rate
        );
        let pcm = &audio.samples;
        if pcm.is_empty() {
            return Ok((vec![], vec![]));
        }

        // Primera ventana para fijar el grid global (frame_dur). Todas las
        // ventanas comparten geometría, así que basta inferir una para conocer
        // `frames` y derivar `frame_dur_s`.
        let first = self.run_window(pcm, 0)?;
        let frames = first.len();
        anyhow::ensure!(
            (560..=780).contains(&frames),
            "segmenter: nº de frames fuera de rango esperado: {frames} (≈[560,780])"
        );
        let frame_dur_s = (WIN as f32 / frames as f32) / SR as f32;

        // Acumuladores del grid global. `count`: suma de hablantes locales
        // activos (invariante a permutación); `weight`: nº de ventanas que
        // cubren ese frame.
        let total_dur = pcm.len() as f32 / SR as f32;
        let n_grid = (total_dur / frame_dur_s).ceil() as usize + frames + 1;
        let mut count = vec![0.0f32; n_grid];
        let mut weight = vec![0.0f32; n_grid];

        // Volcar cada ventana (la primera ya inferida) al grid.
        let mut off = 0usize;
        let mut first = Some(first);
        while off < pcm.len() {
            let multilabel = match first.take() {
                Some(m) => m, // reutiliza la ventana 0
                None => self.run_window(pcm, off)?,
            };
            overlap_add(&multilabel, off, frame_dur_s, &mut count, &mut weight);
            off += self.stride_samples;
        }

        // Fronteras de voz: conteo half-up (voz ≥1 ⇒ media ≥0.5 redondea a 1).
        let speakers_per_frame = speakers_per_frame(&count, &weight);
        let segments = boundaries(&speakers_per_frame, frame_dur_s);
        // Solape: máscara con media ESTRICTA (sin half-up) para no disparar en la
        // franja de solape entre ventanas donde la media es 1.5 (bugs-3).
        let overlap_mask = overlap_mask(&count, &weight);
        let overlaps = overlap_regions(&overlap_mask, frame_dur_s);
        Ok((segments, overlaps))
    }

    /// Infiere una ventana de 10 s a partir de `off` (zero-pad si se sale del
    /// audio) y devuelve el multilabel `[frames][3]` decodificado por argmax.
    fn run_window(&mut self, pcm: &[f32], off: usize) -> Result<Vec<[u8; LOCAL_SPEAKERS]>> {
        // Ventana fija de WIN muestras con zero-pad al final.
        let mut buf = vec![0.0f32; WIN];
        let end = (off + WIN).min(pcm.len());
        if off < pcm.len() {
            let n = end - off;
            buf[..n].copy_from_slice(&pcm[off..end]);
        }
        let input = Array3::from_shape_vec((1, 1, WIN), buf)
            .context("segmenter: crear tensor de entrada (1,1,160000)")?;
        let outputs = self
            .session
            .run(ort::inputs!("x" => Value::from_array(input).context("segmenter: Value::from_array")?))
            .map_err(|e| anyhow::anyhow!("segmenter: inferencia ort: {e}"))?;
        anyhow::ensure!(
            outputs.len() != 0,
            "segmenter: el modelo no devolvió salidas"
        );
        // El contrato fija una única salida posicional [N,frames,7] (pyannote la
        // nombra "y"); se indexa por posición para no depender del nombre.
        let out = &outputs[0];
        let (shape, data) = out
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("segmenter: extraer tensor de salida: {e}"))?;
        let d = shape.as_ref();
        anyhow::ensure!(
            d.len() == 3,
            "segmenter: salida con {} dims, se esperaban 3 [N,frames,7]",
            d.len()
        );
        let frames = d[1] as usize;
        let classes = d[2] as usize;
        anyhow::ensure!(
            classes == POWERSET_CLASSES,
            "segmenter: se esperaban {POWERSET_CLASSES} clases powerset, hay {classes}"
        );
        Ok(decode_powerset(data, frames, classes))
    }
}

/// Overlap-add de una ventana al grid global. Acumula SOLO invariantes a
/// permutación: por cada frame local, el conteo de hablantes locales activos
/// (0..3) se suma a `count[g]` y `weight[g]` cuenta la cobertura.
fn overlap_add(
    multilabel: &[[u8; LOCAL_SPEAKERS]],
    off_samples: usize,
    frame_dur_s: f32,
    count: &mut [f32],
    weight: &mut [f32],
) {
    let off_s = off_samples as f32 / SR as f32;
    for (f, ml) in multilabel.iter().enumerate() {
        let g = ((off_s + f as f32 * frame_dur_s) / frame_dur_s).round() as usize;
        if g >= count.len() {
            continue;
        }
        let active = ml.iter().filter(|&&v| v != 0).count() as f32;
        count[g] += active;
        weight[g] += 1.0;
    }
}

/// Media de hablantes activos por frame del grid (redondeada al entero más
/// próximo). `weight + 1e-9` evita dividir por cero en frames sin cobertura.
fn speakers_per_frame(count: &[f32], weight: &[f32]) -> Vec<f32> {
    count
        .iter()
        .zip(weight)
        .map(|(&c, &w)| ((c / (w + 1e-9)) + 0.5).floor())
        .collect()
}

/// Máscara de solape por frame (bugs-3): `true` solo si la media ESTRICTA de
/// hablantes (`count/weight`, sin redondeo half-up) `>= 2.0 - OVERLAP_EPSILON`,
/// es decir si la mayoría de las ventanas que cubren el frame ve ≥2 hablantes.
/// Separar este umbral del conteo de voz evita el solape espurio en la franja de
/// solape entre ventanas, donde la media 1.5 redondearía a 2 con half-up.
fn overlap_mask(count: &[f32], weight: &[f32]) -> Vec<bool> {
    let thr = 2.0 - OVERLAP_EPSILON;
    count
        .iter()
        .zip(weight)
        .map(|(&c, &w)| (c / (w + 1e-9)) >= thr)
        .collect()
}

/// Fronteras de voz a partir de `speakers_per_frame` (≥1 = hay voz) por
/// histéresis, descartando tramos más cortos que `MIN_SPEECH_FRAMES`.
fn boundaries(speakers_per_frame: &[f32], frame_dur_s: f32) -> Vec<SpeechSegment> {
    let mut segments = Vec::new();
    let mut triggered = false;
    let mut start_f = 0usize;
    for (f, &s) in speakers_per_frame.iter().enumerate() {
        if !triggered && s >= HYST_ON {
            triggered = true;
            start_f = f;
        } else if triggered && s < HYST_OFF {
            push_segment(&mut segments, start_f, f, frame_dur_s);
            triggered = false;
        }
    }
    if triggered {
        push_segment(
            &mut segments,
            start_f,
            speakers_per_frame.len(),
            frame_dur_s,
        );
    }
    segments
}

/// Tramos de solape a partir de la máscara de solape (`true` = solape, ver
/// [`overlap_mask`]), fusionando frames contiguos y descartando tramos más
/// cortos que `MIN_OVERLAP_S`.
fn overlap_regions(mask: &[bool], frame_dur_s: f32) -> Vec<(f32, f32)> {
    let mut out = Vec::new();
    let mut in_ov = false;
    let mut start_f = 0usize;
    for (f, &ov) in mask.iter().enumerate() {
        if !in_ov && ov {
            in_ov = true;
            start_f = f;
        } else if in_ov && !ov {
            push_overlap(&mut out, start_f, f, frame_dur_s);
            in_ov = false;
        }
    }
    if in_ov {
        push_overlap(&mut out, start_f, mask.len(), frame_dur_s);
    }
    out
}

/// Cierra un tramo de voz `[start_f, end_f)` si supera el mínimo de frames.
fn push_segment(out: &mut Vec<SpeechSegment>, start_f: usize, end_f: usize, frame_dur_s: f32) {
    if end_f.saturating_sub(start_f) >= MIN_SPEECH_FRAMES {
        out.push(SpeechSegment {
            start_s: start_f as f32 * frame_dur_s,
            end_s: end_f as f32 * frame_dur_s,
        });
    }
}

/// Cierra un tramo de solape `[start_f, end_f)` si supera el mínimo de duración.
fn push_overlap(out: &mut Vec<(f32, f32)>, start_f: usize, end_f: usize, frame_dur_s: f32) {
    let a = start_f as f32 * frame_dur_s;
    let b = end_f as f32 * frame_dur_s;
    if b - a >= MIN_OVERLAP_S {
        out.push((a, b));
    }
}

/// Decodifica `[frames,classes]` powerset a multilabel `[frames][3]` por
/// **argmax** (NO softmax+suma): por frame se elige la clase de mayor logit y se
/// mapea con [`POWERSET_MAPPING`]. Tras el argmax se aplica un filtro de mediana
/// temporal por columna para quitar el parpadeo de 1 frame.
fn decode_powerset(data: &[f32], frames: usize, classes: usize) -> Vec<[u8; LOCAL_SPEAKERS]> {
    let mut ml = vec![[0u8; LOCAL_SPEAKERS]; frames];
    for (f, slot) in ml.iter_mut().enumerate() {
        let base = f * classes;
        let row = &data[base..base + classes];
        let col = argmax(row);
        *slot = POWERSET_MAPPING[col.min(POWERSET_CLASSES - 1)];
    }
    median_filter(&mut ml);
    ml
}

/// Índice del máximo de `row` (primero en caso de empate). Solo considera
/// valores finitos: ante un slice todo-NaN/-inf (o vacío) devuelve 0 sin que un
/// NaN/-inf colado falsee el máximo (`v > max` con `max=NEG_INFINITY` aceptaría
/// el primer elemento aunque fuese -inf).
fn argmax(row: &[f32]) -> usize {
    let mut idx = 0;
    let mut max = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if v.is_finite() && v > max {
            max = v;
            idx = i;
        }
    }
    idx
}

/// Filtro de mediana temporal (`MEDIAN_WIN` frames) por columna, in-place. Como
/// cada columna es binaria, la mediana de la ventana equivale a la mayoría;
/// elimina parpadeos aislados de 1 frame.
fn median_filter(ml: &mut [[u8; LOCAL_SPEAKERS]]) {
    let n = ml.len();
    if n < MEDIAN_WIN {
        return;
    }
    let half = MEDIAN_WIN / 2;
    let orig: Vec<[u8; LOCAL_SPEAKERS]> = ml.to_vec();
    let mut window = [0u8; MEDIAN_WIN];
    for (f, slot) in ml.iter_mut().enumerate() {
        let lo = f.saturating_sub(half);
        let hi = (f + half + 1).min(n);
        let len = hi - lo;
        for (c, out) in slot.iter_mut().enumerate() {
            for (k, w) in window[..len].iter_mut().enumerate() {
                *w = orig[lo + k][c];
            }
            let mid = len / 2;
            let (_, median, _) = window[..len].select_nth_unstable(mid);
            *out = *median;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// (a) decode por argmax → multilabel para las 7 filas del powerset.
    #[test]
    fn decode_argmax_cubre_las_siete_clases() {
        // 7 frames, cada uno con el máximo en su propia clase.
        let classes = POWERSET_CLASSES;
        let mut data = vec![0.0f32; 7 * classes];
        for (f, chunk) in data.chunks_mut(classes).enumerate() {
            chunk[f] = 10.0; // argmax = f
        }
        // Sin filtro de mediana (7 < MEDIAN_WIN), el decode es directo.
        let ml = decode_powerset(&data, 7, classes);
        assert_eq!(ml.len(), 7);
        for (clase, esperado) in POWERSET_MAPPING.iter().enumerate() {
            assert_eq!(ml[clase], *esperado, "clase {clase}");
        }
    }

    /// El argmax elige el mayor logit aunque otros sean positivos (NO softmax+suma).
    #[test]
    fn decode_argmax_elige_el_mayor_logit() {
        let classes = POWERSET_CLASSES;
        // clase 4 = [1,1,0] gana pese a que la 1 y la 2 también son altas.
        let mut row = vec![0.0f32; classes];
        row[1] = 0.8;
        row[2] = 0.7;
        row[4] = 0.9;
        let ml = decode_powerset(&row, 1, classes);
        assert_eq!(ml[0], [1, 1, 0]);
    }

    /// (c) la mediana quita un parpadeo aislado de 1 frame.
    #[test]
    fn mediana_quita_parpadeo_de_un_frame() {
        // 21 frames de hablante 0 activo con un único frame apagado en medio.
        let mut ml = vec![[1u8, 0, 0]; 21];
        ml[10] = [0, 0, 0]; // parpadeo aislado
        median_filter(&mut ml);
        assert_eq!(ml[10], [1, 0, 0], "el parpadeo debe rellenarse por mayoría");
        // Y un parpadeo de encendido aislado debe apagarse.
        let mut ml2 = vec![[0u8, 0, 0]; 21];
        ml2[10] = [1, 0, 0];
        median_filter(&mut ml2);
        assert_eq!(ml2[10], [0, 0, 0], "el encendido aislado debe quitarse");
    }

    /// Reproduce el overlap-add sobre el grid global tal como lo hace `segment`,
    /// para poder testearlo sin sesión ONNX (las funciones son libres y puras).
    fn overlap_add_grid(
        windows: &[(usize, Vec<[u8; LOCAL_SPEAKERS]>)],
        frame_dur_s: f32,
        n_grid: usize,
    ) -> Vec<f32> {
        let mut count = vec![0.0f32; n_grid];
        let mut weight = vec![0.0f32; n_grid];
        for (off, ml) in windows {
            overlap_add(ml, *off, frame_dur_s, &mut count, &mut weight);
        }
        speakers_per_frame(&count, &weight)
    }

    /// (b) overlap-add con 2 ventanas de offsets distintos: coherente e
    /// invariante a permutar las columnas locales.
    #[test]
    fn overlap_add_invariante_a_permutacion() {
        let frame_dur_s = (WIN as f32 / 600.0) / SR as f32;
        // 600 frames por ventana; stride de medio segundo en frames.
        let frames = 600usize;
        // Ventana A en off=0: un hablante (col 0) activo en [0,300), dos en [300,600).
        let mut wa = vec![[0u8; LOCAL_SPEAKERS]; frames];
        for f in wa.iter_mut().take(300) {
            *f = [1, 0, 0];
        }
        for f in wa.iter_mut().skip(300) {
            *f = [1, 1, 0]; // dos hablantes (solape)
        }
        // Ventana B en off = 300 frames (en muestras), con las MISMAS magnitudes
        // pero columnas permutadas (col1/col2 en vez de col0/col1): debe dar el
        // mismo conteo de hablantes activos.
        let off_b_samples = (300.0 * frame_dur_s * SR as f32).round() as usize;
        let mut wb = vec![[0u8; LOCAL_SPEAKERS]; frames];
        for f in wb.iter_mut().take(300) {
            *f = [0, 1, 0]; // un hablante, columna distinta
        }
        for f in wb.iter_mut().skip(300) {
            *f = [0, 1, 1]; // dos hablantes, columnas distintas
        }

        let n_grid = 1100;
        let g0 = overlap_add_grid(
            &[(0, wa.clone()), (off_b_samples, wb.clone())],
            frame_dur_s,
            n_grid,
        );

        // En el grid global, alrededor del frame 450 (zona [300,600) de A y
        // [150,300) de B) hay solape en A (2) y un solo hablante en B (1):
        // media = 1.5 → redondea a 2 (hay solape).
        assert!(
            g0[450] >= 2.0,
            "debe detectar solape donde A=2,B=1: {}",
            g0[450]
        );
        // Cerca del frame 100 solo cubre A con 1 hablante → 1.
        assert_eq!(g0[100], 1.0);
        // Cerca del frame 800 solo cubre B con 2 hablantes → 2.
        assert!(g0[800] >= 2.0, "solape de B al final: {}", g0[800]);

        // Invariancia a permutación: permutar las columnas locales de AMBAS
        // ventanas no cambia el conteo global.
        let perm = |ml: &[[u8; LOCAL_SPEAKERS]]| -> Vec<[u8; LOCAL_SPEAKERS]> {
            ml.iter().map(|r| [r[2], r[0], r[1]]).collect()
        };
        let wap = perm(&wa);
        let wbp = perm(&wb);
        let g1 = overlap_add_grid(&[(0, wap), (off_b_samples, wbp)], frame_dur_s, n_grid);
        assert_eq!(
            g0, g1,
            "el conteo global debe ser invariante a permutar columnas locales"
        );
    }

    /// boundaries: la histéresis produce un tramo de voz y descarta los cortos.
    #[test]
    fn boundaries_descarta_tramos_cortos() {
        let frame_dur_s = 0.0164;
        let mut spf = vec![0.0f32; 100];
        // Tramo largo de voz [20,60) → 40 frames, se conserva.
        for s in spf.iter_mut().take(60).skip(20) {
            *s = 1.0;
        }
        // Parpadeo corto en [80,83) → 3 frames (< MIN_SPEECH_FRAMES), se descarta.
        for s in spf.iter_mut().take(83).skip(80) {
            *s = 1.0;
        }
        let segs = boundaries(&spf, frame_dur_s);
        assert_eq!(segs.len(), 1, "solo el tramo largo sobrevive");
        assert!((segs[0].start_s - 20.0 * frame_dur_s).abs() < 1e-4);
        assert!((segs[0].end_s - 60.0 * frame_dur_s).abs() < 1e-4);
    }

    /// overlap_regions: fusiona contiguos y descarta tramos < MIN_OVERLAP_S.
    #[test]
    fn overlap_regions_filtra_por_duracion() {
        let frame_dur_s = 0.05; // 20 frames = 1 s
        let mut mask = vec![false; 60];
        // Solape largo [10,40) → 30 frames = 1.5 s, se conserva.
        for m in mask.iter_mut().take(40).skip(10) {
            *m = true;
        }
        // Solape corto de 1 frame en 50 → 0.05 s < 0.1 s, se descarta.
        mask[50] = true;
        let ov = overlap_regions(&mask, frame_dur_s);
        assert_eq!(ov.len(), 1, "solo el solape largo sobrevive");
        assert!((ov[0].0 - 10.0 * frame_dur_s).abs() < 1e-4);
        assert!((ov[0].1 - 40.0 * frame_dur_s).abs() < 1e-4);
    }

    /// bugs-3 (a): franja de solape entre ventanas donde SOLO una de las dos ve
    /// ≥2 hablantes (media 1.5). El conteo de voz half-up redondea a 2, pero la
    /// MÁSCARA DE SOLAPE (media estricta ≥1.75) NO la marca ⇒ sin solape espurio.
    #[test]
    fn overlap_mask_no_marca_media_1_5() {
        // Un frame cubierto por 2 ventanas: una ve 2 hablantes, otra ve 1.
        let count = vec![3.0f32]; // 2 + 1
        let weight = vec![2.0f32]; // dos ventanas
        // Conteo de voz half-up: (1.5 + 0.5).floor() = 2.0 (es voz, correcto).
        let spf = speakers_per_frame(&count, &weight);
        assert_eq!(spf[0], 2.0, "half-up sí redondea 1.5 a 2 (conteo de voz)");
        // Máscara de solape con media estricta: 1.5 < 1.75 ⇒ NO marca.
        let mask = overlap_mask(&count, &weight);
        assert!(!mask[0], "media 1.5 NO debe marcarse como solape");
        // Y por tanto overlap_regions no produce ningún tramo.
        assert!(overlap_regions(&mask, 0.05).is_empty());
    }

    /// bugs-3 (b): cuando AMBAS ventanas ven ≥2 hablantes (media ≥2) la máscara
    /// SÍ marca solape.
    #[test]
    fn overlap_mask_marca_media_2() {
        // Dos ventanas, ambas con 2 hablantes ⇒ media 2.0.
        let count = vec![4.0f32]; // 2 + 2
        let weight = vec![2.0f32];
        let mask = overlap_mask(&count, &weight);
        assert!(mask[0], "media 2.0 sí debe marcarse como solape");
        // Caso límite: media 1.75 (= umbral) también marca (mayoría ve ≥2).
        let mask2 = overlap_mask(&[3.5], &[2.0]); // (2+1.5)? media 1.75
        assert!(mask2[0], "media 1.75 (= umbral) sí marca");
    }

    /// bugs-2-local: argmax con un slice todo-NaN devuelve 0 sin que un NaN
    /// falsee el máximo (`v.is_finite() && v > max`).
    #[test]
    fn argmax_todo_nan_devuelve_cero() {
        let nan = f32::NAN;
        assert_eq!(argmax(&[nan, nan, nan]), 0);
        // -inf tampoco debe considerarse máximo.
        assert_eq!(argmax(&[f32::NEG_INFINITY, f32::NEG_INFINITY]), 0);
        // Slice vacío → 0.
        assert_eq!(argmax(&[]), 0);
        // Con un finito entre NaN, gana el finito.
        assert_eq!(argmax(&[nan, 1.0, nan]), 1);
    }
}
