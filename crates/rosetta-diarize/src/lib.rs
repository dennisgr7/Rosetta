//! Diarización de hablantes (F6), Rust puro sobre `ort`.
//!
//! Estrategia v1: el VAD (Silero) segmenta la voz, se computa un **embedding de
//! hablante** (CAM++ de 3D-Speaker, 192-d) por tramo a partir de su fbank
//! (`kaldi-native-fbank`), y se **agrupan** los tramos por similitud coseno
//! (clustering aglomerativo, `kodama`). Cada tramo recibe un id de hablante que
//! luego se fusiona con la transcripción. (El modelo pyannote-segmentation, ya
//! cacheado, permitirá en una mejora futura manejar solapes de hablantes.)

use std::path::Path;

use kaldi_native_fbank::online::FeatureComputer;
use kaldi_native_fbank::{FbankComputer, FbankOptions, OnlineFeature};
use kodama::{Method, linkage};
use ndarray::{Array2, Axis};
use ort::session::Session;
use ort::value::Value;

use rosetta_accel::{Device, HwProfile};
use rosetta_audio::{AudioPcm, SileroVad, SpeechSegment};
use rosetta_core::{Result, RosettaError, Segment, Transcript, Word};

pub mod segmenter;
pub use segmenter::PyannoteSegmenter;

const SR: f32 = 16_000.0;
const MEL_BINS: usize = 80;

/// Turno de habla atribuido a un hablante.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpeakerTurn {
    pub start_s: f32,
    pub end_s: f32,
    pub speaker: usize,
}

/// Resultado de la diarización: turnos de habla (identidad por CAM+++clustering) y
/// tramos de solape de hablantes `(inicio_s, fin_s)` detectados por pyannote.
pub type Diarization = (Vec<SpeakerTurn>, Vec<(f32, f32)>);

/// Configuración de la diarización.
#[derive(Debug, Clone, Copy)]
pub struct DiarizeConfig {
    /// Nº de hablantes si se conoce (corta el dendrograma en ese nº de clusters).
    /// Si es `None`, se estima automáticamente por el **codo** del dendrograma.
    pub max_speakers: Option<usize>,
    /// Umbral de distancia coseno: tope de seguridad para fusionar clusters cuando
    /// no hay un codo claro (1 solo hablante).
    pub threshold: f32,
    /// Turnos contiguos del mismo hablante separados por menos de este hueco (ms)
    /// se fusionan, reduciendo la fragmentación. 0 desactiva el coalesce.
    pub coalesce_gap_ms: u32,
}

impl Default for DiarizeConfig {
    fn default() -> Self {
        Self {
            max_speakers: None,
            threshold: 0.55,
            coalesce_gap_ms: 500,
        }
    }
}

/// Diarizador: VAD + embedder CAM++ + segmentador pyannote (para solapes).
pub struct Diarizer {
    vad: SileroVad,
    embed: Session,
    seg: PyannoteSegmenter,
    cfg: DiarizeConfig,
}

impl Diarizer {
    /// Carga el VAD, el modelo de embeddings y el segmentador pyannote (todos CPU).
    pub fn new(
        vad_model: &Path,
        emb_model: &Path,
        seg_model: &Path,
        hw: &HwProfile,
        threads: usize,
        cfg: DiarizeConfig,
    ) -> Result<Self> {
        let vad = SileroVad::from_file(vad_model, hw, threads)?;
        let (embed, _ep) = rosetta_accel::ep::build_session(emb_model, hw, Device::Cpu, threads)?;
        let seg = PyannoteSegmenter::from_file(seg_model, hw, threads, 0).map_err(de)?;
        Ok(Self {
            vad,
            embed,
            seg,
            cfg,
        })
    }

    /// Embedding de hablante (192-d) de un tramo de audio mono 16 kHz.
    pub fn embedding(&mut self, samples: &[f32]) -> Result<Vec<f32>> {
        let feats = fbank(samples)?; // [T, 80] con media global restada
        let t = feats.nrows();
        if t == 0 {
            return Ok(vec![0.0; 192]);
        }
        let input = feats.into_shape_with_order((1, t, MEL_BINS)).map_err(de)?;
        let outputs = self
            .embed
            .run(ort::inputs!("x" => Value::from_array(input).map_err(de)?))
            .map_err(de)?;
        let out = outputs
            .get("embedding")
            .ok_or_else(|| de("la salida 'embedding' no está en el modelo"))?;
        let (_s, emb) = out.try_extract_tensor::<f32>().map_err(de)?;
        Ok(emb.to_vec())
    }

    /// Diariza el audio: turnos de habla (identidad por CAM+++clustering) y los
    /// tramos de solape de hablantes detectados por pyannote.
    pub fn diarize(&mut self, audio: &AudioPcm) -> Result<Diarization> {
        let speech = self.vad.detect(audio)?;
        self.diarize_with_segments(audio, &speech)
    }

    /// Como [`Diarizer::diarize`] pero reutilizando segmentos de voz ya computados
    /// (evita la 2ª pasada de VAD cuando el pipeline ya troceó el audio largo;
    /// opt #3). `diarize` con VAD propio queda para el camino de audio corto.
    pub fn diarize_with_segments(
        &mut self,
        audio: &AudioPcm,
        speech: &[SpeechSegment],
    ) -> Result<Diarization> {
        if speech.is_empty() {
            return Ok((vec![], vec![]));
        }
        let mut embs = Vec::with_capacity(speech.len());
        for s in speech {
            let a = ((s.start_s * SR) as usize).min(audio.samples.len());
            let b = ((s.end_s * SR) as usize).min(audio.samples.len());
            embs.push(self.embedding(&audio.samples[a..b])?);
        }
        let labels = cluster(&embs, self.cfg.max_speakers, self.cfg.threshold);
        let mut turns: Vec<SpeakerTurn> = speech
            .iter()
            .zip(labels)
            .map(|(s, speaker)| SpeakerTurn {
                start_s: s.start_s,
                end_s: s.end_s,
                speaker,
            })
            .collect();
        coalesce_turns(&mut turns, self.cfg.coalesce_gap_ms as f32 / 1000.0);

        // Pyannote SOLO para fronteras+solapes; la identidad la dan los turns.
        let (_boundaries, overlaps) = self.seg.segment(audio).map_err(de)?;
        Ok((turns, overlaps))
    }
}

/// Fusiona turnos contiguos del MISMO hablante separados por un hueco menor que
/// `gap_s`, reduciendo la fragmentación que deja el troceo del VAD. Ordena por
/// tiempo de inicio antes de fusionar. `gap_s <= 0` no fusiona nada.
pub fn coalesce_turns(turns: &mut Vec<SpeakerTurn>, gap_s: f32) {
    if turns.len() < 2 || gap_s <= 0.0 {
        return;
    }
    turns.sort_by(|a, b| {
        a.start_s
            .partial_cmp(&b.start_s)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut out: Vec<SpeakerTurn> = Vec::with_capacity(turns.len());
    for t in turns.drain(..) {
        if let Some(last) = out.last_mut()
            && last.speaker == t.speaker
            && t.start_s - last.end_s < gap_s
        {
            last.end_s = last.end_s.max(t.end_s);
            continue;
        }
        out.push(t);
    }
    *turns = out;
}

/// Rellena `Segment.speaker` de la transcripción según los turnos de hablante.
/// Usa los timestamps de palabra (mayoría) si existen; si no, el solape del
/// segmento completo.
pub fn assign_speakers(transcript: &mut Transcript, turns: &[SpeakerTurn]) {
    if turns.is_empty() {
        return;
    }
    for seg in &mut transcript.segments {
        let label = if seg.words.is_empty() {
            best_overlap(seg.start, seg.end, turns, NEAREST_TURN_MAX_GAP_S)
        } else {
            let mut votes = std::collections::HashMap::<usize, usize>::new();
            for w in &seg.words {
                if let Some(sp) = best_overlap(w.start, w.end, turns, NEAREST_TURN_MAX_GAP_S) {
                    *votes.entry(sp).or_default() += 1;
                }
            }
            // Desempate determinista: más votos y, a igualdad, menor id de hablante.
            votes
                .into_iter()
                .max_by_key(|&(sp, n)| (n, std::cmp::Reverse(sp)))
                .map(|(sp, _)| sp)
        };
        seg.speaker = label.map(|l| format!("Hablante {}", l + 1));
    }
}

/// Re-segmenta la transcripción por hablante: agrupa palabras consecutivas del
/// mismo hablante en un segmento propio (un turno = un segmento). Si no hay
/// palabras, cae a [`assign_speakers`].
pub fn segment_by_speaker(transcript: &mut Transcript, turns: &[SpeakerTurn]) {
    if turns.is_empty() {
        return;
    }
    let all_words: Vec<Word> = transcript
        .segments
        .iter()
        .flat_map(|s| s.words.iter().cloned())
        .collect();
    if all_words.is_empty() {
        assign_speakers(transcript, turns);
        return;
    }

    let mut new_segments: Vec<Segment> = Vec::new();
    let mut cur_sp: Option<usize> = None;
    for w in all_words {
        let sp = best_overlap(w.start, w.end, turns, NEAREST_TURN_MAX_GAP_S);
        let extend = !new_segments.is_empty() && cur_sp == sp;
        if extend {
            let seg = new_segments.last_mut().unwrap();
            seg.end = w.end;
            seg.text.push(' ');
            seg.text.push_str(&w.text);
            seg.words.push(w);
        } else {
            cur_sp = sp;
            let id = new_segments.len();
            new_segments.push(Segment {
                id,
                start: w.start,
                end: w.end,
                text: w.text.clone(),
                speaker: sp.map(|l| format!("Hablante {}", l + 1)),
                speakers: Vec::new(),
                overlap: false,
                words: vec![w],
            });
        }
    }
    transcript.segments = new_segments;
}

/// Marca habla solapada: para cada `Segment` que cae dentro de una región de
/// solape de pyannote (por encima de umbral) y donde ≥2 hablantes de `turns`
/// coinciden, pone `overlap=true` y rellena `speakers` (dominante —`speaker`— en
/// `[0]`, resto por solape descendente). Conserva el invariante del schema
/// `overlap == speakers.len() > 1`. Conservador: sin solape acústico de pyannote
/// o con <2 hablantes identificados, no marca nada.
pub fn mark_overlaps(transcript: &mut Transcript, turns: &[SpeakerTurn], overlaps: &[(f32, f32)]) {
    if overlaps.is_empty() || turns.is_empty() {
        return;
    }
    for seg in &mut transcript.segments {
        let dur = (seg.end - seg.start).max(1e-6);
        // ¿intersecta alguna región de solape por encima de umbral (max 0.1s, 10%)?
        let ov = overlaps
            .iter()
            .map(|&(a, b)| (seg.end.min(b) - seg.start.max(a)).max(0.0))
            .fold(0.0f32, f32::max);
        if ov < 0.1f32.max(0.10 * dur) {
            continue;
        }
        // Hablantes (de turns) que solapan el segmento, acumulando solape.
        let mut acc: std::collections::HashMap<usize, f32> = std::collections::HashMap::new();
        for t in turns {
            let o = (seg.end.min(t.end_s) - seg.start.max(t.start_s)).max(0.0);
            if o > 0.0 {
                *acc.entry(t.speaker).or_default() += o;
            }
        }
        if acc.len() < 2 {
            continue; // sin ≥2 hablantes identificados, no hay solape representable
        }
        let mut spks: Vec<(usize, f32)> = acc.into_iter().collect();
        spks.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        let mut names: Vec<String> = spks
            .iter()
            .map(|&(sp, _)| format!("Hablante {}", sp + 1))
            .collect();
        // El dominante (`Segment.speaker`, fijado por segment_by_speaker) va en [0].
        if let Some(dom) = seg.speaker.clone()
            && let Some(pos) = names.iter().position(|n| *n == dom)
        {
            names.swap(0, pos);
        }
        seg.speakers = names;
        seg.overlap = true;
    }
}

/// Tope (s) de distancia al turno más cercano cuando no hay solape. Una palabra
/// que cae en un hueco mayor que esto (silencio largo o timestamp desviado) NO se
/// atribuye a ningún hablante (devuelve `None`) en vez de pegarse a un turno
/// lejano y fundir/partir turnos. Derivado de `coalesce_gap_ms` por defecto (500 ms).
const NEAREST_TURN_MAX_GAP_S: f32 = 0.5;

/// Hablante del turno con mayor solape temporal con `[start, end]`. Si no hay
/// solape con ningún turno, cae al más cercano siempre que su distancia no exceda
/// `max_gap_s`; pasado el tope devuelve `None` (palabra sin hablante asignable).
fn best_overlap(start: f32, end: f32, turns: &[SpeakerTurn], max_gap_s: f32) -> Option<usize> {
    let mut best: Option<(usize, f32)> = None;
    for t in turns {
        let ov = (end.min(t.end_s) - start.max(t.start_s)).max(0.0);
        if ov > 0.0 && best.is_none_or(|(_, b)| ov > b) {
            best = Some((t.speaker, ov));
        }
    }
    if let Some((sp, _)) = best {
        return Some(sp);
    }
    // Sin solape (p. ej. una palabra en el hueco entre turnos): el más cercano,
    // pero solo si está dentro del tope de distancia.
    let mid = (start + end) / 2.0;
    turns
        .iter()
        .map(|t| (t.speaker, turn_distance(mid, t)))
        .filter(|&(_, d)| d <= max_gap_s)
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(sp, _)| sp)
}

fn turn_distance(t: f32, turn: &SpeakerTurn) -> f32 {
    if t < turn.start_s {
        turn.start_s - t
    } else if t > turn.end_s {
        t - turn.end_s
    } else {
        0.0
    }
}

/// fbank Kaldi de 80 bins (25 ms / 10 ms, sin dither) con normalización por
/// media global (lo que espera CAM++ de 3D-Speaker).
fn fbank(samples: &[f32]) -> Result<Array2<f32>> {
    let mut opts = FbankOptions::default();
    opts.frame_opts.samp_freq = SR;
    opts.frame_opts.dither = 0.0;
    opts.mel_opts.num_bins = MEL_BINS;
    opts.use_energy = false; // CAM++ usa 80 bins log-mel, sin la columna de energía
    let computer = FbankComputer::new(opts).map_err(de)?;
    let mut online = OnlineFeature::new(FeatureComputer::Fbank(computer));
    online.accept_waveform(SR, samples);
    online.input_finished();

    let n = online.num_frames_ready();
    let mut feats = Array2::<f32>::zeros((n, MEL_BINS));
    for i in 0..n {
        if let Some(frame) = online.get_frame(i) {
            for (j, &v) in frame.iter().enumerate().take(MEL_BINS) {
                feats[[i, j]] = v;
            }
        }
    }
    if n > 0 {
        let mean = feats.mean_axis(Axis(0)).unwrap();
        feats -= &mean;
    }
    Ok(feats)
}

/// Clustering aglomerativo (enlace medio) sobre distancia coseno. Corta por
/// `max_speakers` si se da, o por `threshold`.
fn cluster(embs: &[Vec<f32>], max_speakers: Option<usize>, threshold: f32) -> Vec<usize> {
    let n = embs.len();
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![0];
    }
    let mut condensed = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            condensed.push(cosine_distance(&embs[i], &embs[j]) as f64);
        }
    }
    let dend = linkage(&mut condensed, n, Method::Average);
    let target = max_speakers.map(|k| k.clamp(1, n));
    cut_labels(&dend, n, target, threshold as f64)
}

/// Etiqueta las hojas recorriendo el dendrograma. Realiza un número de fusiones
/// determinado por `target_clusters` (corte fijo) o, si es `None`, por el **codo**
/// del dendrograma ([`auto_merge_count`]).
fn cut_labels(
    dend: &kodama::Dendrogram<f64>,
    n: usize,
    target_clusters: Option<usize>,
    threshold: f64,
) -> Vec<usize> {
    let total = 2 * n - 1;
    let mut parent: Vec<usize> = (0..total).collect();
    let steps = dend.steps();

    let merges = match target_clusters {
        Some(k) => n.saturating_sub(k.clamp(1, n)), // dejar exactamente k clusters
        None => auto_merge_count(steps, threshold),
    };
    for (s, step) in steps.iter().enumerate().take(merges) {
        let new_node = n + s;
        let a = find(&mut parent, step.cluster1);
        let b = find(&mut parent, step.cluster2);
        parent[a] = new_node;
        parent[b] = new_node;
    }

    let mut remap = std::collections::HashMap::<usize, usize>::new();
    let mut labels = vec![0usize; n];
    for (i, lab) in labels.iter_mut().enumerate() {
        let root = find(&mut parent, i);
        let next = remap.len();
        *lab = *remap.entry(root).or_insert(next);
    }
    labels
}

/// Estima cuántas fusiones realizar (auto nº de hablantes) por el **codo** del
/// dendrograma: el mayor salto de disimilitud entre fusiones consecutivas, por
/// encima de un suelo, marca la frontera natural entre hablantes; se corta antes
/// de ese salto. Si no hay salto claro (un solo hablante), se fusiona todo lo que
/// quede por debajo del umbral de seguridad.
fn auto_merge_count(steps: &[kodama::Step<f64>], threshold: f64) -> usize {
    /// Por debajo de este coseno las fusiones son intra-hablante (no se cortan).
    const FLOOR: f64 = 0.30;
    let mut best_gap = 0.0;
    let mut elbow: Option<usize> = None;
    for i in 0..steps.len().saturating_sub(1) {
        let hi = steps[i + 1].dissimilarity;
        let gap = hi - steps[i].dissimilarity;
        if hi >= FLOOR && gap > best_gap {
            best_gap = gap;
            elbow = Some(i + 1); // fusionar 0..=i, parar antes de i+1
        }
    }
    elbow.unwrap_or_else(|| {
        steps
            .iter()
            .take_while(|s| s.dissimilarity <= threshold)
            .count()
    })
}

fn find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-8 || nb < 1e-8 {
        return 1.0;
    }
    1.0 - dot / (na * nb)
}

fn de<E: std::fmt::Display>(e: E) -> RosettaError {
    RosettaError::Diarize(e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mkseg(start: f32, end: f32, speaker: &str) -> Segment {
        Segment {
            id: 0,
            start,
            end,
            text: "x".into(),
            speaker: Some(speaker.into()),
            speakers: Vec::new(),
            overlap: false,
            words: Vec::new(),
        }
    }

    #[test]
    fn mark_overlaps_marca_y_pone_dominante_primero() {
        let mut t = Transcript {
            segments: vec![mkseg(0.0, 2.0, "Hablante 2")],
            ..Default::default()
        };
        let turns = vec![
            SpeakerTurn {
                start_s: 0.0,
                end_s: 2.0,
                speaker: 0,
            },
            SpeakerTurn {
                start_s: 0.5,
                end_s: 1.5,
                speaker: 1,
            },
        ];
        mark_overlaps(&mut t, &turns, &[(0.5, 1.5)]);
        let s = &t.segments[0];
        assert!(s.overlap);
        assert_eq!(s.speakers.len(), 2);
        assert_eq!(s.speakers[0], "Hablante 2"); // dominante (= seg.speaker) en [0]
        assert!(s.speakers.contains(&"Hablante 1".to_string()));
    }

    #[test]
    fn mark_overlaps_no_marca_sin_region_de_solape() {
        let mut t = Transcript {
            segments: vec![mkseg(0.0, 2.0, "Hablante 1")],
            ..Default::default()
        };
        let turns = vec![
            SpeakerTurn {
                start_s: 0.0,
                end_s: 1.0,
                speaker: 0,
            },
            SpeakerTurn {
                start_s: 1.0,
                end_s: 2.0,
                speaker: 1,
            },
        ];
        mark_overlaps(&mut t, &turns, &[]); // pyannote no detectó solape acústico
        assert!(!t.segments[0].overlap);
        assert!(t.segments[0].speakers.is_empty());
    }

    #[test]
    fn mark_overlaps_no_marca_con_un_solo_hablante() {
        // Pyannote marca solape pero CAM++ solo ve 1 hablante → no representable.
        let mut t = Transcript {
            segments: vec![mkseg(0.0, 2.0, "Hablante 1")],
            ..Default::default()
        };
        let turns = vec![SpeakerTurn {
            start_s: 0.0,
            end_s: 2.0,
            speaker: 0,
        }];
        mark_overlaps(&mut t, &turns, &[(0.5, 1.5)]);
        assert!(!t.segments[0].overlap);
    }

    #[test]
    fn cluster_separa_dos_grupos() {
        let embs = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.98, 0.05, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![0.02, 0.0, 0.99],
        ];
        let labels = cluster(&embs, None, 0.5);
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[2], labels[3]);
        assert_ne!(labels[0], labels[2]);
    }

    #[test]
    fn cluster_respeta_max_speakers() {
        let embs = vec![
            vec![1.0, 0.0],
            vec![0.9, 0.1],
            vec![0.0, 1.0],
            vec![0.1, 0.9],
        ];
        let labels = cluster(&embs, Some(1), 0.5);
        assert!(labels.iter().all(|&l| l == 0));
    }

    #[test]
    fn assign_por_solape() {
        let turns = vec![
            SpeakerTurn {
                start_s: 0.0,
                end_s: 5.0,
                speaker: 0,
            },
            SpeakerTurn {
                start_s: 5.0,
                end_s: 10.0,
                speaker: 1,
            },
        ];
        assert_eq!(
            best_overlap(1.0, 2.0, &turns, NEAREST_TURN_MAX_GAP_S),
            Some(0)
        );
        assert_eq!(
            best_overlap(6.0, 9.0, &turns, NEAREST_TURN_MAX_GAP_S),
            Some(1)
        );
    }

    #[test]
    fn best_overlap_topa_distancia_en_huecos_grandes() {
        // Dos turnos separados por un hueco grande [0,5] y [20,25].
        let turns = vec![
            SpeakerTurn {
                start_s: 0.0,
                end_s: 5.0,
                speaker: 0,
            },
            SpeakerTurn {
                start_s: 20.0,
                end_s: 25.0,
                speaker: 1,
            },
        ];
        // Palabra en mitad del hueco (mid=12.5): a >7 s de ambos turnos → None.
        assert_eq!(
            best_overlap(12.0, 13.0, &turns, NEAREST_TURN_MAX_GAP_S),
            None
        );
        // Palabra cercana al primer turno (mid=5.3, a 0.3 s < 0.5) → Some(0).
        assert_eq!(
            best_overlap(5.1, 5.5, &turns, NEAREST_TURN_MAX_GAP_S),
            Some(0)
        );
        // Justo pasado el tope (mid=5.6, a 0.6 s > 0.5) → None.
        assert_eq!(best_overlap(5.4, 5.8, &turns, NEAREST_TURN_MAX_GAP_S), None);
    }

    #[test]
    fn auto_codo_separa_aunque_el_umbral_fusionaria() {
        // Cross-distancia ~0.4 (< umbral 0.9): por umbral fijo se fusionaría todo
        // en 1 cluster, pero el codo detecta el salto y deja 2 hablantes.
        let embs = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.99, 0.1, 0.0],
            vec![0.6, 0.8, 0.0],
            vec![0.62, 0.78, 0.0],
        ];
        let labels = cluster(&embs, None, 0.9);
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[2], labels[3]);
        assert_ne!(labels[0], labels[2]);
    }

    #[test]
    fn auto_un_solo_hablante() {
        let embs = vec![vec![1.0, 0.0], vec![0.99, 0.05], vec![0.98, 0.03]];
        let labels = cluster(&embs, None, 0.55);
        assert!(labels.iter().all(|&l| l == labels[0]), "un solo hablante");
    }

    #[test]
    fn coalesce_fusiona_mismo_hablante() {
        let mut turns = vec![
            SpeakerTurn {
                start_s: 0.0,
                end_s: 2.0,
                speaker: 0,
            },
            SpeakerTurn {
                start_s: 2.3,
                end_s: 4.0,
                speaker: 0,
            }, // hueco 0.3 < 0.5
            SpeakerTurn {
                start_s: 4.1,
                end_s: 6.0,
                speaker: 1,
            }, // otro hablante
        ];
        coalesce_turns(&mut turns, 0.5);
        assert_eq!(turns.len(), 2);
        assert_eq!(turns[0].speaker, 0);
        assert!((turns[0].end_s - 4.0).abs() < 1e-6);
        assert_eq!(turns[1].speaker, 1);
    }
}
