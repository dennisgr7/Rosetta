//! Tipos centrales y errores compartidos de Rosetta.

use serde::{Deserialize, Serialize};

/// Una palabra con marca temporal (segundos).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Word {
    pub text: String,
    pub start: f32,
    pub end: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
}

/// Un segmento de transcripción (una frase o turno de habla).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Segment {
    pub id: usize,
    pub start: f32,
    pub end: f32,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speaker: Option<String>,
    /// Todos los hablantes presentes en el tramo, ordenados por solape descendente
    /// con el dominante (`speaker`) en `[0]`. Se emite solo si hay solape real
    /// (`len > 1`). Invariante: `overlap == (speakers.len() > 1)`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub speakers: Vec<String>,
    /// Marca de habla solapada (varios hablantes simultáneos en el tramo).
    #[serde(default, skip_serializing_if = "is_false")]
    pub overlap: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub words: Vec<Word>,
}

/// `skip_serializing_if` para `bool` (serde no acepta una función de la std directa).
/// Mantiene el JSON byte-idéntico cuando no hay solape (no emite `"overlap": false`).
fn is_false(b: &bool) -> bool {
    !*b
}

/// Metadatos de la fuente transcrita.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SourceInfo {
    pub file: String,
    pub duration_s: f32,
    pub language: String,
}

/// Metadatos del modelo / inferencia usados.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelInfo {
    pub name: String,
    pub device: String,
}

/// Resultado completo de una transcripción (base del JSON de salida).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Transcript {
    pub version: String,
    pub source: SourceInfo,
    pub model: ModelInfo,
    pub segments: Vec<Segment>,
    pub text: String,
}

/// Contexto para condicionar la decodificación del ASR (E4). Solo Whisper lo usa
/// (inyecta los textos como tokens de prompt); Parakeet lo ignora (método por
/// defecto del trait `Engine`).
#[derive(Debug, Clone, Default)]
pub struct DecodeCtx {
    /// Texto inicial que sesga el vocabulario (términos propios, formato, estilo).
    pub init_prompt: String,
    /// Texto ya transcrito de bloques anteriores (coherencia en audio largo).
    pub prev_text: String,
    /// Código ISO del idioma a forzar; `"auto"` (o cadena vacía vía `Default`) =
    /// autodetección. Solo Whisper lo honra: salta la detección por argmax e
    /// inyecta el token de idioma correspondiente en el prefill. Parakeet lo ignora.
    pub language: String,
}

/// Métricas de una corrida de transcripción, para comparar el comportamiento del
/// hardware entre máquinas. Esquema FIJO (todos los campos siempre presentes) →
/// una línea JSONL por corrida que se puede diferenciar directamente entre hosts
/// (p. ej. Snapdragon/DirectML vs Raspberry Pi/CPU). Tiempos en milisegundos.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RunMetrics {
    pub host: String,
    pub os: String,
    pub arch: String,
    /// `--device` solicitado (auto/npu/gpu/cpu).
    pub device_arg: String,
    /// EP primario planeado de la cascada (etiqueta).
    pub ep_primary: String,
    pub model: String,
    /// Duración del audio de entrada (s) y factor de tiempo real (t_total/audio).
    pub audio_s: f64,
    pub rtf: f64,
    pub t_total_ms: f64,
    /// Decodificación del audio a PCM 16k.
    pub t_load_ms: f64,
    /// Carga del motor ASR (sesiones ONNX); coste de arranque en frío.
    pub t_model_load_ms: f64,
    pub t_denoise_ms: f64,
    /// Inferencia ASR (sin contar la carga del modelo).
    pub t_transcribe_ms: f64,
    pub t_diarize_ms: f64,
    pub t_render_ms: f64,
    /// RSS del proceso al final (MB); 0 si no se pudo medir.
    pub rss_mb: f64,
    pub n_segments: usize,
}

impl RunMetrics {
    /// Serializa a una sola línea JSON (para hacer append a un log JSONL).
    pub fn to_jsonl(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| "{}".to_string())
    }
}

/// Formatos de salida soportados.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    Md,
    Json,
    Txt,
    Srt,
    Vtt,
}

/// Errores de Rosetta.
#[derive(Debug, thiserror::Error)]
pub enum RosettaError {
    #[error("E/S: {0}")]
    Io(#[from] std::io::Error),
    #[error("decodificación de audio: {0}")]
    Audio(String),
    #[error("inferencia/ASR: {0}")]
    Asr(String),
    #[error("diarización: {0}")]
    Diarize(String),
    #[error("aceleración/dispositivo: {0}")]
    Accel(String),
    #[error("modelo: {0}")]
    Model(String),
    #[error("{0}")]
    Other(String),
}

/// Alias de resultado del proyecto.
pub type Result<T> = std::result::Result<T, RosettaError>;

mod render;
pub use render::{extension, render};
