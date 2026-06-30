//! Decodificación de audio/vídeo a PCM con symphonia 0.6 (primario) y
//! ffmpeg-sidecar (fallback).

use std::path::Path;

use rosetta_core::{Result, RosettaError};
use symphonia::core::codecs::CodecParameters;
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, TrackType};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;

use crate::AudioPcm;
use crate::resample::resample_mono;

/// Cota dura de muestras PCM mono acumuladas durante el decode, como red de
/// seguridad ante un archivo gigante o una cabecera engañosa que, sin límite,
/// haría crecer el `Vec` hasta agotar la RAM (OOM). Cálculo: 16 000 Hz × 60 s ×
/// 60 min × 8 h = 8 horas de audio a 16 kHz mono ≈ 460,8 M muestras (~1,8 GiB en
/// f32). Es deliberadamente GENEROSA: el audio normal nunca se acerca, así que el
/// comportamiento no cambia salvo para entradas patológicas, que se abortan con
/// error en vez de seguir acumulando. Nota: en `decode_symphonia` la cota se
/// aplica al PCM acumulado a la SR de ORIGEN (antes del resample a 16 kHz), por lo
/// que con fuentes de mayor SR la cota equivale a menos horas (p. ej. ~2,7 h a
/// 48 kHz); sigue siendo una red anti-OOM holgada en todos los casos.
const MAX_PCM_SAMPLES: usize = 16_000 * 60 * 60 * 8;

/// Mensaje de error compartido al superar `MAX_PCM_SAMPLES`.
fn pcm_cap_error() -> RosettaError {
    RosettaError::Audio(format!(
        "el audio supera la cota de {MAX_PCM_SAMPLES} muestras (~8 h a 16 kHz); \
         entrada demasiado grande o cabecera corrupta"
    ))
}

/// Promedia los canales de un buffer intercalado a una sola muestra mono por
/// frame, escribiendo en `out`. Función pura extraída para poder testearla: para
/// estéreo L/R devuelve la media `(L+R)/2`; para mono pasa la muestra tal cual.
/// `channels` se asume ≥1 (el llamante lo garantiza con `.max(1)`).
fn downmix_to_mono(interleaved: &[f32], channels: usize, out: &mut Vec<f32>) {
    for frame in interleaved.chunks(channels) {
        let sum: f32 = frame.iter().copied().sum();
        out.push(sum / channels as f32);
    }
}

/// Decodifica `path` a PCM mono a `target_sr` Hz. Symphonia primero; si falla por
/// formato/códec no soportado, cae a ffmpeg-sidecar.
pub fn decode_file(path: &Path, target_sr: u32) -> Result<AudioPcm> {
    match decode_symphonia(path, target_sr) {
        Ok(pcm) => Ok(pcm),
        Err(e) => {
            tracing::warn!("symphonia no pudo decodificar ({e}); intentando ffmpeg-sidecar");
            decode_ffmpeg(path, target_sr)
        }
    }
}

fn decode_symphonia(path: &Path, target_sr: u32) -> Result<AudioPcm> {
    let file = std::fs::File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let mut reader = symphonia::default::get_probe()
        .probe(
            &hint,
            mss,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .map_err(|e| RosettaError::Audio(format!("probe: {e}")))?;

    // Selecciona la pista de audio por defecto y crea su decodificador.
    let track = reader
        .default_track(TrackType::Audio)
        .ok_or_else(|| RosettaError::Audio("sin pista de audio".into()))?;
    let track_id = track.id;
    let audio_params = match track.codec_params.as_ref() {
        Some(CodecParameters::Audio(a)) => a,
        _ => return Err(RosettaError::Audio("la pista no es de audio".into())),
    };
    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(audio_params, &AudioDecoderOptions::default())
        .map_err(|e| RosettaError::Audio(format!("codec: {e}")))?;

    let mut mono: Vec<f32> = Vec::new();
    let mut interleaved: Vec<f32> = Vec::new();
    let mut sr_in: Option<u32> = None;

    while let Some(packet) = reader
        .next_packet()
        .map_err(|e| RosettaError::Audio(format!("packet: {e}")))?
    {
        if packet.track_id != track_id {
            continue;
        }
        match decoder.decode(&packet) {
            Ok(decoded) => {
                let spec = decoded.spec();
                sr_in = Some(spec.rate());
                let channels = spec.channels().count().max(1);
                decoded.copy_to_vec_interleaved(&mut interleaved);
                // Downmix a mono promediando los canales.
                downmix_to_mono(&interleaved, channels, &mut mono);
                // Cota de recursos: aborta si la acumulación se desboca (OOM).
                if mono.len() > MAX_PCM_SAMPLES {
                    return Err(pcm_cap_error());
                }
            }
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(e) => return Err(RosettaError::Audio(format!("decode: {e}"))),
        }
    }

    if mono.is_empty() {
        return Err(RosettaError::Audio("no se decodificaron muestras".into()));
    }

    let sr_in = sr_in.unwrap_or(target_sr);
    let samples = resample_mono(&mono, sr_in, target_sr)?;
    Ok(AudioPcm {
        samples,
        sample_rate: target_sr,
    })
}

/// Resuelve el binario de ffmpeg a una ruta ABSOLUTA de confianza. Prioriza la
/// variable `ROSETTA_FFMPEG`; si no, busca `ffmpeg` en las entradas ABSOLUTAS de
/// PATH (las relativas, incluido el cwd, se ignoran para evitar binary-planting
/// en Windows, donde `CreateProcess` busca en el directorio de trabajo). Devuelve
/// `None` si no se encuentra.
fn resolve_ffmpeg() -> Option<std::path::PathBuf> {
    if let Some(p) = std::env::var_os("ROSETTA_FFMPEG") {
        let p = std::path::PathBuf::from(p);
        if p.is_file() {
            return Some(p);
        }
    }
    let exe = if cfg!(windows) {
        "ffmpeg.exe"
    } else {
        "ffmpeg"
    };
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        // Ignorar entradas vacías o relativas (incluido "."): el loader las
        // resolvería contra el cwd, permitiendo un ffmpeg plantado.
        if dir.as_os_str().is_empty() || dir.is_relative() {
            continue;
        }
        let cand = dir.join(exe);
        if cand.is_file() {
            return Some(cand);
        }
    }
    None
}

fn decode_ffmpeg(path: &Path, target_sr: u32) -> Result<AudioPcm> {
    use ffmpeg_sidecar::command::FfmpegCommand;
    use std::io::Read;

    // Resuelve `ffmpeg` a una ruta ABSOLUTA de confianza (ROSETTA_FFMPEG o PATH,
    // excluyendo el cwd) para evitar el binary-planting de Windows (CreateProcess
    // busca el ejecutable en el directorio de trabajo). La auto-descarga de
    // ffmpeg-sidecar se omite a propósito: arrastra ureq→rustls→ring (compila C,
    // problemático en Windows ARM64 / CI). symphonia cubre los formatos comunes
    // sin ffmpeg; este fallback sólo actúa para códecs raros y, si falta ffmpeg,
    // falla con un mensaje claro.
    let path_str = path
        .to_str()
        .ok_or_else(|| RosettaError::Audio("ruta no UTF-8".into()))?;

    let mut cmd = match resolve_ffmpeg() {
        Some(p) => FfmpegCommand::new_with_path(p),
        None => {
            return Err(RosettaError::Audio(
                "ffmpeg no encontrado en PATH (instala ffmpeg o define ROSETTA_FFMPEG con su ruta absoluta)"
                    .into(),
            ));
        }
    };
    let mut child = cmd
        .input(path_str)
        .args([
            "-vn",
            "-ac",
            "1",
            "-ar",
            &target_sr.to_string(),
            "-f",
            "f32le",
            "-nostats",
            "-loglevel",
            "error",
        ])
        .output("-")
        .spawn()
        .map_err(|e| RosettaError::Audio(format!("spawn ffmpeg: {e}")))?;

    let mut stdout = child
        .take_stdout()
        .ok_or_else(|| RosettaError::Audio("ffmpeg sin stdout".into()))?;
    // Cota de recursos: f32le ⇒ 4 bytes/muestra mono. Limitamos los bytes leídos
    // a `MAX_PCM_SAMPLES * 4` para no acumular sin freno ante un stream gigante o
    // engañoso (OOM). Leemos como mucho 1 byte de más para distinguir "justo en el
    // límite" de "se ha pasado". SIEMPRE esperamos al proceso (evita dejar ffmpeg
    // huérfano y detectar fallos de ffmpeg a mitad de stream).
    let byte_cap = MAX_PCM_SAMPLES.saturating_mul(4);
    let mut bytes = Vec::new();
    let read_res = (&mut stdout)
        .take(byte_cap as u64 + 1)
        .read_to_end(&mut bytes);
    let status = child.wait();
    read_res.map_err(|e| RosettaError::Audio(format!("lectura ffmpeg: {e}")))?;
    if bytes.len() as u64 > byte_cap as u64 {
        return Err(pcm_cap_error());
    }
    match status {
        Ok(s) if !s.success() => {
            return Err(RosettaError::Audio(format!("ffmpeg salió con error: {s}")));
        }
        Err(e) => return Err(RosettaError::Audio(format!("esperar ffmpeg: {e}"))),
        _ => {}
    }

    if bytes.is_empty() {
        return Err(RosettaError::Audio(
            "ffmpeg no produjo audio (formato no soportado?)".into(),
        ));
    }

    let samples: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    Ok(AudioPcm {
        samples,
        sample_rate: target_sr,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn downmix_estereo_promedia_canales() {
        // Intercalado L/R: (1.0,3.0) y (2.0,4.0) → medias 2.0 y 3.0.
        let interleaved = [1.0f32, 3.0, 2.0, 4.0];
        let mut out = Vec::new();
        downmix_to_mono(&interleaved, 2, &mut out);
        assert_eq!(out, vec![2.0f32, 3.0]);
    }

    #[test]
    fn downmix_mono_pasa_igual() {
        let interleaved = [0.5f32, -0.25, 1.0];
        let mut out = Vec::new();
        downmix_to_mono(&interleaved, 1, &mut out);
        assert_eq!(out, vec![0.5f32, -0.25, 1.0]);
    }

    #[test]
    fn downmix_cuatro_canales() {
        // Un frame de 4 canales: media de (1,2,3,4) = 2.5.
        let interleaved = [1.0f32, 2.0, 3.0, 4.0];
        let mut out = Vec::new();
        downmix_to_mono(&interleaved, 4, &mut out);
        assert_eq!(out, vec![2.5f32]);
    }

    #[test]
    fn cota_pcm_es_generosa() {
        // Sanidad: la cota equivale a 8 horas de audio mono a 16 kHz.
        assert_eq!(MAX_PCM_SAMPLES, 16_000 * 60 * 60 * 8);
    }
}
