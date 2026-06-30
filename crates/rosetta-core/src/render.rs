//! Renderizado de un [`Transcript`] a los formatos de salida soportados.

use crate::{OutputFormat, Segment, Transcript};

/// Renderiza el transcript al formato dado. `timestamps` añade marcas de tiempo
/// en los formatos que lo permiten (md).
pub fn render(transcript: &Transcript, format: OutputFormat, timestamps: bool) -> String {
    match format {
        OutputFormat::Txt => render_txt(transcript),
        OutputFormat::Md => render_md(transcript, timestamps),
        OutputFormat::Json => render_json(transcript),
        OutputFormat::Srt => render_srt(transcript),
        OutputFormat::Vtt => render_vtt(transcript),
    }
}

/// Extensión de archivo recomendada para el formato.
pub fn extension(format: OutputFormat) -> &'static str {
    match format {
        OutputFormat::Md => "md",
        OutputFormat::Json => "json",
        OutputFormat::Txt => "txt",
        OutputFormat::Srt => "srt",
        OutputFormat::Vtt => "vtt",
    }
}

/// Prefijo de hablante para un segmento (vacío si no hay hablante).
fn speaker_prefix(seg: &Segment, f: impl FnOnce(&str) -> String) -> String {
    seg.speaker.as_deref().map(f).unwrap_or_default()
}

/// Co-hablantes (todos menos el dominante en `[0]`) cuando el segmento marca solape.
/// `None` si no hay solape → la salida queda idéntica al caso de un solo hablante.
fn co_speakers(seg: &Segment) -> Option<String> {
    if seg.overlap && seg.speakers.len() > 1 {
        Some(seg.speakers[1..].join(", "))
    } else {
        None
    }
}

fn render_txt(t: &Transcript) -> String {
    if t.segments.iter().any(|s| s.speaker.is_some()) {
        t.segments
            .iter()
            .map(|s| match &s.speaker {
                Some(spk) => format!("{spk}: {}", s.text.trim()),
                None => s.text.trim().to_string(),
            })
            .collect::<Vec<_>>()
            .join("\n")
    } else {
        t.text.trim().to_string()
    }
}

fn render_md(t: &Transcript, timestamps: bool) -> String {
    let mut out = String::new();
    let title = if t.source.file.is_empty() {
        "Transcripción"
    } else {
        t.source.file.as_str()
    };
    out.push_str(&format!("# Transcripción — {title}\n\n"));
    out.push_str(&format!(
        "- **Modelo:** {} ({})\n",
        t.model.name, t.model.device
    ));
    if !t.source.language.is_empty() {
        out.push_str(&format!("- **Idioma:** {}\n", t.source.language));
    }
    out.push_str(&format!(
        "- **Duración:** {}\n\n",
        fmt_hms(t.source.duration_s)
    ));

    let has_speakers = t.segments.iter().any(|s| s.speaker.is_some());
    if timestamps || has_speakers {
        for s in &t.segments {
            let ts = if timestamps {
                format!("`[{} → {}]` ", fmt_hms(s.start), fmt_hms(s.end))
            } else {
                String::new()
            };
            let spk = match (&s.speaker, co_speakers(s)) {
                (Some(dom), Some(co)) => format!("**{dom} (+{co}):** "),
                (Some(dom), None) => format!("**{dom}:** "),
                (None, _) => String::new(),
            };
            out.push_str(&format!("{ts}{spk}{}\n\n", s.text.trim()));
        }
    } else {
        out.push_str(t.text.trim());
        out.push('\n');
    }
    out
}

fn render_json(t: &Transcript) -> String {
    serde_json::to_string_pretty(t).unwrap_or_else(|e| {
        // Escapar el mensaje para no emitir JSON sintácticamente inválido.
        let msg = serde_json::to_string(&e.to_string())
            .unwrap_or_else(|_| "\"error de serialización\"".to_string());
        format!("{{\"error\":{msg}}}")
    })
}

fn render_srt(t: &Transcript) -> String {
    let mut out = String::new();
    // Numeración secuencial 1..N (posicional, independiente de Segment.id, como
    // exige SRT) saltando los cues sin texto.
    let mut idx = 1;
    for s in &t.segments {
        let text = clean_cue_text(&s.text);
        if text.is_empty() {
            continue;
        }
        let spk = match (&s.speaker, co_speakers(s)) {
            (Some(dom), Some(co)) => format!("{dom} [+{co}]: "),
            (Some(dom), None) => format!("{dom}: "),
            (None, _) => String::new(),
        };
        out.push_str(&format!(
            "{idx}\n{} --> {}\n{spk}{text}\n\n",
            fmt_ts(s.start, ','),
            fmt_ts(s.end, ','),
        ));
        idx += 1;
    }
    out
}

fn render_vtt(t: &Transcript) -> String {
    let mut out = String::from("WEBVTT\n\n");
    for s in &t.segments {
        let text = clean_cue_text(&s.text);
        if text.is_empty() {
            continue;
        }
        let spk = speaker_prefix(s, |x| format!("<v {x}>"));
        out.push_str(&format!(
            "{} --> {}\n{spk}{text}\n\n",
            fmt_ts(s.start, '.'),
            fmt_ts(s.end, '.'),
        ));
    }
    out
}

/// Limpia el texto de un cue SRT/VTT: colapsa cualquier secuencia de espacios o
/// saltos de línea internos a un único espacio (un `\n\n` interno rompería la
/// separación de cues) y recorta los extremos.
fn clean_cue_text(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Formato de marca de tiempo `HH:MM:SS<sep>mmm` (sep `,` para SRT, `.` para VTT).
fn fmt_ts(secs: f32, sep: char) -> String {
    let secs = if secs.is_finite() { secs.max(0.0) } else { 0.0 };
    let total_ms = (secs * 1000.0).round() as u64;
    let ms = total_ms % 1000;
    let s = (total_ms / 1000) % 60;
    let m = (total_ms / 60_000) % 60;
    let h = total_ms / 3_600_000;
    format!("{h:02}:{m:02}:{s:02}{sep}{ms:03}")
}

/// Duración legible (`1h02m03s` o `2m03s`).
fn fmt_hms(secs: f32) -> String {
    let secs = if secs.is_finite() { secs.max(0.0) } else { 0.0 };
    let total = secs.round() as u64;
    let s = total % 60;
    let m = (total / 60) % 60;
    let h = total / 3600;
    if h > 0 {
        format!("{h}h{m:02}m{s:02}s")
    } else {
        format!("{m}m{s:02}s")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ModelInfo, Segment, SourceInfo, Transcript};

    fn sample() -> Transcript {
        Transcript {
            version: "1.0".into(),
            source: SourceInfo {
                file: "a.wav".into(),
                duration_s: 4.2,
                language: "es".into(),
            },
            model: ModelInfo {
                name: "parakeet".into(),
                device: "cpu".into(),
            },
            segments: vec![
                Segment {
                    id: 0,
                    start: 0.0,
                    end: 2.0,
                    text: "hola".into(),
                    speaker: None,
                    speakers: vec![],
                    overlap: false,
                    words: vec![],
                },
                Segment {
                    id: 1,
                    start: 2.0,
                    end: 4.2,
                    text: "mundo".into(),
                    speaker: None,
                    speakers: vec![],
                    overlap: false,
                    words: vec![],
                },
            ],
            text: "hola mundo".into(),
        }
    }

    #[test]
    fn outputs_txt() {
        assert_eq!(render(&sample(), OutputFormat::Txt, false), "hola mundo");
    }

    #[test]
    fn outputs_srt() {
        let s = render(&sample(), OutputFormat::Srt, false);
        assert!(s.contains("1\n00:00:00,000 --> 00:00:02,000"), "{s}");
        assert!(s.contains("2\n00:00:02,000 --> 00:00:04,200"), "{s}");
    }

    #[test]
    fn outputs_vtt() {
        let s = render(&sample(), OutputFormat::Vtt, false);
        assert!(s.starts_with("WEBVTT"), "{s}");
        assert!(s.contains("00:00:02.000 --> 00:00:04.200"), "{s}");
    }

    #[test]
    fn outputs_json() {
        let s = render(&sample(), OutputFormat::Json, false);
        assert!(s.contains("\"text\""));
        assert!(s.contains("hola mundo"));
    }

    #[test]
    fn outputs_md() {
        let s = render(&sample(), OutputFormat::Md, true);
        assert!(s.contains("# Transcripción"));
        assert!(s.contains("hola"));
        assert!(s.contains("00:00:00") || s.contains("0m00s"));
    }

    fn sample_overlap() -> Transcript {
        Transcript {
            version: "1.0".into(),
            source: SourceInfo {
                file: "a.wav".into(),
                duration_s: 2.0,
                language: "es".into(),
            },
            model: ModelInfo {
                name: "parakeet".into(),
                device: "cpu".into(),
            },
            segments: vec![Segment {
                id: 0,
                start: 0.0,
                end: 2.0,
                text: "hola".into(),
                speaker: Some("Hablante 1".into()),
                speakers: vec!["Hablante 1".into(), "Hablante 2".into()],
                overlap: true,
                words: vec![],
            }],
            text: "hola".into(),
        }
    }

    #[test]
    fn outputs_overlap_aditivo() {
        let t = sample_overlap();
        let md = render(&t, OutputFormat::Md, false);
        assert!(md.contains("**Hablante 1 (+Hablante 2):**"), "{md}");
        let srt = render(&t, OutputFormat::Srt, false);
        assert!(srt.contains("Hablante 1 [+Hablante 2]:"), "{srt}");
        // VTT mantiene un único <v> con el dominante.
        let vtt = render(&t, OutputFormat::Vtt, false);
        assert!(vtt.contains("<v Hablante 1>"), "{vtt}");
        // JSON enriquecido: emite overlap y speakers cuando hay solape.
        let json = render(&t, OutputFormat::Json, false);
        assert!(json.contains("\"overlap\": true"), "{json}");
        assert!(json.contains("\"speakers\""), "{json}");
    }

    #[test]
    fn json_sin_solape_no_emite_campos_nuevos() {
        // Garantía no-breaking: sin solape, el JSON no incluye speakers/overlap.
        let json = render(&sample(), OutputFormat::Json, false);
        assert!(!json.contains("overlap"), "{json}");
        assert!(!json.contains("speakers"), "{json}");
    }
}
