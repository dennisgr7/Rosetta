//! Tokenizer de Whisper (byte-level BPE de GPT-2), Rust puro y SIN la crate
//! `tokenizers` (que arrastraría `esaxx-rs`/`onig` = C en Windows).
//!
//! Decode: lee `vocab.json` (token→id), lo invierte a id→token y reconstruye los
//! bytes con la tabla GPT-2 bytes↔unicode (un carácter multibyte puede partirse
//! entre dos tokens, por eso se acumulan bytes y se decodifica UTF-8 al final).
//!
//! Encode (E4, para `--init-prompt` / `prev_text`): pre-tokeniza con la regex de
//! GPT-2 (`fancy-regex`, Rust puro), codifica cada pieza a nivel de byte y aplica
//! BPE greedy con los rangos de `merges.txt`. Requiere `from_files` (vocab+merges).

use std::collections::HashMap;
use std::path::Path;
use std::sync::OnceLock;

use fancy_regex::Regex;

use rosetta_core::{Result, RosettaError};

/// Tokenizer de Whisper (decode siempre; encode si se cargaron las merges).
pub struct WhisperTokenizer {
    id_to_token: HashMap<i64, String>,
    token_to_id: HashMap<String, i64>,
    /// char unicode (espacio de bytes GPT-2) → byte original.
    byte_decoder: HashMap<char, u8>,
    /// byte original → char unicode (para codificar).
    byte_encoder: HashMap<u8, char>,
    /// `(izq, der) → rango` de las merges BPE; vacío si solo se cargó el vocab.
    merge_ranks: HashMap<(String, String), usize>,
}

impl WhisperTokenizer {
    /// Carga vocab + merges (habilita `encode`; necesario para E4).
    pub fn from_files(vocab: &Path, merges: &Path) -> Result<Self> {
        let mut t = Self::from_vocab(vocab)?;
        t.merge_ranks = load_merges(merges)?;
        Ok(t)
    }

    /// Carga solo el vocabulario (decode-only; `encode` sin merges devuelve tokens
    /// por byte, subóptimo pero no incorrecto al re-decodificar).
    pub fn from_vocab(path: &Path) -> Result<Self> {
        let data =
            std::fs::read(path).map_err(|e| RosettaError::Asr(format!("leer vocab.json: {e}")))?;
        let map: HashMap<String, i64> = serde_json::from_slice(&data)
            .map_err(|e| RosettaError::Asr(format!("parsear vocab.json: {e}")))?;
        let token_to_id = map.clone();
        let id_to_token = map.into_iter().map(|(k, v)| (v, k)).collect();
        let (byte_encoder, byte_decoder) = build_byte_tables();
        Ok(Self {
            id_to_token,
            token_to_id,
            byte_decoder,
            byte_encoder,
            merge_ranks: HashMap::new(),
        })
    }

    /// Decodifica una secuencia de ids a texto. Ignora cualquier id ausente del
    /// vocabulario base (especiales: eos, idioma, task, timestamps).
    pub fn decode(&self, ids: &[i64]) -> String {
        let mut bytes: Vec<u8> = Vec::new();
        for id in ids {
            if let Some(tok) = self.id_to_token.get(id) {
                for ch in tok.chars() {
                    if let Some(&b) = self.byte_decoder.get(&ch) {
                        bytes.push(b);
                    }
                }
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Codifica texto a ids (BPE byte-level GPT-2). Requiere haber cargado las
    /// merges (`from_files`); en su ausencia cada byte queda como un token.
    pub fn encode(&self, text: &str) -> Vec<i64> {
        let mut ids = Vec::new();
        for m in gpt2_regex().find_iter(text) {
            let piece = match m {
                Ok(m) => m.as_str(),
                Err(_) => continue,
            };
            let mut symbols: Vec<String> = piece
                .bytes()
                .filter_map(|b| self.byte_encoder.get(&b).map(|c| c.to_string()))
                .collect();
            bpe(&mut symbols, &self.merge_ranks);
            for s in symbols {
                if let Some(&id) = self.token_to_id.get(&s) {
                    ids.push(id);
                }
            }
        }
        ids
    }
}

/// Regex de pre-tokenización de GPT-2/Whisper (con lookahead → `fancy-regex`).
fn gpt2_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")
            .expect("regex de pre-tokenización GPT-2 válida")
    })
}

/// Lee `merges.txt` (`#version` + pares `izq der` por línea) a un mapa
/// `(izq, der) → rango` (rango = orden de aparición).
fn load_merges(path: &Path) -> Result<HashMap<(String, String), usize>> {
    let data = std::fs::read_to_string(path)
        .map_err(|e| RosettaError::Asr(format!("leer merges.txt: {e}")))?;
    let mut ranks = HashMap::new();
    let mut rank = 0usize;
    for line in data.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut it = line.split(' ');
        if let (Some(a), Some(b)) = (it.next(), it.next()) {
            ranks.insert((a.to_string(), b.to_string()), rank);
            rank += 1;
        }
    }
    Ok(ranks)
}

/// BPE greedy: fusiona repetidamente el par adyacente de menor rango hasta que no
/// quede ninguno fusionable. `O(n²·merges)`, suficiente para prompts cortos.
fn bpe(symbols: &mut Vec<String>, ranks: &HashMap<(String, String), usize>) {
    if ranks.is_empty() || symbols.len() < 2 {
        return;
    }
    loop {
        let mut best: Option<(usize, usize)> = None; // (índice, rango)
        for i in 0..symbols.len() - 1 {
            if let Some(&r) = ranks.get(&(symbols[i].clone(), symbols[i + 1].clone()))
                && best.map(|(_, br)| r < br).unwrap_or(true)
            {
                best = Some((i, r));
            }
        }
        let Some((i, _)) = best else { break };
        let merged = format!("{}{}", symbols[i], symbols[i + 1]);
        symbols[i] = merged;
        symbols.remove(i + 1);
        if symbols.len() < 2 {
            break;
        }
    }
}

/// Tabla GPT-2 byte→char unicode imprimible (256 entradas, biyectiva).
fn bytes_to_unicode() -> Vec<(u8, char)> {
    let mut bs: Vec<u32> = Vec::new();
    bs.extend(0x21..=0x7e); // '!'..='~'
    bs.extend(0xa1..=0xac); // '¡'..='¬'
    bs.extend(0xae..=0xff); // '®'..='ÿ'
    let mut cs: Vec<u32> = bs.clone();
    let mut n = 0u32;
    for b in 0u32..256 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    bs.into_iter()
        .zip(cs)
        .map(|(b, c)| (b as u8, char::from_u32(c).expect("codepoint válido")))
        .collect()
}

/// Tablas byte↔char (encoder y decoder) de la codificación de bytes de GPT-2.
fn build_byte_tables() -> (HashMap<u8, char>, HashMap<char, u8>) {
    let table = bytes_to_unicode();
    let encoder = table.iter().copied().collect();
    let decoder = table.into_iter().map(|(b, c)| (c, b)).collect();
    (encoder, decoder)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_table_es_biyectiva() {
        let table = bytes_to_unicode();
        assert_eq!(table.len(), 256);
        let (_enc, decoder) = build_byte_tables();
        for (b, c) in table {
            assert_eq!(decoder.get(&c), Some(&b));
        }
        assert_eq!(decoder.len(), 256);
    }

    #[test]
    fn bpe_fusiona_por_rango() {
        let ranks: HashMap<(String, String), usize> = [
            (("a".to_string(), "b".to_string()), 0),
            (("ab".to_string(), "c".to_string()), 1),
        ]
        .into_iter()
        .collect();
        let mut s = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        bpe(&mut s, &ranks);
        assert_eq!(s, vec!["abc".to_string()]);

        // Con solo (b,c): "a" queda suelto y "bc" se fusiona.
        let ranks2: HashMap<(String, String), usize> = [(("b".to_string(), "c".to_string()), 0)]
            .into_iter()
            .collect();
        let mut s2 = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        bpe(&mut s2, &ranks2);
        assert_eq!(s2, vec!["a".to_string(), "bc".to_string()]);
    }

    #[test]
    fn merges_vacias_no_fusionan() {
        let ranks = HashMap::new();
        let mut s = vec!["h".to_string(), "i".to_string()];
        bpe(&mut s, &ranks);
        assert_eq!(s.len(), 2);
    }

    #[test]
    fn encode_decode_roundtrip() {
        // Sin #[ignore] (cobertura-2e): CORRE donde el modelo esté presente y se salta
        // limpio si falta (ROSETTA_MODELS_DIR o models/ por defecto). Así cubre el
        // roundtrip en CI/local con el modelo descargado sin romper sin él.
        let models = std::env::var("ROSETTA_MODELS_DIR")
            .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/../../models").to_string());
        let dir = std::path::Path::new(&models).join("whisper-large-v3-turbo");
        let vocab = dir.join("vocab.json");
        let merges = dir.join("merges.txt");
        if !vocab.exists() || !merges.exists() {
            eprintln!("skip encode_decode_roundtrip: faltan vocab.json/merges.txt en {dir:?}");
            return;
        }
        let tok = WhisperTokenizer::from_files(&vocab, &merges).expect("cargar tokenizer");
        for text in [
            "Hola mundo",
            " transcripción con Rosetta",
            "GPT-2 byte-level BPE.",
        ] {
            let ids = tok.encode(text);
            assert!(!ids.is_empty(), "encode vacío para {text:?}");
            assert_eq!(tok.decode(&ids), text, "roundtrip falló para {text:?}");
        }
    }
}
