//! Vocabulario / detokenizador para el formato `vocab.txt` de los modelos NeMo
//! TDT (una línea `token id` por entrada; subwords SentencePiece con `▁` como
//! marca de espacio).

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use rosetta_core::{Result, RosettaError};

/// Marca de espacio de SentencePiece (U+2581).
const SPACE_MARK: char = '\u{2581}';

/// Vocabulario id → token.
#[derive(Debug, Clone)]
pub struct Vocabulary {
    id_to_token: Vec<String>,
    /// Id del token blank (`<blk>`), normalmente el último.
    pub blank_id: usize,
}

impl Vocabulary {
    /// Carga el vocabulario desde un `vocab.txt` (`token id` por línea).
    pub fn from_file(path: &Path) -> Result<Self> {
        let file = File::open(path)
            .map_err(|e| RosettaError::Model(format!("abrir vocab {}: {e}", path.display())))?;
        let reader = BufReader::new(file);

        let mut id_to_token: Vec<String> = Vec::new();
        let mut blank_id = 0usize;

        for line in reader.lines() {
            let line = line.map_err(|e| RosettaError::Model(format!("leer vocab: {e}")))?;
            // Separar por el ÚLTIMO espacio: el token puede contener espacios/marcas.
            let Some(sep) = line.rfind(' ') else {
                continue;
            };
            let token = &line[..sep];
            let Ok(id) = line[sep + 1..].trim().parse::<usize>() else {
                continue;
            };
            if id >= id_to_token.len() {
                id_to_token.resize(id + 1, String::new());
            }
            id_to_token[id] = token.to_string();
            if token == "<blk>" || token == "<blank>" {
                blank_id = id;
            }
        }

        if id_to_token.is_empty() {
            return Err(RosettaError::Model("vocab vacío".into()));
        }
        if blank_id == 0 {
            blank_id = id_to_token.len() - 1;
        }

        Ok(Self {
            id_to_token,
            blank_id,
        })
    }

    /// Número de tokens (incluido el blank).
    pub fn size(&self) -> usize {
        self.id_to_token.len()
    }

    /// Detokeniza una secuencia de ids a texto, ignorando tokens especiales y
    /// convirtiendo la marca `▁` en espacios.
    pub fn decode(&self, ids: &[usize]) -> String {
        let mut out = String::new();
        for &id in ids {
            if let Some(tok) = self.id_to_token.get(id) {
                if is_special(tok) {
                    continue;
                }
                out.push_str(tok);
            }
        }
        out.replace(SPACE_MARK, " ").trim().to_string()
    }

    /// Agrupa `(ids, frames)` en palabras: `(texto, frame_inicio, frame_fin)`.
    /// Una palabra nueva empieza en cada token con la marca `▁`; los demás se
    /// anexan a la palabra en curso.
    pub fn decode_words(&self, ids: &[usize], frames: &[usize]) -> Vec<(String, usize, usize)> {
        let mut words: Vec<(String, usize, usize)> = Vec::new();
        for (&id, &fr) in ids.iter().zip(frames) {
            let Some(tok) = self.id_to_token.get(id) else {
                continue;
            };
            if is_special(tok) {
                continue;
            }
            let starts_word = tok.starts_with(SPACE_MARK);
            let piece = tok.replace(SPACE_MARK, "");
            if starts_word || words.is_empty() {
                if piece.is_empty() {
                    continue;
                }
                words.push((piece, fr, fr));
            } else if let Some(last) = words.last_mut() {
                last.0.push_str(&piece);
                last.2 = fr;
            }
        }
        words
    }
}

/// ¿Es un token especial que no debe aparecer en la salida?
fn is_special(tok: &str) -> bool {
    matches!(
        tok,
        "<unk>" | "<pad>" | "<blk>" | "<blank>" | "<s>" | "</s>"
    ) || tok.starts_with("<|")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detokeniza_sentencepiece() {
        let v = Vocabulary {
            id_to_token: vec![
                "<unk>".into(),
                "\u{2581}hola".into(),
                "\u{2581}mun".into(),
                "do".into(),
                "<blk>".into(),
            ],
            blank_id: 4,
        };
        assert_eq!(v.decode(&[1, 2, 3]), "hola mundo");
        // ignora especiales (<unk>=0, <blk>=4) y concatena los subwords
        assert_eq!(v.decode(&[0, 1, 2, 3, 4]), "hola mundo");
    }
}
