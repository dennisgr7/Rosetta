//! Parseo de los Chrome-trace JSON de ONNX Runtime (`EnableProfiling`, activado
//! en Rosetta con la env `ROSETTA_ORT_PROFILE`) para reportar el **placement
//! real** por Execution Provider: cuántos nodos y cuánto tiempo se ejecutaron en
//! cada EP (DmlExecutionProvider, CPUExecutionProvider, QNNExecutionProvider, …).
//!
//! Parser (Rust puro) del Chrome-trace de ONNX Runtime. Regla dura del proyecto:
//! "EP activo = % de nodos colocados, no solo `register()`". Esta es la
//! verificación honesta de ese %.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::Context;

/// Acumulador por provider: nº de nodos y microsegundos totales.
#[derive(Default, Clone, Copy)]
struct Acc {
    nodos: u64,
    dur_us: u64,
}

/// Parsea todos los `prof_*.json` de `dir` (orden alfabético) y, por cada
/// archivo, imprime el reparto de nodos/tiempo por Execution Provider. Réplica
/// del placement real por Execution Provider (% de nodos y de tiempo).
pub fn cmd_profile(dir: &Path) -> anyhow::Result<()> {
    // glob "prof_*.json": leemos el directorio y filtramos por prefijo+sufijo,
    // luego ordenamos por nombre (como hace `sorted(glob.glob(...))` en Python).
    let mut paths: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("listar el directorio de perfiles {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("prof_") && n.ends_with(".json"))
        })
        .collect();
    paths.sort();

    for path in &paths {
        let base = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default();

        // Lee+parsea; un archivo corrupto no aborta el resto (como el Python).
        let data: serde_json::Value = match std::fs::read_to_string(path)
            .map_err(anyhow::Error::from)
            .and_then(|s| serde_json::from_str(&s).map_err(anyhow::Error::from))
        {
            Ok(v) => v,
            Err(e) => {
                println!("{base}: ERROR {e}");
                continue;
            }
        };

        // Acumula por provider los eventos `*_kernel_time` que llevan
        // `args.provider`. BTreeMap = orden estable de claves para los empates.
        let mut accs: BTreeMap<String, Acc> = BTreeMap::new();
        if let Some(events) = data.as_array() {
            for ev in events {
                let Some(ev) = ev.as_object() else { continue };
                let name = ev.get("name").and_then(|v| v.as_str()).unwrap_or("");
                let prov = ev
                    .get("args")
                    .and_then(|a| a.as_object())
                    .and_then(|a| a.get("provider"))
                    .and_then(|p| p.as_str());
                if let Some(prov) = prov
                    && name.ends_with("_kernel_time")
                {
                    let acc = accs.entry(prov.to_string()).or_default();
                    acc.nodos += 1;
                    // `dur` puede faltar (default 0); ORT lo emite en microsegundos.
                    acc.dur_us += ev.get("dur").and_then(|d| d.as_u64()).unwrap_or(0);
                }
            }
        }

        let total_nodes: u64 = accs.values().map(|a| a.nodos).sum::<u64>().max(1);
        let total_dur: u64 = accs.values().map(|a| a.dur_us).sum::<u64>().max(1);

        println!("\n=== {base} ===");
        // Orden: por duración descendente (como `sorted(..., key=lambda p: -dur[p])`).
        let mut provs: Vec<(&String, &Acc)> = accs.iter().collect();
        provs.sort_by_key(|p| std::cmp::Reverse(p.1.dur_us));
        for (prov, acc) in provs {
            let pct_nodos = 100.0 * acc.nodos as f64 / total_nodes as f64;
            let ms = acc.dur_us as f64 / 1000.0;
            let pct_dur = 100.0 * acc.dur_us as f64 / total_dur as f64;
            println!(
                "  {prov:28} nodos={n:5} ({pct_nodos:5.1}%)  tiempo={ms:8.1} ms ({pct_dur:5.1}%)",
                n = acc.nodos,
            );
        }
    }
    Ok(())
}
