//! `xtask`: tooling interno del workspace Rosetta (100% Rust, sin C).
//!
//! Subcomandos:
//! - `check-no-c`: guardia de la regla dura "sin C/C++" en la build de Windows.
//!   Guardia sin-C por-target, en Rust puro: por cada
//!   target de Windows resuelve el grafo REAL por-target con `cargo tree` y verifica
//!   que ningún crate de la lista prohibida (que compila C/C++) aparezca en él.
//!
//! Uso: `cargo xtask check-no-c` (alias en `.cargo/config.toml`).

use std::process::{Command, ExitCode};

/// Targets de Windows que NO deben arrastrar compilación de C/C++.
const TARGETS: &[&str] = &["aarch64-pc-windows-msvc", "x86_64-pc-windows-msvc"];

/// Crates que compilan C/C++ o invocan un compilador de C en su build script,
/// prohibidos en el build de Windows.
const BANNED: &[&str] = &[
    "cc",
    "cmake",
    "bindgen",
    "openssl-sys",
    "bzip2-sys",
    "knf-rs",
    "openblas-src",
    "openblas-build",
    "hdf5-sys",
    "ring",
    "aws-lc-sys",
    "aws-lc-fips-sys",
];

fn main() -> ExitCode {
    // El primer argumento es el subcomando (igual que clap, pero a mano: sin deps).
    let sub = std::env::args().nth(1);
    match sub.as_deref() {
        Some("check-no-c") => check_no_c(),
        Some(other) => {
            eprintln!("xtask: subcomando desconocido '{other}'");
            eprintln!("Subcomandos disponibles: check-no-c");
            ExitCode::FAILURE
        }
        None => {
            eprintln!("xtask: falta el subcomando");
            eprintln!("Subcomandos disponibles: check-no-c");
            ExitCode::FAILURE
        }
    }
}

/// Guardia sin-C por-target (objetivos de Windows).
///
/// Para cada target de Windows ejecuta:
///   `cargo tree --target <tgt> -e normal,build --prefix none -f {p}`
/// que resuelve el grafo de ESE target (features y plataforma resueltas) sin
/// necesitar el toolchain instalado, por lo que corre igual en Linux (CI). Luego
/// extrae los nombres de crate (primer token de cada línea, deduplicados) y
/// comprueba que ninguno de `BANNED` aparezca con coincidencia EXACTA de nombre.
fn check_no_c() -> ExitCode {
    let mut fail = false;

    for tgt in TARGETS {
        let names = match crate_names_for_target(tgt) {
            Ok(names) => names,
            Err(err) => {
                eprintln!("ERROR [{tgt}]: no se pudo ejecutar `cargo tree`: {err}");
                return ExitCode::FAILURE;
            }
        };

        for bad in BANNED {
            // grep -qx => coincidencia de línea completa (nombre exacto de crate).
            if names.iter().any(|n| n == bad) {
                println!("ERROR [{tgt}]: la crate '{bad}' está en la build (compilaría C/C++)");
                fail = true;
            }
        }
    }

    if fail {
        println!("Regla sin-C violada: la build de Windows no debe requerir compilar C/C++.");
        return ExitCode::FAILURE;
    }
    println!("OK: las builds de Windows (ARM64 y x64) no arrastran crates que compilen C/C++.");
    ExitCode::SUCCESS
}

/// Ejecuta `cargo tree` para un target y devuelve los nombres de crate únicos.
///
/// Replica el pipeline del .sh: `cargo tree … -f '{p}'` (el formato `{p}` imprime
/// `<nombre> <versión> [ruta]`), de cada línea se toma el primer token (el nombre)
/// y se deduplica (equivalente a `awk '{print $1}' | sort -u`). El orden no importa
/// porque la comprobación es por pertenencia.
fn crate_names_for_target(tgt: &str) -> Result<Vec<String>, std::io::Error> {
    // `cargo` respeta CARGO si está definido (toolchain seleccionado), si no usa PATH.
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let output = Command::new(cargo)
        .args([
            "tree",
            "--target",
            tgt,
            "-e",
            "normal,build",
            "--prefix",
            "none",
            "-f",
            "{p}",
        ])
        .output()?;

    // El .sh redirige stderr a /dev/null (2>/dev/null) y sigue; aquí ignoramos
    // stderr salvo que el comando ni siquiera arrancase (eso lo captura el `?`).
    let stdout = String::from_utf8_lossy(&output.stdout);

    let mut names: Vec<String> = stdout
        .lines()
        .filter_map(|line| line.split_whitespace().next())
        .map(|s| s.to_string())
        .collect();
    names.sort();
    names.dedup();
    Ok(names)
}
