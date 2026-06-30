# Releasing

Distribution is handled by [`dist`](https://github.com/axodotdev/cargo-dist) (config in the
root `Cargo.toml` under `[workspace.metadata.dist]`, workflow in
`.github/workflows/release.yml`).

## Cutting a release

1. Bump `version` in the root `Cargo.toml` (`[workspace.package]`).
2. Tag and push:
   ```sh
   git tag v0.1.0
   git push origin v0.1.0
   ```
3. GitHub Actions (`release.yml`) builds every target, produces the installers
   (shell, PowerShell, npm, Homebrew) and the self-updater, and publishes a GitHub Release.

Users then update with `rosetta-update`, `npm update -g rosetta-cli`, `brew upgrade rosetta`,
or by re-running the install one-liner.

To regenerate the workflow after changing the dist config: `dist init` / `dist generate`.

## ⚠️ TODO before the first real release — ONNX Runtime sidecar

The binary loads ONNX Runtime dynamically (`load-dynamic`). On first run, if it can't find the
shared library (next to the executable, in `runtime/<platform>/`, or via `ORT_DYLIB_PATH`), it
**downloads the official build for the current platform** — sha256-verified — into the model cache,
the same on-demand mechanism used for models (`rosetta_models::ensure_ort_runtime`). So a
`dist`-installed binary works out of the box; **no dylib sidecar is bundled in the release archive.**

Covered targets: Windows x64/arm64, Linux x64/arm64, macOS arm64. **Exception:** macOS x86_64 (Intel)
has no official ONNX Runtime release for v1.24.4 — set `ORT_DYLIB_PATH` manually there.

(The downloaded runtime is the CPU build, which is the sensible default everywhere — on Snapdragon
DirectML is slower than CPU anyway. For a discrete GPU, provide a GPU-enabled dylib via `ORT_DYLIB_PATH`.)
