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

The binary loads ONNX Runtime dynamically (`load-dynamic`); `ensure_ort_dylib()` resolves the
shared library next to the executable. The generated `release.yml` builds the binary but does
**not yet ship the onnxruntime shared library** alongside it, so an installed binary would not
find a runtime out of the box.

Pick one before tagging:

- **Bundle the dylib (recommended).** Add a step to the `build-local-artifacts` job that, per
  target, downloads the matching `onnxruntime` shared library (Windows DirectML wheel / Linux /
  macOS tarball — same URLs and SHA-256 verification as `ci.yml`) into the staging directory, and
  reference it from `include = [...]` in `[workspace.metadata.dist]`. Then set `allow-dirty = ["ci"]`
  so `dist generate` keeps the manual step. This needs one real tag run to validate.
- **Fetch at runtime.** Alternatively, extend `ensure_ort_dylib()` to download the matching dylib
  on first run if missing (like models are fetched), and cache it next to the binary or in the
  model cache. Keeps releases trivial; it is a runtime download of a prebuilt library (no C compiled).

Until this is done, install from source or use the Docker image (which bundles the dylib).
