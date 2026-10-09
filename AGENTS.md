# Agent notes

- Build instructions for macOS and Windows: `BUILDING.md` (follow it top to bottom; every step has a check).
- Code: `src-tauri/src/` (Rust). `engine.rs` = audio -> speech gate -> model -> caption events;
  `engines.rs` = engine catalogue + model downloads; `parakeet.rs` = native FFI (parakeet.cpp); `gguf.rs` = model-file metadata;
  `cli.rs` = every command; `lib.rs` = Tauri app, tray, settings watcher. UI: `ui/*.html` (no build step).
- Test: `cargo test --release --lib`; measure with `rt-captions bench <wav> --ref @<txt>`.
- Never commit engine libraries, models or audio (see .gitignore).
