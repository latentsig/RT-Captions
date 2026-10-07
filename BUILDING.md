# Building RT Captions

Written so a person **or a coding agent** can follow it top to bottom. Every step is a command; each
ends with a check. Tested on Windows 11 (x64). The macOS path is written but has not yet been run on a
Mac: if a step fails there, fix it and update this file.

## What gets built

| Piece | From | Notes |
|---|---|---|
| `rt-captions-app` | Rust + Tauri 2 (`src-tauri/`) | the app: settings window, caption overlay, tray |
| `rt-captions` | same crate | console CLI; opens the app when run with no arguments |
| `libparakeet` | [parakeet.cpp](https://github.com/mudler/parakeet.cpp), built from source at a pinned commit | runs Nemotron, Parakeet Redux/Ultra |
| `libneedle` | Cactus's official prebuilt wheel | runs Cactus Whistle |
| models | downloaded by the app on first use (Hugging Face) | 17 MB to 1 GB each; not in the repo |

`src-tauri/scripts/setup-engines.mjs` fetches/builds the two native libraries into
`src-tauri/engine/<macos|windows>/` (gitignored). Builds are portable (`GGML_NATIVE=OFF`), so a
build made on one machine runs on others of the same OS/architecture.

## macOS (Apple Silicon)

```sh
# 1. Tools: Xcode command line tools, Homebrew packages, Rust, Tauri CLI
xcode-select --install                        # skip if already installed
brew install cmake ninja node git             # https://brew.sh if brew is missing
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
cargo install tauri-cli --version "^2" --locked
# check: cargo --version && cargo tauri --version && cmake --version && ninja --version

# 2. Get the code
git clone https://github.com/latentsig/RT-Captions.git
cd RT-Captions

# 3. Engine libraries (clones + builds parakeet.cpp with Metal, ~5-10 min; downloads libneedle)
node src-tauri/scripts/setup-engines.mjs
# check: ls src-tauri/engine/macos  ->  libneedle.dylib  libparakeet.dylib

# 4. Tests, then the app bundle and .dmg
cd src-tauri
cargo test --release --lib
cargo tauri build
# check: target/release/bundle/macos/RT Captions.app and target/release/bundle/dmg/*.dmg

# 5. Smoke test without the GUI (downloads the default engine, ~720 MB, on first run)
./target/release/rt-captions download nemotron-q4
./target/release/rt-captions doctor          # "loads OK"
./target/release/rt-captions listen --source mic --seconds 20   # speak; captions print
```

Install: open the `.dmg`, drag **RT Captions** to Applications. The build is unsigned, so the first
launch is right-click -> **Open** (or `xattr -dr com.apple.quarantine "/Applications/RT Captions.app"`).
macOS asks for microphone access on first use. Put the CLI on your PATH with
`"/Applications/RT Captions.app/Contents/MacOS/rt-captions" install-cli`.

macOS today: microphone captions only. System-audio capture (calls, videos) needs ScreenCaptureKit and
is not implemented yet; choosing "System" or "Both" captions the mic and says so. Engines may use
Metal (GPU) automatically.

## Windows (x64)

```powershell
# 1. Tools (PowerShell): Rust (MSVC toolchain), Node, Git, CMake + Ninja, Tauri CLI
winget install Rustlang.Rustup OpenJS.NodeJS.LTS Git.Git
rustup default stable-msvc
pip install cmake ninja                       # or: winget install Kitware.CMake Ninja-build.Ninja
cargo install tauri-cli --version "^2" --locked
# Visual Studio Build Tools with "Desktop development with C++" are needed by Rust/Tauri.

# 2. Code + engines (downloads the llvm-mingw clang toolchain and builds parakeet.cpp, ~5-10 min)
git clone https://github.com/latentsig/RT-Captions.git
cd RT-Captions
node src-tauri/scripts/setup-engines.mjs
# check: src-tauri\engine\windows has libparakeet.dll libc++.dll libomp.dll libunwind.dll libneedle.dll

# 3. Tests and installer
cd src-tauri
cargo test --release --lib
cargo tauri build
# check: target\release\bundle\nsis\RT Captions_<version>_x64-setup.exe
```

The installer bundles the app, the CLI (added to your PATH), and the engine libraries; models
download on first use.

## For agents

- Do not commit `src-tauri/engine/`, `src-tauri/binaries/`, models (`*.gguf`, `*.cact`) or audio.
- `cargo tauri build` runs `src-tauri/scripts/prepare-cli.mjs` first (builds the CLI and stages it as a
  sidecar); run it from the repo root or `src-tauri/`.
- Verify a change with the CLI instead of the GUI: `rt-captions bench clip.wav --ref @reference.txt`
  prints WER, word latency (p50/p95) and engine load; `rt-captions --help` lists everything.
- Settings live in `rt-captions settings` (a JSON file); a running app picks up `rt-captions set` live.
- If parakeet.cpp's pinned commit stops building, update `PARAKEET_COMMIT` in `setup-engines.mjs`
  and re-run the smoke test above before committing.
