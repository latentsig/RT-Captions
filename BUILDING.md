# Building RT Captions

Written so a person **or a coding agent** can follow it top to bottom. Every step is a command; each
ends with a check. Tested on Windows 11 x64 and macOS 15.6 on Apple Silicon.

## What gets built

| Piece | From | Notes |
|---|---|---|
| `rt-captions-app` | Rust + Tauri 2 (`src-tauri/`) | the app: settings window, caption overlay, tray |
| `rt-captions` | same crate | console CLI; opens the app when run with no arguments |
| `libparakeet` | [parakeet.cpp](https://github.com/mudler/parakeet.cpp), built from source at a pinned commit | runs Nemotron, Parakeet Redux/Ultra |
| models | downloaded by the app on first use (Hugging Face) | 17 MB to 1 GB each; not in the repo |

`src-tauri/scripts/setup-engines.mjs` fetches and builds the two native libraries into
`src-tauri/engine/<macos|windows>/`, which Git ignores. Builds use `GGML_NATIVE=OFF`, so a build made
on one machine runs on others with the same OS and architecture. On macOS, the script embeds the
Metal shaders in `libparakeet.dylib` and keeps all GGML code inside that library.

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

# 3. Engine libraries. This builds parakeet.cpp with Metal.
node src-tauri/scripts/setup-engines.mjs
# check: ls src-tauri/engine/macos  ->  libparakeet.dylib
# check: otool -L src-tauri/engine/macos/libparakeet.dylib
#        shows its @rpath install name and Apple system libraries, with no engine/.build paths

# 4. Tests, then the app bundle and .dmg
cd src-tauri
cargo test --release --lib
cargo tauri build
# check: target/release/bundle/macos/RT Captions.app and target/release/bundle/dmg/*.dmg
# check: target/release/bundle/macos/RT Captions.app/Contents/MacOS/rt-captions exists
#        and Contents/Resources/engine contains both dylibs

# 5. Smoke test without the GUI (downloads the default engine, ~720 MB, on first run)
./target/release/rt-captions download nemotron-q4
./target/release/rt-captions doctor          # "loads OK"
./target/release/rt-captions listen --source mic --seconds 20   # speak; captions print
```

Install: open the `.dmg`, drag **RT Captions** to Applications. The local build is ad hoc signed but
not notarized, so the first launch is right-click -> **Open**. You can also run
`xattr -dr com.apple.quarantine "/Applications/RT Captions.app"`.
macOS asks for microphone and Screen & System Audio Recording access on first use. Put the CLI on your PATH with
`"/Applications/RT Captions.app/Contents/MacOS/rt-captions" install-cli`.

System audio (calls, videos) uses ScreenCaptureKit (macOS 13+); first use asks for Screen & System
Audio Recording permission (the app, or the terminal when using the CLI). The default
Nemotron engine uses Metal automatically. Parakeet Redux currently fails in its upstream packed
ternary kernel on Metal. Run Redux with `PARAKEET_DEVICE=cpu`, for example:

```sh
rt-captions set audio.engine redux
PARAKEET_DEVICE=cpu rt-captions listen --source mic
```

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
# check: src-tauri\engine\windows has libparakeet.dll libc++.dll libomp.dll libunwind.dll

# 3. Tests and installer
cd src-tauri
cargo test --release --lib
cargo tauri build
# check: target\release\bundle\nsis\RT Captions_<version>_x64-setup.exe
```

The installer bundles the app, the CLI (added to your PATH), and the engine libraries; models
download on first use.

## For agents

- Do not commit `src-tauri/engine/`, `src-tauri/binaries/`, models (`*.gguf`) or audio.
- `cargo tauri build` builds both Cargo binaries and packages `rt-captions` with the app.
- Verify a change with the CLI instead of the GUI: `rt-captions bench clip.wav --ref @reference.txt`
  prints WER, word latency (p50/p95) and engine load; `rt-captions --help` lists everything.
- Named voices live in `people.json` (next to `models/`; vectors only, never audio): `rt-captions people`, `rt-captions enroll <name> <wav>`; test speaker names without speakers via `RT_CAPTIONS_FAKE_MIC=<wav> rt-captions listen --source mic`.
- Settings live in `rt-captions settings` (a JSON file); a running app picks up `rt-captions set` live.
- If parakeet.cpp's pinned commit stops building, update `PARAKEET_COMMIT` in `setup-engines.mjs`
  and re-run the smoke test above before committing.
