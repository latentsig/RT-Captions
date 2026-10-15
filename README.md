# RT Captions

Real-time, on-device captions for anything you hear: meetings, calls, videos and the person next to you.
Useful to everyone, including people who are hard of hearing, in a noisy room, or following along in a second language.
Private by design: audio never leaves your computer. Windows and macOS (Apple Silicon); Tauri 2 + Rust.

- Captions your microphone, your system audio (calls, videos), or both. With both, your voice is light yellow and everyone else light cyan.
- A floating, click-through caption overlay that appears only while someone speaks.
- Pick a speech engine; its model downloads on first use. English; Low / Balanced / Performance CPU modes.
- Your own models: drop a `.gguf` into the models folder and it appears in the engine list.
  English/Hindi hybrid models get a decoder choice; with CTC, optional beam search and **words to
  recognise** (names, medical terms such as "jaundice", including near-miss spellings).
- Everything in the settings window is also a CLI command, so scripts and agents can drive it.

## Engines

Measured live on real accented meeting audio (5 min, 3 speakers), Intel i5-12450H, 4 threads:

| Engine | Kind | Model | WER | Word on screen after |
|---|---|---|---|---|
| **Nemotron fine-tuned, 22 layers** (default) | streaming, 240 ms | 832 MB | ~10.5% | ~0.5 s (on 4-core laptops use Performance mode) |
| Nemotron, 19 layers | streaming, 240 ms | 724 MB | ~12% | ~0.4 s |
| Nemotron fine-tuned, 24 layers | streaming, 1040 ms | 904 MB | ~12% | ~1 s, light on the CPU |
| Parakeet Ultra | rolling window | 942 MB | ~17-21% live (11% offline) | ~1.8-3 s |
| Parakeet Redux | rolling window | 213 MB | ~20% | ~1 s |

Streaming engines process each piece of audio once; rolling-window engines re-transcribe the last
few seconds on every update (grey preview words, then solid). Silence is never sent to a model.
Engines run through our fork of [parakeet.cpp](https://github.com/mudler/parakeet.cpp) (MIT),
[latentsig/parakeet.cpp](https://github.com/latentsig/parakeet.cpp) (branch `rt-captions`); models come from their Hugging Face repos.

The three Nemotron engines are [NVIDIA Nemotron Speech Streaming EN 0.6B](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b)
fine-tuned and/or layer-pruned by fhai50032 ([fhai50032/bibo-asr-nemotron-gguf](https://huggingface.co/fhai50032/bibo-asr-nemotron-gguf)).
Licensed by NVIDIA Corporation under the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).

## CLI

    rt-captions                          open the app
    rt-captions engines                  list engines and which are downloaded
    rt-captions download ultra           fetch a model
    rt-captions set audio.engine redux audio.source both    (a running app applies it live)
    rt-captions set audio.hotwords '["jaundice", "Nikesh"]'  words to recognise (CTC decoder)
    rt-captions listen --source both     captions in the terminal, no window
    rt-captions transcribe clip.wav      run a recording through the live pipeline
    rt-captions bench clip.wav --ref @reference.txt   accuracy, latency and CPU
    rt-captions install-cli              put rt-captions on your PATH

`rt-captions --help` lists every command and setting key.

## Build

See **[BUILDING.md](BUILDING.md)** for macOS and Windows (also written for coding agents). In short:

    node src-tauri/scripts/setup-engines.mjs   # engine libraries for this OS
    cd src-tauri && cargo tauri build          # app + installer (.exe / .app + .dmg)

## Not yet

code signing / notarization, speaker diarization within one
source, dictation into text fields, Hinglish written in mixed scripts.
