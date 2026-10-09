//! The speech engines users can pick, and on-demand download of their models.
//! Engine libraries (libneedle, libparakeet) ship with the app; models (17 MB to 1 GB) are
//! fetched from their original Hugging Face repos the first time an engine is selected.

use anyhow::{bail, Context, Result};
use serde::Serialize;
use std::io::{Read, Write};
use std::path::PathBuf;

#[derive(Serialize, Clone, Copy, PartialEq, Debug)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    /// Offline model re-run on a rolling window (grey preview text, then solid).
    Rolling,
    /// Cache-aware streaming model: each piece of audio is processed once.
    Streaming,
}

#[derive(Serialize, Clone, Copy, Debug)]
pub struct Engine {
    pub id: &'static str,
    pub name: &'static str,
    pub blurb: &'static str,
    pub kind: Kind,
    pub file: &'static str,
    pub url: &'static str,
    pub bytes: u64,
    /// Fastest update interval this engine keeps up with (rolling engines).
    pub min_hop_s: f64,
}

pub const DEFAULT: &str = "nemotron-q4";

// Measured on a 5-minute real meeting (i5-12450H, 4 threads).
pub const ENGINES: &[Engine] = &[
    Engine {
        id: "nemotron-q4",
        name: "Nemotron streaming (4-bit)",
        blurb: "Fastest accurate engine. ~17% WER, words in ~0.3-0.5 s",
        kind: Kind::Streaming,
        file: "nemotron-3.5-asr-streaming-0.6b-q4_k.gguf",
        url: "https://huggingface.co/mudler/parakeet-cpp-gguf/resolve/main/nemotron-3.5-asr-streaming-0.6b-q4_k.gguf",
        bytes: 718_102_624,
        min_hop_s: 0.0,
    },
    Engine {
        id: "nemotron-q8",
        name: "Nemotron streaming (8-bit)",
        blurb: "~15% WER, words in ~0.3-0.5 s, more CPU",
        kind: Kind::Streaming,
        file: "nemotron-3.5-asr-streaming-0.6b-q8_0.gguf",
        url: "https://huggingface.co/mudler/parakeet-cpp-gguf/resolve/main/nemotron-3.5-asr-streaming-0.6b-q8_0.gguf",
        bytes: 983_696_512,
        min_hop_s: 0.0,
    },
    Engine {
        id: "ultra",
        name: "Parakeet Ultra",
        blurb: "Most accurate (~11-15% WER). Words in ~1.5-2.5 s",
        kind: Kind::Rolling,
        file: "ultra-q8_0.gguf",
        url: "https://huggingface.co/mudler/parakeet-cpp-gguf/resolve/main/ultra-q8_0.gguf",
        bytes: 941_517_728,
        min_hop_s: 0.75,
    },
    Engine {
        id: "redux",
        name: "Parakeet Redux",
        blurb: "~20% WER, words in ~1 s",
        kind: Kind::Rolling,
        file: "redux-packed.gguf",
        url: "https://huggingface.co/mudler/parakeet-cpp-gguf/resolve/main/redux-packed.gguf",
        bytes: 213_319_296,
        min_hop_s: 0.5,
    },
    Engine {
        id: "whistle",
        name: "Cactus Whistle",
        blurb: "Tiny and fastest, ~32% WER on accented speech",
        kind: Kind::Rolling,
        file: "whistle.cact",
        url: "https://huggingface.co/Cactus-Compute/whistle/resolve/main/whistle.cact",
        bytes: 16_919_407,
        min_hop_s: 0.15,
    },
];

/// Optional add-on models for speaker labels (not speech engines; `kind`/`min_hop_s` are unused).
pub const EXTRAS: &[Engine] = &[
    Engine {
        id: "diarization",
        name: "Speaker diarization",
        blurb: "Tells voices apart (Nemotron 3 diarization, 8-bit)",
        kind: Kind::Streaming,
        file: "nemotron-3-diarization-q8_0.gguf",
        url: "https://huggingface.co/mudler/parakeet-cpp-gguf/resolve/main/nemotron-3-diarization-q8_0.gguf",
        bytes: 108_674_624,
        min_hop_s: 0.0,
    },
    Engine {
        id: "speakers",
        name: "Speaker recognition",
        blurb: "Recognises saved voices (WeSpeaker ResNet34, f32)",
        kind: Kind::Streaming,
        file: "wespeaker-resnet34-voxceleb.gguf",
        url: "https://huggingface.co/mudler/voice-detect-gguf/resolve/main/wespeaker-resnet34-voxceleb.gguf",
        bytes: 26_527_360,
        min_hop_s: 0.0,
    },
];

/// An engine or an extra model by id (what `download` accepts).
pub fn get_any(id: &str) -> Result<&'static Engine> {
    EXTRAS.iter().find(|e| e.id == id).map_or_else(|| get(id), Ok)
}

pub fn get(id: &str) -> Result<&'static Engine> {
    ENGINES.iter().find(|e| e.id == id).with_context(|| {
        format!("unknown engine '{id}' (one of: {})", ENGINES.iter().map(|e| e.id).collect::<Vec<_>>().join(", "))
    })
}

pub fn models_dir() -> PathBuf {
    std::env::var_os("RT_CAPTIONS_MODELS").map(PathBuf::from)
        .unwrap_or_else(|| dirs::data_local_dir().unwrap_or_default().join(crate::settings::IDENTIFIER).join("models"))
}

/// The model file if it is on disk (downloaded, or a dev override in PARAKEET_MODEL_DIR).
pub fn installed(e: &Engine) -> Option<PathBuf> {
    let dev = std::env::var_os("PARAKEET_MODEL_DIR").map(|d| PathBuf::from(d).join(e.file));
    dev.into_iter().chain([models_dir().join(e.file)]).find(|p| p.exists())
}

/// Downloads the model (resumable via a .part file); `progress(done, total)` is called as it goes.
pub fn download(e: &Engine, mut progress: impl FnMut(u64, u64)) -> Result<PathBuf> {
    if let Some(p) = installed(e) {
        return Ok(p);
    }
    let dir = models_dir();
    std::fs::create_dir_all(&dir)?;
    let part = dir.join(format!("{}.part", e.file));
    let have = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
    let mut req = ureq::get(e.url);
    if have > 0 {
        req = req.set("Range", &format!("bytes={have}-"));
    }
    let resp = req.call().with_context(|| format!("downloading {}", e.url))?;
    let resumed = resp.status() == 206;
    let mut done = if resumed { have } else { 0 };
    let mut out = std::fs::OpenOptions::new().create(true).write(true).append(resumed).truncate(!resumed).open(&part)?;
    let mut body = resp.into_reader();
    let mut buf = vec![0u8; 1 << 20];
    let mut last = 0;
    loop {
        let n = body.read(&mut buf)?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n])?;
        done += n as u64;
        if done - last >= 4 << 20 {
            progress(done, e.bytes);
            last = done;
        }
    }
    out.flush()?;
    drop(out);
    if done != e.bytes {
        bail!("download of {} stopped at {done} of {} bytes; try again to resume", e.file, e.bytes);
    }
    let path = dir.join(e.file);
    std::fs::rename(&part, &path)?;
    progress(done, e.bytes);
    Ok(path)
}
