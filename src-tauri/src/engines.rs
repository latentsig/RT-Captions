//! The speech engines users can pick, and on-demand download of their models.
//! The engine library (libparakeet) ships with the app; models (17 MB to 1 GB) are
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

/// Folders searched for model files: the app's models folder, plus a dev override.
fn model_dirs() -> Vec<PathBuf> {
    std::env::var_os("PARAKEET_MODEL_DIR").map(PathBuf::from).into_iter().chain([models_dir()]).collect()
}

/// One row of the engine picker: a catalogue engine (known: curated name, description,
/// download URL) or a model file found in the models folder (unknown: described from its own
/// metadata). Installed models also carry what their file says (quantization, look-ahead...).
#[derive(Serialize, Clone, Debug)]
pub struct Model {
    /// Catalogue id, or the file name for a local model.
    pub id: String,
    pub name: String,
    pub blurb: String,
    pub kind: Kind,
    pub file: String,
    pub bytes: u64,
    pub installed: bool,
    pub known: bool,
    pub quant: Option<String>,
    pub lookahead_ms: Option<u32>,
    pub languages: Vec<String>,
    /// Has a CTC head (hybrid model): decoded with CTC unless audio.decoder = "rnnt".
    pub ctc: bool,
    /// One line for the picker, e.g. "q8_0 · streaming, 480 ms look-ahead · en, hi".
    pub specs: String,
}

impl Model {
    fn finish(mut self) -> Self {
        let mut parts = Vec::new();
        if let Some(q) = &self.quant {
            parts.push(q.clone());
        }
        parts.push(match (self.kind, self.lookahead_ms) {
            (Kind::Streaming, Some(ms)) => format!("streaming{}, {ms} ms look-ahead", if self.ctc { " CTC" } else { "" }),
            (Kind::Streaming, None) => format!("streaming{}", if self.ctc { " CTC" } else { "" }),
            (Kind::Rolling, _) => "rolling window".into(),
        });
        if !self.languages.is_empty() {
            let shown: Vec<&str> = self.languages.iter().take(6).map(String::as_str).collect();
            let more = self.languages.len() - shown.len();
            parts.push(if more > 0 { format!("{} +{more} more", shown.join(", ")) } else { shown.join(", ") });
        }
        self.specs = parts.join(" · ");
        self
    }
}

/// Fills what the model file says about itself (falls back to the file name for quantization).
fn describe(m: &mut Model, path: &std::path::Path) {
    if let Ok(meta) = crate::gguf::read(path) {
        m.quant = meta.quant.filter(|q| !q.starts_with("type "));
        m.lookahead_ms = meta.lookahead_ms;
        m.languages = meta.languages;
        m.ctc = meta.ctc;
    }
    if m.quant.is_none() {
        m.quant = crate::gguf::quant_from_name(&m.file);
    }
}

/// Every engine the picker shows: the catalogue, then any other speech model file in the
/// models folder (dropped in by hand, or a custom build).
pub fn list() -> Vec<Model> {
    let blank = |id: String, name: String, blurb: String, kind: Kind, file: String, bytes: u64, installed: bool, known: bool| Model {
        id, name, blurb, kind, file, bytes, installed, known, quant: None, lookahead_ms: None, languages: Vec::new(), ctc: false, specs: String::new(),
    };
    let mut out: Vec<Model> = ENGINES.iter().map(|e| {
        let path = installed(e);
        let mut m = blank(e.id.into(), e.name.into(), e.blurb.into(), e.kind, e.file.into(), e.bytes, path.is_some(), true);
        match path {
            Some(p) => describe(&mut m, &p),
            None => m.quant = crate::gguf::quant_from_name(e.file),
        }
        m.finish()
    }).collect();
    let known: Vec<&str> = ENGINES.iter().chain(EXTRAS).map(|e| e.file).collect();
    let mut seen = std::collections::HashSet::new();
    for dir in model_dirs() {
        let Ok(rd) = std::fs::read_dir(&dir) else { continue };
        let mut files: Vec<PathBuf> = rd.flatten().map(|e| e.path()).collect();
        files.sort();
        for path in files {
            let file = path.file_name().unwrap_or_default().to_string_lossy().into_owned();
            if known.contains(&file.as_str()) || !seen.insert(file.clone()) {
                continue;
            }
            let bytes = std::fs::metadata(&path).map_or(0, |m| m.len());
            let stem = path.file_stem().unwrap_or_default().to_string_lossy().into_owned();
            let model = match path.extension().and_then(|x| x.to_str()) {
                Some("gguf") => match crate::gguf::read(&path) {
                    Ok(meta) if meta.is_asr => {
                        let kind = if meta.streaming { Kind::Streaming } else { Kind::Rolling };
                        let mut m = blank(file.clone(), stem, "Local model".into(), kind, file, bytes, true, false);
                        describe(&mut m, &path);
                        m
                    }
                    _ => continue, // diarization / speaker / VAD models, or not a model at all
                },
                _ => continue,
            };
            out.push(model.finish());
        }
    }
    out
}

/// Where an engine id's model is and how to run it: a catalogue id, or a local model's file
/// name (a bare name inside the models folder; no paths, so settings can't point elsewhere).
pub fn resolve(id: &str) -> Result<(PathBuf, Kind, String)> {
    if let Ok(e) = get(id) {
        let path = installed(e).with_context(|| format!("{} is not downloaded yet (run: rt-captions download {})", e.name, e.id))?;
        return Ok((path, e.kind, e.name.into()));
    }
    if id.is_empty() || id.contains(['/', '\\']) || id.contains("..") {
        bail!("unknown engine '{id}'");
    }
    let m = list().into_iter().find(|m| !m.known && m.id == id).with_context(|| {
        format!("unknown engine '{id}': not in the catalogue, and no such speech model in {}", models_dir().display())
    })?;
    let path = model_dirs().into_iter().map(|d| d.join(&m.file)).find(|p| p.exists()).context("model file vanished")?;
    Ok((path, m.kind, m.name))
}

/// True if `id` is a catalogue engine (downloaded or not) or a speech model in the models folder.
pub fn exists(id: &str) -> bool {
    get(id).is_ok() || list().iter().any(|m| !m.known && m.id == id)
}

/// The model file if it is on disk (downloaded, or a dev override in PARAKEET_MODEL_DIR).
pub fn installed(e: &Engine) -> Option<PathBuf> {
    model_dirs().into_iter().map(|d| d.join(e.file)).find(|p| p.exists())
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
