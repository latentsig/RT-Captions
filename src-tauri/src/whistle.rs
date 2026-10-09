//! Cactus Whistle via the needle engine's C API (libneedle). Signatures mirror the
//! official Python binding (needle/agent/whistle.py). The engine keeps one model per
//! process and is not thread-safe: own a `Whistle` from a single thread.

use anyhow::{anyhow, bail, Context, Result};
use libloading::Library;
use serde::{Deserialize, Serialize};
use std::ffi::{c_char, c_int, CStr, CString};
use std::path::{Path, PathBuf};

pub const SAMPLE_RATE: usize = 16_000;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Word {
    pub word: String,
    pub start: f64,
    pub end: f64,
}

#[derive(Deserialize)]
struct Output {
    #[serde(default)]
    words: Vec<Word>,
}

type LoadFn = unsafe extern "C" fn(*const c_char, u64) -> c_int;
type TranscribeFn =
    unsafe extern "C" fn(*const f32, c_int, *const c_char, *const c_char, c_int, *mut c_char, c_int) -> c_int;
type LastErrorFn = unsafe extern "C" fn() -> *const c_char;

pub struct Whistle {
    _lib: Library,
    transcribe: TranscribeFn,
    last_error: LastErrorFn,
    out: Vec<u8>,
    lang: CString,
    keywords: Option<CString>,
}

impl Whistle {
    pub fn load(lib_path: &Path, weights_path: &Path) -> Result<Self> {
        unsafe {
            let lib = Library::new(lib_path).with_context(|| format!("loading engine {}", lib_path.display()))?;
            let load: LoadFn = *lib.get(b"needle_load\0")?;
            let transcribe: TranscribeFn = *lib.get(b"needle_transcribe\0")?;
            let last_error: LastErrorFn = *lib.get(b"needle_last_error\0")?;
            let weights = std::fs::read(weights_path)
                .with_context(|| format!("reading weights {}", weights_path.display()))?;
            if load(weights.as_ptr() as *const c_char, weights.len() as u64) < 0 {
                bail!("needle_load: {}", CStr::from_ptr(last_error()).to_string_lossy());
            }
            Ok(Self { _lib: lib, transcribe, last_error, out: vec![0; 1 << 18], lang: CString::new("en")?, keywords: None })
        }
    }

    /// Words and phrases to favour (names, jargon). Empty clears.
    pub fn set_keywords(&mut self, keywords: &str) {
        let k = keywords.trim();
        self.keywords = if k.is_empty() { None } else { CString::new(k).ok() };
    }

    /// 16 kHz mono samples in [-1, 1], at most 30 s. Word times are relative to the clip start.
    pub fn transcribe(&mut self, samples: &[f32]) -> Result<Vec<Word>> {
        let kw = self.keywords.as_ref().map_or(std::ptr::null(), |k| k.as_ptr());
        let code = unsafe {
            (self.transcribe)(
                samples.as_ptr(),
                samples.len() as c_int,
                self.lang.as_ptr(),
                kw,
                1,
                self.out.as_mut_ptr() as *mut c_char,
                self.out.len() as c_int,
            )
        };
        if code < 0 {
            return Err(anyhow!("needle_transcribe: {}", unsafe { CStr::from_ptr((self.last_error)()) }.to_string_lossy()));
        }
        let json = CStr::from_bytes_until_nul(&self.out)?.to_str()?;
        Ok(serde_json::from_str::<Output>(json)?.words)
    }
}

fn lib_name() -> &'static str {
    if cfg!(windows) { "libneedle.dll" } else if cfg!(target_os = "macos") { "libneedle.dylib" } else { "libneedle.so" }
}

fn engine_platform() -> &'static str {
    if cfg!(windows) { "windows" } else if cfg!(target_os = "macos") { "macos" } else { "linux" }
}

fn home() -> PathBuf {
    std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME")).map(PathBuf::from).unwrap_or_default()
}

fn exe_dir() -> PathBuf {
    std::env::current_exe().ok().and_then(|p| p.parent().map(Path::to_path_buf)).unwrap_or_default()
}

/// Newest subdirectory (by name) of `dir` that contains `file`.
fn newest_with(dir: &Path, file: &str) -> Option<PathBuf> {
    let mut found: Vec<PathBuf> = std::fs::read_dir(dir).ok()?.flatten().map(|e| e.path().join(file)).filter(|p| p.exists()).collect();
    found.sort();
    found.pop()
}

/// The bundled libneedle (next to the exe, or Contents/Resources on macOS); NEEDLE_LIB overrides,
/// and the cache `pip install cactus-needle` fills is a dev fallback.
pub fn find_lib() -> Result<PathBuf> {
    let d = exe_dir();
    std::env::var_os("NEEDLE_LIB").map(PathBuf::from).into_iter()
        .chain([
            d.join(lib_name()),
            d.join("engine").join(lib_name()),
            d.join("../Resources/engine").join(lib_name()),
            d.join("../../engine").join(engine_platform()).join(lib_name()),
        ])
        .find(|p| p.exists())
        .or_else(|| newest_with(&home().join(".cache/cactus-needle/v3"), lib_name()))
        .ok_or_else(|| anyhow!("{} not found next to the app", lib_name()))
}
