//! NVIDIA Parakeet (Moondream Redux / Ultra GGUF) via parakeet.cpp's C API (libparakeet).
//! Offline models (Redux) decode a whole clip per call; cache-aware streaming models
//! (Nemotron) take audio a piece at a time through `stream_feed`.

use crate::whistle::{Word, SAMPLE_RATE};
use anyhow::{anyhow, bail, Context, Result};
use libloading::Library;
use serde::Deserialize;
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::path::{Path, PathBuf};

type LoadFn = unsafe extern "C" fn(*const c_char) -> *mut c_void;
type LoadErrFn = unsafe extern "C" fn() -> *const c_char;
type BatchJsonFn = unsafe extern "C" fn(*mut c_void, *const f32, *const c_int, c_int, c_int, c_int) -> *mut c_char;
type FreeStrFn = unsafe extern "C" fn(*mut c_char);
type LastErrFn = unsafe extern "C" fn(*mut c_void) -> *const c_char;
type FreeFn = unsafe extern "C" fn(*mut c_void);
type StreamBeginFn = unsafe extern "C" fn(*mut c_void, *const c_char) -> *mut c_void;
type StreamFeedFn = unsafe extern "C" fn(*mut c_void, *const f32, c_int) -> *mut c_char;
type StreamFinalizeFn = unsafe extern "C" fn(*mut c_void) -> *mut c_char;

#[derive(Deserialize)]
struct Out {
    #[serde(default)]
    text: String,
    #[serde(default)]
    words: Vec<PWord>,
}

#[derive(Deserialize)]
struct PWord {
    w: String,
    start: f64,
    end: f64,
}

pub struct Parakeet {
    _lib: Library,
    ctx: *mut c_void,
    batch_json: BatchJsonFn,
    free_str: FreeStrFn,
    last_err: LastErrFn,
    free: FreeFn,
    stream_begin: StreamBeginFn,
    stream_feed: StreamFeedFn,
    stream_finalize: StreamFinalizeFn,
    stream_free: FreeFn,
    /// One streaming session per audio source.
    streams: Vec<*mut c_void>,
    /// Language prompt for new streaming sessions ("en", "hi", "auto").
    lang: CString,
}

// The context is only ever used from the engine thread that owns it.
unsafe impl Send for Parakeet {}

impl Parakeet {
    /// `threads` > 0 sets the compute threads (library default: 8).
    pub fn load(lib_path: &Path, model_path: &Path, threads: usize) -> Result<Self> {
        // Windows: CPU (measured). macOS: let the library pick Metal on Apple Silicon.
        if !cfg!(target_os = "macos") {
            std::env::set_var("PARAKEET_DEVICE", "cpu");
        }
        unsafe {
            let lib = load_library(lib_path)?;
            if threads > 0 {
                // ponytail: not in the C API; pk::set_num_threads(int) is exported from the clang
                // (Itanium-mangled) build. Upgrade: upstream a parakeet_capi_set_threads.
                if let Ok(set) = lib.get::<unsafe extern "C" fn(c_int)>(b"_ZN2pk15set_num_threadsEi\0") {
                    set(threads as c_int);
                }
            }
            let load: LoadFn = *lib.get(b"parakeet_capi_load\0")?;
            let load_err: LoadErrFn = *lib.get(b"parakeet_capi_load_error\0")?;
            let path = CString::new(model_path.to_string_lossy().as_bytes())?;
            let ctx = load(path.as_ptr());
            if ctx.is_null() {
                bail!("parakeet load {}: {}", model_path.display(), CStr::from_ptr(load_err()).to_string_lossy());
            }
            Ok(Self {
                batch_json: *lib.get(b"parakeet_capi_transcribe_pcm_batch_json\0")?,
                free_str: *lib.get(b"parakeet_capi_free_string\0")?,
                last_err: *lib.get(b"parakeet_capi_last_error\0")?,
                free: *lib.get(b"parakeet_capi_free\0")?,
                stream_begin: *lib.get(b"parakeet_capi_stream_begin_lang\0")?,
                stream_feed: *lib.get(b"parakeet_capi_stream_feed_json\0")?,
                stream_finalize: *lib.get(b"parakeet_capi_stream_finalize_json\0")?,
                stream_free: *lib.get(b"parakeet_capi_stream_free\0")?,
                streams: Vec::new(),
                lang: c"en".to_owned(),
                ctx,
                _lib: lib,
            })
        }
    }

    pub fn transcribe(&mut self, samples: &[f32]) -> Result<Vec<Word>> {
        if samples.len() < SAMPLE_RATE / 10 {
            return Ok(Vec::new());
        }
        let n = [samples.len() as c_int];
        let json = unsafe {
            let p = (self.batch_json)(self.ctx, samples.as_ptr(), n.as_ptr(), 1, SAMPLE_RATE as c_int, 0);
            if p.is_null() {
                return Err(anyhow!("parakeet: {}", CStr::from_ptr((self.last_err)(self.ctx)).to_string_lossy()));
            }
            let s = CStr::from_ptr(p).to_string_lossy().into_owned();
            (self.free_str)(p);
            s
        };
        // One object per clip; we always send one clip.
        let out = serde_json::from_str::<Vec<Out>>(&json)?.into_iter().next().unwrap_or(Out { text: String::new(), words: Vec::new() });
        Ok(out.words.into_iter().map(|w| Word { word: w.w, start: w.start, end: w.end }).collect())
    }
}

impl Parakeet {
    fn take_json(&self, p: *mut c_char) -> Result<String> {
        if p.is_null() {
            return Err(anyhow!("parakeet: {}", unsafe { CStr::from_ptr((self.last_err)(self.ctx)) }.to_string_lossy()));
        }
        let s = unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned();
        unsafe { (self.free_str)(p) };
        Ok(s)
    }

    /// Streaming models: feed 16 kHz audio for `source`. Returns the new raw text (word pieces as
    /// soon as decoded, e.g. " the tra" then "ining") and the words completed by it, with times
    /// from that source's stream start. Show the text: a word is only completed once the next
    /// word starts, so the last word before a pause would wait for the next sentence.
    /// The language prompt (default English) stops multilingual drift.
    pub fn stream_feed(&mut self, source: usize, pcm: &[f32]) -> Result<(String, Vec<Word>)> {
        while self.streams.len() <= source {
            self.streams.push(std::ptr::null_mut());
        }
        if self.streams[source].is_null() {
            let s = unsafe { (self.stream_begin)(self.ctx, self.lang.as_ptr()) };
            if s.is_null() {
                bail!("parakeet stream: {}", unsafe { CStr::from_ptr((self.last_err)(self.ctx)) }.to_string_lossy());
            }
            self.streams[source] = s;
        }
        let p = unsafe { (self.stream_feed)(self.streams[source], pcm.as_ptr(), pcm.len() as c_int) };
        self.stream_out(p)
    }

    /// End of an utterance: flush the audio the model is still holding (so the last word shows
    /// now, not when someone speaks again) and close the session; the next feed starts a new one.
    pub fn stream_finish(&mut self, source: usize) -> Result<(String, Vec<Word>)> {
        let Some(&st) = self.streams.get(source).filter(|s| !s.is_null()) else { return Ok(Default::default()) };
        let p = unsafe { (self.stream_finalize)(st) };
        let out = self.stream_out(p);
        unsafe { (self.stream_free)(st) };
        self.streams[source] = std::ptr::null_mut();
        out
    }

    fn stream_out(&self, p: *mut c_char) -> Result<(String, Vec<Word>)> {
        let out: Out = serde_json::from_str(&self.take_json(p)?)?;
        // The model writes language tags like "<en-US>" into its text; captions don't want them.
        let text = strip_tags(&out.text);
        let words = out.words.into_iter().filter(|w| !(w.w.starts_with('<') && w.w.ends_with('>')))
            .map(|w| Word { word: w.w, start: w.start, end: w.end }).collect();
        Ok((text, words))
    }

    /// Language for streaming sessions started from now on (call `reset_streams` to apply).
    /// Only English and Hindi are offered: "auto" would let the model drift into 40 languages.
    pub fn set_language(&mut self, lang: &str) {
        self.lang = if lang == "hi" { c"hi".to_owned() } else { c"en".to_owned() };
    }

    /// Drop all streaming sessions (new settings start fresh streams).
    pub fn reset_streams(&mut self) {
        for s in self.streams.drain(..).filter(|s| !s.is_null()) {
            unsafe { (self.stream_free)(s) };
        }
    }
}

/// Removes `<...>` tags (language markers) and the space before them.
fn strip_tags(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(i) = rest.find('<') {
        match rest[i..].find('>') {
            Some(j) => {
                out.push_str(rest[..i].trim_end_matches(' '));
                rest = &rest[i + j + 1..];
            }
            None => break,
        }
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    #[test]
    fn strips_language_tags() {
        assert_eq!(super::strip_tags("होगी। <hi-IN> प्लीज"), "होगी। प्लीज");
        assert_eq!(super::strip_tags(" the tra"), " the tra");
    }
}

impl Drop for Parakeet {
    fn drop(&mut self) {
        self.reset_streams();
        unsafe { (self.free)(self.ctx) }
    }
}

/// libparakeet needs libc++/libomp/libunwind from its own folder; on Windows a plain
/// LoadLibrary only searches the exe's folder for those, so ask for the DLL's folder too.
fn load_library(path: &Path) -> Result<Library> {
    #[cfg(windows)]
    let lib = unsafe {
        use libloading::os::windows::{Library as WinLib, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR};
        WinLib::load_with_flags(path, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS).map(Library::from)
    };
    #[cfg(not(windows))]
    let lib = unsafe { Library::new(path) };
    lib.with_context(|| format!("loading {}", path.display()))
}

fn lib_name() -> &'static str {
    if cfg!(windows) { "libparakeet.dll" } else if cfg!(target_os = "macos") { "libparakeet.dylib" } else { "libparakeet.so" }
}

/// The bundled libparakeet (next to the exe, or Contents/Resources on macOS); PARAKEET_LIB overrides.
pub fn find_lib() -> Result<PathBuf> {
    let exe = std::env::current_exe()?.parent().context("exe dir")?.to_path_buf();
    std::env::var_os("PARAKEET_LIB").map(PathBuf::from).into_iter()
        .chain([exe.join(lib_name()), exe.join("engine").join(lib_name()), exe.join("../Resources/engine").join(lib_name())])
        .find(|p| p.exists())
        .ok_or_else(|| anyhow!("{} not found next to the app", lib_name()))
}
