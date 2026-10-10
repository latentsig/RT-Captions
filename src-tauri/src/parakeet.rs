//! NVIDIA Parakeet (Moondream Redux / Ultra GGUF) via parakeet.cpp's C API (libparakeet).
//! Offline models (Redux) decode a whole clip per call; cache-aware streaming models
//! (Nemotron) take audio a piece at a time through `stream_feed`.

use crate::asr::{Word, SAMPLE_RATE};
use anyhow::{anyhow, bail, Context, Result};
use libloading::Library;
use serde::Deserialize;
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

type LoadFn = unsafe extern "C" fn(*const c_char) -> *mut c_void;
type LoadErrFn = unsafe extern "C" fn() -> *const c_char;
type BatchJsonFn = unsafe extern "C" fn(*mut c_void, *const f32, *const c_int, c_int, c_int, c_int) -> *mut c_char;
type FreeStrFn = unsafe extern "C" fn(*mut c_char);
type LastErrFn = unsafe extern "C" fn(*mut c_void) -> *const c_char;
type FreeFn = unsafe extern "C" fn(*mut c_void);
type StreamBeginFn = unsafe extern "C" fn(*mut c_void, *const c_char) -> *mut c_void;
type StreamFeedFn = unsafe extern "C" fn(*mut c_void, *const f32, c_int) -> *mut c_char;
type StreamFinalizeFn = unsafe extern "C" fn(*mut c_void) -> *mut c_char;
type ShutdownBackendFn = unsafe extern "C" fn();

// parakeet.cpp shares one GGML backend across contexts. Loading and final teardown must not overlap.
static LIVE_CONTEXTS: Mutex<usize> = Mutex::new(0);

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
    shutdown_backend: ShutdownBackendFn,
    /// One streaming session per audio source.
    streams: Vec<*mut c_void>,
    /// Language prompt for new streaming sessions ("en", "hi", "auto").
    lang: CString,
    /// parakeet_capi_set_decode_options (our fork's patch); None with an upstream library.
    set_decode_options: Option<unsafe extern "C" fn(*const c_char, f32, f32)>,
    /// parakeet_capi_set_suppressed_tokens (our fork's patch).
    set_suppressed: Option<unsafe extern "C" fn(*const c_int, c_int)>,
    /// parakeet_capi_stream_ctc_rows (our fork's patch).
    stream_ctc_rows: Option<unsafe extern "C" fn(*mut c_void, *mut c_int, *mut c_int) -> *const f32>,
    /// CTC scores of the last feed/finish ([frames x classes]), for hotwords.rs.
    pub ctc: (Vec<f32>, usize),
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
            let mut live_contexts = LIVE_CONTEXTS.lock().unwrap_or_else(|e| e.into_inner());
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
            let batch_json: BatchJsonFn = *lib.get(b"parakeet_capi_transcribe_pcm_batch_json\0")?;
            let free_str: FreeStrFn = *lib.get(b"parakeet_capi_free_string\0")?;
            let last_err: LastErrFn = *lib.get(b"parakeet_capi_last_error\0")?;
            let free: FreeFn = *lib.get(b"parakeet_capi_free\0")?;
            let stream_begin: StreamBeginFn = *lib.get(b"parakeet_capi_stream_begin_lang\0")?;
            let stream_feed: StreamFeedFn = *lib.get(b"parakeet_capi_stream_feed_json\0")?;
            let stream_finalize: StreamFinalizeFn = *lib.get(b"parakeet_capi_stream_finalize_json\0")?;
            let stream_free: FreeFn = *lib.get(b"parakeet_capi_stream_free\0")?;
            // This pinned clang build exports pk::shutdown_backend(), but the C API does not.
            let shutdown_backend: ShutdownBackendFn = *lib.get(b"_ZN2pk16shutdown_backendEv\0")?;
            let path = CString::new(model_path.to_string_lossy().as_bytes())?;
            let ctx = load(path.as_ptr());
            if ctx.is_null() {
                let error = CStr::from_ptr(load_err()).to_string_lossy().into_owned();
                if *live_contexts == 0 {
                    shutdown_backend();
                }
                bail!("parakeet load {}: {error}", model_path.display());
            }
            *live_contexts += 1;
            Ok(Self {
                batch_json,
                free_str,
                last_err,
                free,
                stream_begin,
                stream_feed,
                stream_finalize,
                stream_free,
                shutdown_backend,
                streams: Vec::new(),
                set_decode_options: lib.get::<unsafe extern "C" fn(*const c_char, f32, f32)>(b"parakeet_capi_set_decode_options\0").ok().map(|f| *f),
                set_suppressed: lib.get::<unsafe extern "C" fn(*const c_int, c_int)>(b"parakeet_capi_set_suppressed_tokens\0").ok().map(|f| *f),
                stream_ctc_rows: lib.get::<unsafe extern "C" fn(*mut c_void, *mut c_int, *mut c_int) -> *const f32>(b"parakeet_capi_stream_ctc_rows\0").ok().map(|f| *f),
                ctc: (Vec::new(), 0),
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
        self.keep_ctc(self.streams[source]);
        self.stream_out(p)
    }

    /// Copies the stream's CTC rows from its last feed/finalize (empty unless decoding with CTC).
    fn keep_ctc(&mut self, st: *mut c_void) {
        self.ctc.0.clear();
        self.ctc.1 = 0;
        let Some(f) = self.stream_ctc_rows else { return };
        let (mut frames, mut classes) = (0, 0);
        let p = unsafe { f(st, &mut frames, &mut classes) };
        if !p.is_null() && frames > 0 && classes > 0 {
            self.ctc.0.extend_from_slice(unsafe { std::slice::from_raw_parts(p, (frames * classes) as usize) });
            self.ctc.1 = classes as usize;
        }
    }

    /// End of an utterance: flush the audio the model is still holding (so the last word shows
    /// now, not when someone speaks again) and close the session; the next feed starts a new one.
    pub fn stream_finish(&mut self, source: usize) -> Result<(String, Vec<Word>)> {
        let Some(&st) = self.streams.get(source).filter(|s| !s.is_null()) else { return Ok(Default::default()) };
        let p = unsafe { (self.stream_finalize)(st) };
        self.keep_ctc(st);
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
    /// Decoder for hybrid *_ctc models ("ctc" or "rnnt") and blank penalties, process-wide in
    /// the library; affects streams begun afterwards. Needs our parakeet.cpp fork.
    pub fn set_decoding(&mut self, stream_decoder: &str, rnnt_blank_penalty: f32, ctc_blank_penalty: f32) -> Result<()> {
        let Some(f) = self.set_decode_options else {
            bail!("this libparakeet has no decode options (build it from latentsig/parakeet.cpp, branch rt-captions)");
        };
        let dec = CString::new(stream_decoder)?;
        unsafe { f(dec.as_ptr(), rnnt_blank_penalty, ctc_blank_penalty) };
        Ok(())
    }

    /// Token ids the decoder may never emit (empty = no restriction). Needs our fork.
    pub fn set_suppressed(&mut self, ids: &[i32]) {
        if let Some(f) = self.set_suppressed {
            unsafe { f(ids.as_ptr(), ids.len() as c_int) };
        }
    }

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

/// A loaded non-ASR model (diarization or speaker encoder): library handle, context, and the
/// shared live-context bookkeeping. Drop order matters: owners free streams first.
struct Ctx {
    _lib: Library,
    ptr: *mut c_void,
    last_err: LastErrFn,
    free: FreeFn,
    shutdown_backend: ShutdownBackendFn,
}

impl Ctx {
    fn load(model_path: &Path) -> Result<Self> {
        if !cfg!(target_os = "macos") {
            std::env::set_var("PARAKEET_DEVICE", "cpu");
        }
        unsafe {
            let mut live = LIVE_CONTEXTS.lock().unwrap_or_else(|e| e.into_inner());
            let lib = load_library(&find_lib()?)?;
            let load: LoadFn = *lib.get(b"parakeet_capi_load\0")?;
            let load_err: LoadErrFn = *lib.get(b"parakeet_capi_load_error\0")?;
            let last_err: LastErrFn = *lib.get(b"parakeet_capi_last_error\0")?;
            let free: FreeFn = *lib.get(b"parakeet_capi_free\0")?;
            let shutdown_backend: ShutdownBackendFn = *lib.get(b"_ZN2pk16shutdown_backendEv\0")?;
            let path = CString::new(model_path.to_string_lossy().as_bytes())?;
            let ptr = load(path.as_ptr());
            if ptr.is_null() {
                let error = CStr::from_ptr(load_err()).to_string_lossy().into_owned();
                if *live == 0 {
                    shutdown_backend();
                }
                bail!("parakeet load {}: {error}", model_path.display());
            }
            *live += 1;
            Ok(Self { _lib: lib, ptr, last_err, free, shutdown_backend })
        }
    }

    fn error(&self) -> String {
        unsafe { CStr::from_ptr((self.last_err)(self.ptr)) }.to_string_lossy().into_owned()
    }
}

impl Drop for Ctx {
    fn drop(&mut self) {
        let mut live = LIVE_CONTEXTS.lock().unwrap_or_else(|e| e.into_inner());
        unsafe { (self.free)(self.ptr) };
        *live -= 1;
        if *live == 0 {
            unsafe { (self.shutdown_backend)() };
        }
    }
}

/// A stretch of one speaker, seconds from the stream start.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Segment {
    pub speaker: usize,
    pub start: f64,
    pub end: f64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RawSeg {
    speaker: c_int,
    start: f32,
    end: f32,
}

type DiarBeginFn = unsafe extern "C" fn(*mut c_void, c_int) -> *mut c_void;
type DiarFeedFn = unsafe extern "C" fn(*mut c_void, *const f32, c_int, c_int, *mut *mut RawSeg, *mut c_int) -> c_int;
type DiarActiveFn = unsafe extern "C" fn(*mut c_void, *mut *mut RawSeg, *mut c_int) -> c_int;
type DiarTimeFn = unsafe extern "C" fn(*mut c_void) -> f32;
type FreeSegsFn = unsafe extern "C" fn(*mut RawSeg);

const DIAR_LATENCY_ULTRA_LOW: c_int = 3; // 0.32 s chunks
pub const DIAR_LATENCY_DEFAULT: c_int = DIAR_LATENCY_ULTRA_LOW;

/// Streaming diarization of one 16 kHz source: who is speaking, with stable speaker indices.
pub struct Diarizer {
    stream: *mut c_void,
    feed_fn: DiarFeedFn,
    active_fn: DiarActiveFn,
    time_fn: DiarTimeFn,
    free_segs: FreeSegsFn,
    stream_free: FreeFn,
    ctx: Ctx,
}

unsafe impl Send for Diarizer {}

impl Diarizer {
    pub fn load(model_path: &Path, latency: c_int) -> Result<Self> {
        let ctx = Ctx::load(model_path)?;
        unsafe {
            let lib = &ctx._lib;
            let begin: DiarBeginFn = *lib.get(b"parakeet_capi_diarize_stream_begin_latency\0")?;
            let feed_fn: DiarFeedFn = *lib.get(b"parakeet_capi_diarize_stream_feed\0")?;
            let active_fn: DiarActiveFn = *lib.get(b"parakeet_capi_diarize_stream_active\0")?;
            let time_fn: DiarTimeFn = *lib.get(b"parakeet_capi_diarize_stream_time\0")?;
            let free_segs: FreeSegsFn = *lib.get(b"parakeet_capi_free_diar_segments\0")?;
            let stream_free: FreeFn = *lib.get(b"parakeet_capi_diarize_stream_free\0")?;
            let stream = begin(ctx.ptr, latency);
            if stream.is_null() {
                bail!("diarization stream: {}", ctx.error());
            }
            Ok(Self { stream, feed_fn, active_fn, time_fn, free_segs, stream_free, ctx })
        }
    }

    fn take(&self, p: *mut RawSeg, n: c_int) -> Vec<Segment> {
        let out = if p.is_null() { Vec::new() } else {
            unsafe { std::slice::from_raw_parts(p, n.max(0) as usize) }.iter().filter(|s| s.speaker >= 0)
                .map(|s| Segment { speaker: s.speaker as usize, start: s.start as f64, end: s.end as f64 }).collect()
        };
        if !p.is_null() {
            unsafe { (self.free_segs)(p) };
        }
        out
    }

    /// Feed 16 kHz audio; returns the segments that ended since the last call.
    pub fn feed(&mut self, pcm: &[f32]) -> Result<Vec<Segment>> {
        let (mut p, mut n) = (std::ptr::null_mut(), 0);
        let rc = unsafe { (self.feed_fn)(self.stream, pcm.as_ptr(), pcm.len() as c_int, 0, &mut p, &mut n) };
        if rc != 0 {
            bail!("diarization: {}", self.ctx.error());
        }
        Ok(self.take(p, n))
    }

    /// Segments still open ("who is speaking now"), ending at `time()`.
    pub fn active(&mut self) -> Vec<Segment> {
        let (mut p, mut n) = (std::ptr::null_mut(), 0);
        if unsafe { (self.active_fn)(self.stream, &mut p, &mut n) } != 0 {
            return Vec::new();
        }
        self.take(p, n)
    }

    /// Seconds of audio diarized so far (trails the audio fed by up to the 0.32 s latency).
    pub fn time(&self) -> f64 {
        unsafe { (self.time_fn)(self.stream) as f64 }
    }
}

impl Drop for Diarizer {
    fn drop(&mut self) {
        unsafe { (self.stream_free)(self.stream) };
    }
}

type EmbedFn = unsafe extern "C" fn(*mut c_void, *const f32, c_int, c_int, *mut *mut f32, *mut c_int) -> c_int;
type FreeFloatsFn = unsafe extern "C" fn(*mut f32);
type DimFn = unsafe extern "C" fn(*const c_void) -> c_int;

/// Speaker encoder: audio in, L2-normalized voice vector out (cosine = dot product).
pub struct Encoder {
    embed_fn: EmbedFn,
    free_floats: FreeFloatsFn,
    dim_fn: DimFn,
    ctx: Ctx,
}

unsafe impl Send for Encoder {}

impl Encoder {
    pub fn load(model_path: &Path) -> Result<Self> {
        let ctx = Ctx::load(model_path)?;
        unsafe {
            let lib = &ctx._lib;
            let embed_fn: EmbedFn = *lib.get(b"parakeet_capi_speaker_embed_pcm\0")?;
            let free_floats: FreeFloatsFn = *lib.get(b"parakeet_capi_free_floats\0")?;
            let dim_fn: DimFn = *lib.get(b"parakeet_capi_speaker_dim\0")?;
            Ok(Self { embed_fn, free_floats, dim_fn, ctx })
        }
    }

    /// Vector length.
    pub fn dim(&self) -> usize {
        unsafe { (self.dim_fn)(self.ctx.ptr) }.max(0) as usize
    }

    /// Voice vector of 16 kHz mono audio (>= ~2 s of one speaker works best).
    pub fn embed(&mut self, pcm: &[f32]) -> Result<Vec<f32>> {
        let (mut p, mut n) = (std::ptr::null_mut(), 0);
        let rc = unsafe { (self.embed_fn)(self.ctx.ptr, pcm.as_ptr(), pcm.len() as c_int, SAMPLE_RATE as c_int, &mut p, &mut n) };
        if rc != 0 || p.is_null() {
            bail!("speaker embed: {}", self.ctx.error());
        }
        let v = unsafe { std::slice::from_raw_parts(p, n.max(0) as usize) }.to_vec();
        unsafe { (self.free_floats)(p) };
        Ok(v)
    }
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
        let mut live_contexts = LIVE_CONTEXTS.lock().unwrap_or_else(|e| e.into_inner());
        self.reset_streams();
        unsafe { (self.free)(self.ctx) };
        debug_assert!(*live_contexts > 0);
        *live_contexts -= 1;
        if *live_contexts == 0 {
            unsafe { (self.shutdown_backend)() };
        }
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

fn engine_platform() -> &'static str {
    if cfg!(windows) { "windows" } else if cfg!(target_os = "macos") { "macos" } else { "linux" }
}

/// The bundled libparakeet (next to the exe, or Contents/Resources on macOS); PARAKEET_LIB overrides.
pub fn find_lib() -> Result<PathBuf> {
    let exe = std::env::current_exe()?.parent().context("exe dir")?.to_path_buf();
    std::env::var_os("PARAKEET_LIB").map(PathBuf::from).into_iter()
        .chain([
            exe.join(lib_name()),
            exe.join("engine").join(lib_name()),
            exe.join("../Resources/engine").join(lib_name()),
            exe.join("../../engine").join(engine_platform()).join(lib_name()),
        ])
        .find(|p| p.exists())
        .ok_or_else(|| anyhow!("{} not found next to the app", lib_name()))
}
