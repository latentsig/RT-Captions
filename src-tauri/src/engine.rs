//! The one thread that owns the speech model: capture -> dsp -> captioner(s) -> events.
//! "both" runs a captioner per source on the shared engine, so captions keep their source label.

use crate::captioner::{join, Captioner, StreamConfig};
use crate::engines::{self, Kind};
use crate::{asr::Asr, audio, dsp::Dsp, speakers::{LabelerThread, VoiceInfo}};
use serde::{Deserialize, Serialize};
use std::sync::mpsc::{channel, Receiver, RecvTimeoutError, SendError, Sender, TryRecvError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(default)]
pub struct AudioSettings {
    /// "mic" | "system" | "both"
    pub source: String,
    /// Speech engine id from engines.rs (default: Nemotron fine-tuned, 22 layers).
    pub engine: String,
    /// Compute threads for the model; 0 = pick from `mode`.
    pub threads: usize,
    /// "low" (2 threads, least battery) | "balanced" (4) | "performance" (up to 8, <= ~60% of cores).
    pub mode: String,
    /// Spoken language for Nemotron: "en" | "hi" (other engines are English-only).
    pub language: String,
    /// Hybrid RNN-T/CTC models: "rnnt" (default: better on Hindi, Hinglish and conversation) or
    /// "ctc" (cheaper; only better on clean English). Other models use their own decoder.
    pub decoder: String,
    /// Subtracted from the blank score before each decision; > 0 counters dropped words, too high
    /// adds stray ones. Hybrid models (0.5, picked on 17 eval sets):
    pub blank_penalty: f64,
    pub ctc_blank_penalty: f64,
    /// Nemotron (1.0: WER 16.7 -> 12.0% / 22.8 -> 21.5% on two meeting clips vs 0).
    pub nemotron_blank_penalty: f64,
    /// Hybrid models with the CTC decoder: beam search width (0 = greedy) and words to recognise
    /// (names, medical terms). Either one decodes in hotwords.rs; final text then lags ~0.3 s
    /// behind a grey preview of the last word or two.
    pub beam: usize,
    pub hotwords: Vec<String>,
    /// Streaming look-ahead in ms for models trained on several (e.g. 0/80/240/480/1040): how much
    /// future audio a word waits for. Less = captions sooner, a little less accurate. A value the
    /// model was not trained for leaves it at its own.
    pub lookahead_ms: u32,
    pub rnnoise: bool,
    /// Streaming engines: feed only what the loudness detector calls speech. Off (default): feed
    /// everything and let the model decide; with the model on the GPU this costs little and
    /// quiet or distant voices are not lost.
    pub skip_silence: bool,
    /// Label who is speaking on the system audio (needs the diarization model).
    pub speakers: bool,
    /// Cosine a saved voice must reach to name a speaker automatically.
    pub speaker_threshold: f64,
    pub stream: StreamConfig,
}

impl Default for AudioSettings {
    fn default() -> Self {
        Self { source: "both".into(), engine: engines::DEFAULT.into(), threads: 0, mode: "balanced".into(), language: "en".into(), decoder: "rnnt".into(), blank_penalty: 0.5, ctc_blank_penalty: 0.5, nemotron_blank_penalty: 1.0, beam: 0, hotwords: Vec::new(), lookahead_ms: 240, rnnoise: false, skip_silence: false, speakers: true, speaker_threshold: 0.6, stream: StreamConfig::default() }
    }
}

#[derive(Serialize, Clone, Debug)]
pub struct Caption {
    /// "mic" or "system"
    pub source: &'static str,
    pub committed: String,
    pub pending: String,
    pub end_of_utterance: bool,
    pub pass_ms: f64,
    /// Seconds between the newest committed word being spoken and it reaching the screen.
    pub delay_s: Option<f64>,
    /// `committed` is a raw text fragment to append as-is (it carries its own spaces and may end
    /// mid-word), not a list of whole words to join with spaces.
    pub raw: bool,
    /// System input with speaker labels on: the speaker's turn id (a new one each time someone starts talking) and its label.
    pub speaker: Option<usize>,
    pub speaker_name: Option<String>,
}

/// What the engine thread is told: new settings (restart the session) or something for the speaker thread.
pub enum Msg {
    Settings(AudioSettings),
    Assign { key: String, name: String },
    ListVoices(Sender<Vec<VoiceInfo>>),
    VoiceSample { key: String, reply: Sender<Vec<f32>> },
    ReloadPeople,
}

pub enum Event {
    Caption(Caption),
    /// Empty string = all good.
    Status(String),
}

/// Threads for the model: an explicit `threads`, else by mode. Measured on a 4P+4E i5: 4 beat 8
/// (extra threads land on slow cores and spin), so "performance" only helps on chips with more
/// fast cores (e.g. Apple M-series); it is capped at ~60% of logical cores.
pub fn effective_threads(a: &AudioSettings) -> usize {
    if a.threads > 0 {
        return a.threads;
    }
    let cores = std::thread::available_parallelism().map_or(4, |n| n.get());
    match a.mode.as_str() {
        "low" => 2,
        "performance" => (cores * 6 / 10).clamp(4, 8),
        _ => 4.min(cores.max(2)),
    }
}

pub struct EngineHandle {
    sender: Option<Sender<Msg>>,
    worker: Option<JoinHandle<()>>,
}

impl EngineHandle {
    pub fn send(&self, settings: AudioSettings) -> Result<(), SendError<AudioSettings>> {
        match &self.sender {
            Some(sender) => sender.send(Msg::Settings(settings)).map_err(|SendError(m)| match m {
                Msg::Settings(s) => SendError(s),
                _ => unreachable!(),
            }),
            None => Err(SendError(settings)),
        }
    }

    fn tell(&self, msg: Msg) {
        if let Some(sender) = &self.sender {
            let _ = sender.send(msg);
        }
    }

    /// Names a voice ("turn:<id>", "person:<id>" or "voice:<n>") and saves its signatures for next time.
    pub fn assign_voice(&self, key: String, name: String) {
        self.tell(Msg::Assign { key, name });
    }

    /// The people database changed on disk: the speaker thread re-reads it.
    pub fn reload_people(&self) {
        self.tell(Msg::ReloadPeople);
    }

    /// Voices heard in this meeting (empty when there is no session or it does not answer in 500 ms).
    pub fn list_voices(&self) -> Vec<VoiceInfo> {
        let (tx, rx) = channel();
        self.tell(Msg::ListVoices(tx));
        rx.recv_timeout(Duration::from_millis(500)).unwrap_or_default()
    }

    /// Playback sample of a voice (16 kHz mono).
    pub fn voice_sample(&self, key: String) -> Vec<f32> {
        let (tx, rx) = channel();
        self.tell(Msg::VoiceSample { key, reply: tx });
        rx.recv_timeout(Duration::from_millis(500)).unwrap_or_default()
    }

    pub fn shutdown(&mut self) {
        self.sender.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

impl Drop for EngineHandle {
    fn drop(&mut self) {
        self.shutdown();
    }
}

pub fn spawn(emit: impl Fn(Event) + Send + 'static) -> EngineHandle {
    let (tx, rx) = channel::<Msg>();
    let worker = std::thread::Builder::new()
        .name("engine".into())
        .spawn(move || {
            realtime_priority();
            let Some(mut settings) = next_settings(&rx) else { return };
            let mut loaded: Option<(String, usize, Asr)> = None;
            loop {
                let threads = effective_threads(&settings);
                if loaded.as_ref().map_or(true, |(e, t, _)| *e != settings.engine || *t != threads) {
                    loaded = None; // one model in memory at a time
                    match Asr::load(&settings.engine, threads) {
                        Ok(asr) => loaded = Some((settings.engine.clone(), threads, asr)),
                        Err(e) => {
                            emit(Event::Status(format!("Speech engine failed to load: {e:#}")));
                            match next_settings(&rx) {
                                Some(next) => {
                                    settings = next;
                                    continue;
                                }
                                None => return,
                            }
                        }
                    }
                }
                let asr = &mut loaded.as_mut().unwrap().2;
                match session(asr, &settings, &rx, &emit) {
                    Some(next) => settings = next,
                    None => return,
                }
            }
        })
        .expect("engine thread");
    EngineHandle { sender: Some(tx), worker: Some(worker) }
}

/// Blocks for the next settings, ignoring speaker messages (there is no session; their reply channels just close).
fn next_settings(rx: &Receiver<Msg>) -> Option<AudioSettings> {
    loop {
        if let Msg::Settings(s) = rx.recv().ok()? {
            return Some(s);
        }
    }
}

/// Speech gate for streaming engines: silence is never fed to the model (no CPU), a short
/// pre-roll keeps word onsets, and a trailing tail lets the model finish the last words.
struct Gate {
    active: bool,
    silent: usize,
    preroll: Vec<f32>,
}

const PREROLL: usize = 4_800; // 0.3 s
const TAIL: usize = 19_200; // 1.2 s

impl Gate {
    /// `blocks` are (audio, is_speech) pieces. Returns the audio to feed and whether an
    /// utterance just ended.
    fn pass<'a>(&mut self, blocks: impl IntoIterator<Item = (&'a [f32], bool)>, muted: bool) -> (Vec<f32>, bool) {
        let mut out = Vec::new();
        let mut ended = false;
        for (frame, speech) in blocks {
            if speech && !muted {
                if !self.active {
                    self.active = true;
                    out.append(&mut self.preroll);
                }
                self.silent = 0;
                out.extend_from_slice(frame);
            } else if self.active {
                out.extend_from_slice(frame);
                self.silent += frame.len();
                if self.silent >= TAIL {
                    self.active = false;
                    ended = true;
                }
            } else {
                self.preroll.extend_from_slice(frame);
                let extra = self.preroll.len().saturating_sub(PREROLL);
                self.preroll.drain(..extra);
            }
        }
        (out, ended)
    }
}

/// Speech detector that adapts to the room: tracks the background loudness and calls a block
/// speech only when it is clearly louder (3x the floor, and at least the configured threshold)
/// for ~100 ms. A fixed threshold fails in real rooms: fans and hum crossed it 43% of the time.
// ponytail: loudness-based. A learned VAD (Silero) was more exact but cost 1-2 cores idle through
// libparakeet's spinning thread pool; revisit with a single-thread VAD.
struct Detector {
    floor: f64,
    loud_run: u32,
    speech: bool,
}

impl Detector {
    fn new() -> Self {
        Self { floor: -1.0, loud_run: 0, speech: false }
    }

    fn next(&mut self, block: &[f32], min_threshold: f64) -> bool {
        let rms = ((block.iter().map(|x| x * x).sum::<f32>() / block.len().max(1) as f32).sqrt()) as f64;
        if self.floor < 0.0 {
            self.floor = rms; // learn the room from the first block
        }
        let threshold = (self.floor * 3.0).max(min_threshold);
        let loud = rms >= threshold;
        // The floor follows quiet blocks quickly and loud ones slowly (~6 s): constant noise is
        // learned, while speech (which has gaps between syllables) barely moves it.
        let rate = if loud { 0.005 } else { 0.02 };
        self.floor += (rms - self.floor) * rate;
        self.loud_run = if loud { self.loud_run + 1 } else { 0 };
        self.speech = if self.speech { rms >= threshold * 0.7 } else { self.loud_run >= 3 };
        self.speech
    }
}

struct Input {
    name: &'static str,
    detector: Detector,
    /// Samples fed to this source's streaming session (its clock).
    fed: usize,
    /// Samples received since the session started (debug clock: audio time vs wall time).
    heard_total: usize,
    _stream: audio::Capture,
    dsp: Dsp,
    cap: Captioner,
    gate: Gate,
    queued: Vec<f32>,
    last_audio: Instant,
    /// Ungated streaming: when the model last produced text (None = nothing since the last flush).
    heard: Option<Instant>,
    /// Host-side decoding: the pending text last sent (to notice when it changes).
    pending: String,
    /// RT_CAPTIONS_DEBUG: per-second input level and speech/mute counts on stderr.
    probe: Probe,
}

/// RT_CAPTIONS_DEBUG=1: one line per event to stderr; RT_CAPTIONS_DEBUG=<file>: appended there
/// (the GUI app has no terminal).
/// The engine and speaker threads run at the highest non-realtime priority, so a busy Mac (or
/// macOS deprioritising a background app) never makes captions fall behind.
pub fn realtime_priority() {
    #[cfg(target_os = "macos")]
    unsafe {
        extern "C" {
            fn pthread_set_qos_class_self_np(qos: u32, relative_priority: i32) -> i32;
        }
        const QOS_CLASS_USER_INTERACTIVE: u32 = 0x21;
        pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    }
}

pub fn debug_log(line: std::fmt::Arguments) {
    use std::io::Write;
    static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    let t = START.get_or_init(Instant::now).elapsed().as_secs_f64();
    match std::env::var_os("RT_CAPTIONS_DEBUG") {
        Some(v) if v == "1" => eprintln!("[debug {t:.2}] {line}"),
        Some(path) => {
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
                let _ = writeln!(f, "{t:.2} {line}");
            }
        }
        None => {}
    }
}

#[derive(Default)]
struct Probe {
    since: Option<Instant>,
    sum_sq: f64,
    n: usize,
    speech: usize,
    muted: usize,
    blocks: usize,
}

impl Probe {
    fn add(&mut self, name: &str, audio: &[f32], speech: usize, blocks: usize, muted: bool, floor: f64) {
        if std::env::var_os("RT_CAPTIONS_DEBUG").is_none() {
            return;
        }
        let since = *self.since.get_or_insert_with(Instant::now);
        self.sum_sq += audio.iter().map(|x| (*x as f64).powi(2)).sum::<f64>();
        self.n += audio.len();
        self.speech += speech;
        self.blocks += blocks;
        self.muted += if muted { blocks } else { 0 };
        if since.elapsed() >= Duration::from_secs(1) {
            debug_log(format_args!("{name}: {:.1}s audio rms={:.4} floor={:.4} speech={}/{} muted={}",
                      self.n as f64 / 16_000.0, (self.sum_sq / self.n.max(1) as f64).sqrt(), floor, self.speech, self.blocks, self.muted));
            *self = Probe::default();
        }
    }
}

/// Ungated streaming: this long without new text ends an utterance.
const UTTERANCE_GAP: f64 = 1.2;
/// Ungated "both": the mic stays muted this long after the system side last produced text.
const ECHO_HOLD: Duration = Duration::from_millis(800);
const STARVED: Duration = Duration::from_millis(200);
// On the GPU a call is cheap, so work in 20 ms steps; on CPU 100 ms (each call wakes a thread pool).
const BATCH: usize = if cfg!(target_os = "macos") { 320 } else { 1_600 };

/// Runs until new settings arrive (returned) or the app goes away (None).
fn session(asr: &mut Asr, s: &AudioSettings, rx: &Receiver<Msg>, emit: &impl Fn(Event)) -> Option<AudioSettings> {
    asr.set_language(&s.language);
    asr.set_decoding(&s.decoder, s.blank_penalty, s.ctc_blank_penalty, s.nemotron_blank_penalty);
    asr.set_hotwords(s.beam, &s.hotwords);
    if let Some(ms) = asr.set_lookahead(s.lookahead_ms) {
        debug_log(format_args!("look-ahead {ms} ms"));
    }
    asr.reset_streams();
    let streaming = asr.kind() == Kind::Streaming;
    // Never update faster than the engine can keep up with (a slow pass would build a backlog).
    let mut stream_cfg = s.stream.clone();
    if let Ok(e) = engines::get(&s.engine) {
        stream_cfg.hop_s = stream_cfg.hop_s.max(e.min_hop_s);
    }
    let names: &[&'static str] = match s.source.as_str() {
        "both" => &["mic", "system"],
        "system" => &["system"],
        _ => &["mic"],
    };
    let (atx, arx) = channel::<(usize, Vec<f32>)>();
    let mut inputs = Vec::new();
    let mut fallback = false;
    for (i, &name) in names.iter().enumerate() {
        let tx = atx.clone();
        match audio::start(name, move |chunk| {
            let _ = tx.send((i, chunk));
        }) {
            Ok((stream, rate)) => inputs.push(Input {
                name,
                _stream: stream,
                dsp: Dsp::new(rate, s.rnnoise),
                cap: Captioner::new(stream_cfg.clone()),
                gate: Gate { active: false, silent: 0, preroll: Vec::new() },
                detector: Detector::new(),
                fed: 0,
                heard_total: 0,
                queued: Vec::new(),
                last_audio: Instant::now(),
                heard: None,
                pending: String::new(),
                probe: Probe::default(),
            }),
            // ponytail: if system capture fails (e.g. no permission), "both"
            // degrades to the mic instead of failing outright.
            Err(e) if name == "system" && !inputs.is_empty() => {
                debug_log(format_args!("system audio failed: {e:#}"));
                emit(Event::Status(format!("Captioning the microphone only. System audio: {e:#}")));
                fallback = true;
            }
            Err(e) => {
                emit(Event::Status(format!("{name}: {e:#}")));
                return next_settings(rx);
            }
        }
    }
    drop(atx);
    debug_log(format_args!("session: engine={} sources={:?} skip_silence={}", s.engine, inputs.iter().map(|i| i.name).collect::<Vec<_>>(), s.skip_silence));
    if !fallback {
        emit(Event::Status(String::new()));
    }
    // Speaker labels on the other people: the system input in "both" (the mic is you), else the
    // only input (a mic in a room hears everyone). Only when the models are on disk.
    let labeled: &str = if inputs.len() > 1 { "system" } else { inputs[0].name };
    let mut labeler = None;
    // The speaker of the last labeled caption, so a sentence's trailing "." stays with its speaker.
    let mut last_who: Option<(usize, String)> = None;
    if s.speakers && streaming {
        match LabelerThread::spawn(s.speaker_threshold) {
            Ok(l) => labeler = Some(l),
            Err(e) => debug_log(format_args!("speaker labels off: {e:#}")),
        }
    }
    let both = inputs.len() > 1;
    let queue = |inputs: &mut Vec<Input>, (i, chunk): (usize, Vec<f32>)| {
        let input = &mut inputs[i];
        let audio = input.dsp.process(chunk);
        input.queued.extend(audio);
        input.last_audio = Instant::now();
    };
    loop {
        match rx.try_recv() {
            Ok(Msg::Settings(next)) => return Some(next),
            Ok(Msg::Assign { key, name }) => {
                if let Some(l) = labeler.as_ref() {
                    l.assign(key, name);
                }
            }
            Ok(Msg::ReloadPeople) => {
                if let Some(l) = labeler.as_ref() {
                    l.reload_people();
                }
            }
            Ok(Msg::ListVoices(reply)) => {
                if let Some(l) = labeler.as_ref() {
                    l.voices(reply); // answered from the speaker thread, so captions never wait
                }
            }
            Ok(Msg::VoiceSample { key, reply }) => {
                if let Some(l) = labeler.as_ref() {
                    l.sample(key, reply);
                }
            }
            Err(TryRecvError::Disconnected) => return None,
            Err(TryRecvError::Empty) => {}
        }
        match arx.recv_timeout(STARVED) {
            Ok(msg) => queue(&mut inputs, msg),
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => {
                emit(Event::Status("Audio device stopped".into()));
                return next_settings(rx);
            }
        }
        // Drain whatever queued up during the last pass, so we always work on the newest audio.
        while let Ok(msg) = arx.try_recv() {
            queue(&mut inputs, msg);
        }
        for input in inputs.iter_mut() {
            // Loopback sends nothing while nothing plays: that is silence, so let the tail flush.
            if input.last_audio.elapsed() >= STARVED {
                input.queued.extend_from_slice(&[0.0; 3200]);
                input.last_audio = Instant::now();
            }
        }
        // ponytail: echo gate. Without headphones the mic hears the speakers, so while the
        // system side is talking the mic is ignored (which also drops talking over someone).
        // Upgrade: acoustic echo cancellation against the loopback signal.
        let system_talking = both && match (streaming, s.skip_silence) {
            (true, true) => inputs[1].gate.active,
            (false, true) => inputs[1].cap.is_active(),
            (_, false) => inputs[1].heard.is_some_and(|t| t.elapsed() < ECHO_HOLD),
        };
        for (i, input) in inputs.iter_mut().enumerate() {
            // Work in >= 100 ms batches: every model call wakes a pool of compute threads, so
            // fewer, larger calls cost far less CPU (at most 0.1 s of added delay).
            if input.queued.len() < BATCH {
                continue;
            }
            let audio = std::mem::take(&mut input.queued);
            input.heard_total += audio.len();
            // Speech verdict per 32 ms block.
            let blocks: Vec<(Vec<f32>, bool)> =
                audio.chunks(512).map(|b| (b.to_vec(), input.detector.next(b, s.stream.vad_threshold))).collect();
            let muted = both && i == 0 && system_talking;
            input.probe.add(input.name, &audio, blocks.iter().filter(|b| b.1).count(), blocks.len(), muted, input.detector.floor);
            if let (Some(l), true) = (labeler.as_ref(), input.name == labeled) {
                // Ungated: the diarizer hears the same audio as the model, muted or not.
                l.feed(&audio);
            }
            let t = Instant::now();
            let update = if streaming {
                let (feed, gate_ended) = if s.skip_silence {
                    input.gate.pass(blocks.iter().map(|(b, sp)| (b.as_slice(), *sp)), muted)
                } else if muted {
                    (Vec::new(), false)
                } else {
                    (audio, false)
                };
                input.fed += feed.len();
                let (mut text, mut words) = if feed.is_empty() { Default::default() } else {
                    asr.stream_feed(i, &feed).unwrap_or_else(|e| { eprintln!("{e:#}"); Default::default() })
                };
                // Ungated: the model is the speech detector. A pause with no new text ends the
                // utterance, which flushes the last word and keeps each session short.
                if !text.is_empty() {
                    input.heard = Some(Instant::now());
                }
                let quiet = input.heard.is_some_and(|t| t.elapsed().as_secs_f64() >= s.stream.silence_s.max(UTTERANCE_GAP));
                let ended = gate_ended || (!s.skip_silence && quiet);
                if ended {
                    input.heard = None;
                }
                if ended {
                    let (t2, w2) = asr.stream_finish(i).unwrap_or_else(|e| { eprintln!("{e:#}"); Default::default() });
                    text.push_str(&t2);
                    words.extend(w2);
                }
                let delay = words.last().map(|w| input.fed as f64 / 16_000.0 - w.end + t.elapsed().as_secs_f64());
                if ended {
                    input.fed = 0; // the next session's word times start from zero
                }
                // Host-side decoding (beam / hotwords) shows its undecided last word or two as pending.
                let pending = asr.pending(i).to_string();
                let changed = pending != input.pending;
                input.pending.clone_from(&pending);
                (!text.is_empty() || ended || changed).then(|| (text, pending, ended, delay))
            } else {
                if muted {
                    input.cap.discard();
                }
                for (b, sp) in &blocks {
                    // Ungated: every block counts as speech and the model decides what was said.
                    input.cap.feed_with(b, (*sp || !s.skip_silence) && !muted);
                }
                input.cap.step(|buf| asr.transcribe(buf).unwrap_or_else(|e| { eprintln!("{e:#}"); Vec::new() }))
                    .map(|u| (join(&u.committed), u.pending, u.end_of_utterance, None))
            };
            if let Some((committed, pending, end_of_utterance, delay_s)) = update {
                if !streaming && !(committed.is_empty() && pending.is_empty()) {
                    input.heard = Some(Instant::now());
                }
                if std::env::var_os("RT_CAPTIONS_DEBUG").is_some() && !committed.is_empty() {
                    debug_log(format_args!("{} caption at audio {:.2}s: {committed:?}", input.name, input.heard_total as f64 / 16_000.0));
                }
                let who = if input.name == labeled && !committed.trim().is_empty() { labeler.as_ref().and_then(|l| l.current()) } else { None };
                let mut committed = committed;
                if who.is_some() {
                    let changed = last_who.as_ref().is_some_and(|l| Some(l.0) != who.as_ref().map(|w| w.0));
                    let (punct, rest) = split_lead_punct(&committed);
                    if changed && !punct.is_empty() {
                        emit(Event::Caption(Caption { source: input.name, committed: punct.to_owned(), pending: String::new(),
                            end_of_utterance: false, pass_ms: 0.0, delay_s: None, raw: streaming,
                            speaker: last_who.as_ref().map(|w| w.0), speaker_name: last_who.as_ref().map(|w| w.1.clone()) }));
                        committed = rest.to_owned();
                    }
                    if !committed.trim().is_empty() {
                        last_who = who.clone();
                    }
                }
                emit(Event::Caption(Caption {
                    source: input.name,
                    committed,
                    pending,
                    end_of_utterance,
                    pass_ms: t.elapsed().as_secs_f64() * 1000.0,
                    delay_s,
                    raw: streaming,
                    speaker: who.as_ref().map(|w| w.0),
                    speaker_name: who.map(|w| w.1),
                }));
            }
        }
    }
}

/// Leading punctuation of a streamed fragment (". I think" -> (".", " I think")).
fn split_lead_punct(text: &str) -> (&str, &str) {
    let n = text.find(|c: char| !matches!(c, '.' | ',' | '?' | '!' | ';' | ':')).unwrap_or(text.len());
    text.split_at(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leading_punctuation_splits_off() {
        assert_eq!(split_lead_punct(". I think"), (".", " I think"));
        assert_eq!(split_lead_punct(" that"), ("", " that"));
        assert_eq!(split_lead_punct("?"), ("?", ""));
    }

    #[test]
    fn detector_ignores_steady_room_noise_but_hears_speech() {
        let mut d = Detector::new();
        let noise = |amp: f32, i: usize| (0..512).map(|k| if (k + i) % 2 == 0 { amp } else { -amp }).collect::<Vec<f32>>();
        // ~10 s of room noise around 0.012 RMS, above the old fixed 0.008 threshold
        let noisy = (0..300).filter(|&i| d.next(&noise(0.012, i), 0.008)).count();
        assert!(noisy < 10, "room noise counted as speech in {noisy} of 300 blocks");
        let heard = (0..20).filter(|&i| d.next(&noise(0.1, i), 0.008)).count();
        assert!(heard >= 15, "speech heard in only {heard} of 20 blocks");
    }

    #[test]
    fn gate_skips_silence_keeps_onset_and_tail() {
        let mut g = Gate { active: false, silent: 0, preroll: Vec::new() };
        let blocks = |n: usize, speech: bool| vec![(&[0.0f32; 480][..], speech); n];
        let (out, ended) = g.pass(blocks(40, false), false);
        assert!(out.is_empty() && !ended, "silence is never fed");
        let (out, _) = g.pass(blocks(10, true), false);
        assert_eq!(out.len(), PREROLL + 4_800, "speech brings its pre-roll");
        let (out, ended) = g.pass(blocks(TAIL / 480, false), false);
        assert!(ended && out.len() == TAIL, "trailing silence is fed, then the utterance ends");
        let (out, _) = g.pass(blocks(10, true), true);
        assert!(out.is_empty(), "a muted source feeds nothing");
    }

    #[test]
    fn engine_handle_shutdown_disconnects_and_joins_worker() {
        let mut engine = spawn(|_| {});
        engine.shutdown();
        engine.shutdown();
        assert!(engine.send(AudioSettings::default()).is_err());
    }
}
