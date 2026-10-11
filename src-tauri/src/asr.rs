//! The speech engine behind the captions (parakeet.cpp); see engines.rs for the catalogue.

use crate::engines::{self, Kind};
use crate::hotwords::{Decoder, Hotwords, Vocab};
use crate::parakeet::{self, Parakeet};
use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Every model takes 16 kHz mono audio.
pub const SAMPLE_RATE: usize = 16_000;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Word {
    pub word: String,
    pub start: f64,
    pub end: f64,
}

pub struct Asr {
    model: Parakeet,
    kind: Kind,
    /// Hybrid RNN-T/CTC model (our own models): the decoder knobs apply only to these.
    hybrid: bool,
    /// Our English/Hindi models have no language prompt, so English mode locks them to Latin
    /// script by forbidding these tokens (Devanagari pieces and non-ASCII bytes).
    non_latin: Vec<i32>,
    /// Tokenizer text for host-side decoding (hybrid models).
    vocab: Option<Arc<Vocab>>,
    /// Beam search / hotwords over the CTC head (hotwords.rs): width (0 = off) and list.
    beam: usize,
    hot: Arc<Hotwords>,
    ctc: bool,
    /// One decoder per source while host-side decoding is on.
    decoders: Vec<Option<Decoder>>,
    pending: Vec<String>,
    /// Look-aheads (ms) the streaming model was trained for, and its encoder frame (ms).
    lookaheads: Vec<u32>,
    frame_ms: u32,
}

impl Asr {
    /// Loads an installed engine; fails with a clear message if its model isn't downloaded.
    /// `id`: a catalogue engine, or a local model file in the models folder (see engines::list).
    pub fn load(id: &str, threads: usize) -> Result<Self> {
        let (model, kind, _) = engines::resolve(id)?;
        let threads = if threads > 0 { threads } else { 4 };
        let meta = crate::gguf::read(&model).ok();
        let hybrid = meta.as_ref().is_some_and(|m| m.ctc);
        // Prompt models (Nemotron) take the language directly; promptless ones get a script lock.
        let promptless = meta.as_ref().is_some_and(|m| m.languages.is_empty());
        let pieces = if hybrid || promptless { crate::gguf::pieces(&model).unwrap_or_default() } else { Vec::new() };
        let non_latin = if promptless { crate::gguf::non_latin_tokens(&pieces) } else { Vec::new() };
        let vocab = (hybrid && !pieces.is_empty()).then(|| Arc::new(Vocab::new(&pieces)));
        let (lookaheads, frame_ms) = meta.as_ref().map_or((Vec::new(), 80), |m| (m.lookaheads_ms.clone(), m.frame_ms.max(1)));
        Ok(Self {
            model: Parakeet::load(&parakeet::find_lib()?, &model, threads)?, kind, hybrid, non_latin, vocab,
            beam: 0, hot: Arc::new(Hotwords::new(&[])), ctc: false, decoders: Vec::new(), pending: Vec::new(),
            lookaheads, frame_ms,
        })
    }

    pub fn hybrid(&self) -> bool {
        self.hybrid
    }

    pub fn kind(&self) -> Kind {
        self.kind
    }

    /// Language prompt for streaming models that have one (Nemotron); others ignore it.
    /// English: a prompt model is prompted "en"; a promptless English/Hindi model is locked to
    /// Latin script (it otherwise writes accented English in Devanagari, as its Hindi training
    /// transcripts do for loanwords). Hindi: prompt "hi" / no lock (Hindi and Hinglish).
    pub fn set_language(&mut self, lang: &str) {
        self.model.set_language(lang);
        let lock: &[i32] = if lang == "hi" { &[] } else { &self.non_latin };
        self.model.set_suppressed(lock);
    }

    /// Decoder knobs for our parakeet.cpp patch (latentsig/parakeet.cpp, branch rt-captions);
    /// call before `reset_streams`. Only hybrid *_ctc models have a CTC head to switch to.
    /// Each model family gets its own blank penalty (the best value differs: 0.5 over-inserts less
    /// on our hybrids, 1.0 suits Nemotron). Offline engines (Ultra, Redux) are left untouched.
    pub fn set_decoding(&mut self, decoder: &str, blank_penalty: f64, ctc_blank_penalty: f64, nemotron_blank_penalty: f64) {
        let (dec, rnnt_pen, ctc_pen) = if self.hybrid {
            (if decoder == "ctc" { "ctc" } else { "rnnt" }, blank_penalty as f32, ctc_blank_penalty as f32)
        } else if self.kind == Kind::Streaming {
            ("rnnt", nemotron_blank_penalty as f32, 0.0)
        } else {
            ("rnnt", 0.0, 0.0)
        };
        self.ctc = dec == "ctc";
        if let Err(e) = self.model.set_decoding(dec, rnnt_pen, ctc_pen) {
            eprintln!("{e:#}");
        }
    }

    /// Streaming look-ahead: `ms` when the model was trained for it, else the model's own; call
    /// before `reset_streams`. Returns the look-ahead used when it is not the model's own.
    pub fn set_lookahead(&mut self, ms: u32) -> Option<u32> {
        let ok = self.lookaheads.contains(&ms);
        self.model.set_lookahead(if ok { (ms / self.frame_ms) as i32 } else { -1 });
        ok.then_some(ms)
    }

    /// Beam width (0/1 = greedy) and hotwords for hybrid models decoding with CTC; call before
    /// `reset_streams`. Either one turns on host-side decoding (hotwords.rs).
    pub fn set_hotwords(&mut self, beam: usize, words: &[String]) {
        self.beam = beam;
        self.hot = Arc::new(Hotwords::new(words));
    }

    fn host_decoding(&self) -> bool {
        self.ctc && self.vocab.is_some() && (self.beam > 1 || !self.hot.is_empty())
    }

    /// Text after the last final word that may still change (host-side decoding only).
    pub fn pending(&self, source: usize) -> &str {
        self.pending.get(source).map_or("", String::as_str)
    }

    /// Runs the host-side decoder on the CTC rows of the last feed/finish.
    fn decode(&mut self, source: usize, finish: bool) -> (String, Vec<Word>) {
        while self.decoders.len() <= source {
            self.decoders.push(None);
            self.pending.push(String::new());
        }
        let (vocab, hot, width) = (self.vocab.clone().expect("host decoding needs a vocab"), self.hot.clone(), self.beam);
        let d = self.decoders[source].get_or_insert_with(|| Decoder::new(vocab, hot, width));
        let (rows, classes) = &self.model.ctc;
        let (mut words, pending) = if *classes > 0 { d.feed(rows, *classes) } else { (Vec::new(), String::new()) };
        self.pending[source] = pending;
        if finish {
            words.extend(d.finish());
            self.pending[source].clear();
        }
        let text = words.iter().map(|w| format!(" {}", w.word)).collect();
        (text, words)
    }

    /// Rolling engines: transcribe a whole window.
    pub fn transcribe(&mut self, samples: &[f32]) -> Result<Vec<Word>> {
        self.model.transcribe(samples)
    }

    /// Streaming engines: feed the next piece of one source's audio.
    pub fn stream_feed(&mut self, source: usize, pcm: &[f32]) -> Result<(String, Vec<Word>)> {
        if self.kind != Kind::Streaming {
            bail!("this engine does not stream");
        }
        let out = self.model.stream_feed(source, pcm)?;
        Ok(if self.host_decoding() { self.decode(source, false) } else { out })
    }

    /// Streaming engines: flush and close one source's session at the end of an utterance.
    pub fn stream_finish(&mut self, source: usize) -> Result<(String, Vec<Word>)> {
        if self.kind != Kind::Streaming {
            return Ok(Default::default());
        }
        let out = self.model.stream_finish(source)?;
        Ok(if self.host_decoding() { self.decode(source, true) } else { out })
    }

    pub fn reset_streams(&mut self) {
        self.model.reset_streams();
        self.decoders.clear();
        self.pending.clear();
    }
}
