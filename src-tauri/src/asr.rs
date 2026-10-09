//! The speech engine behind the captions (parakeet.cpp); see engines.rs for the catalogue.

use crate::engines::{self, Kind};
use crate::parakeet::{self, Parakeet};
use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

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
        let non_latin = if promptless { crate::gguf::non_latin_tokens(&model).unwrap_or_default() } else { Vec::new() };
        Ok(Self { model: Parakeet::load(&parakeet::find_lib()?, &model, threads)?, kind, hybrid, non_latin })
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
        if let Err(e) = self.model.set_decoding(dec, rnnt_pen, ctc_pen) {
            eprintln!("{e:#}");
        }
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
        self.model.stream_feed(source, pcm)
    }

    /// Streaming engines: flush and close one source's session at the end of an utterance.
    pub fn stream_finish(&mut self, source: usize) -> Result<(String, Vec<Word>)> {
        if self.kind != Kind::Streaming {
            return Ok(Default::default());
        }
        self.model.stream_finish(source)
    }

    pub fn reset_streams(&mut self) {
        self.model.reset_streams();
    }
}
