//! The speech engine behind the captions: see engines.rs for the catalogue.

use crate::engines::{self, Kind};
use crate::parakeet::{self, Parakeet};
use crate::whistle::{self, Whistle, Word};
use anyhow::{bail, Result};

pub enum Asr {
    Whistle(Whistle),
    Parakeet(Parakeet, Kind),
}

impl Asr {
    /// Loads an installed engine; fails with a clear message if its model isn't downloaded.
    pub fn load(id: &str, threads: usize) -> Result<Self> {
        let e = engines::get(id)?;
        let threads = if threads > 0 { threads } else { 4 };
        let Some(model) = engines::installed(e) else {
            bail!("{} is not downloaded yet (run: rt-captions download {})", e.name, e.id);
        };
        Ok(match e.id {
            "whistle" => Asr::Whistle(Whistle::load(&whistle::find_lib()?, &model)?),
            _ => Asr::Parakeet(Parakeet::load(&parakeet::find_lib()?, &model, threads)?, e.kind),
        })
    }

    pub fn kind(&self) -> Kind {
        match self {
            Asr::Whistle(_) => Kind::Rolling,
            Asr::Parakeet(_, k) => *k,
        }
    }

    pub fn set_keywords(&mut self, keywords: &str) {
        // ponytail: only Whistle has keyword biasing; parakeet.cpp's C API has none.
        if let Asr::Whistle(w) = self {
            w.set_keywords(keywords);
        }
    }

    /// Language prompt for streaming engines that have one (Nemotron); others ignore it.
    pub fn set_language(&mut self, lang: &str) {
        if let Asr::Parakeet(p, _) = self {
            p.set_language(lang);
        }
    }

    /// Rolling engines: transcribe a whole window.
    pub fn transcribe(&mut self, samples: &[f32]) -> Result<Vec<Word>> {
        match self {
            Asr::Whistle(w) => w.transcribe(samples),
            Asr::Parakeet(p, _) => p.transcribe(samples),
        }
    }

    /// Streaming engines: feed the next piece of one source's audio.
    pub fn stream_feed(&mut self, source: usize, pcm: &[f32]) -> Result<(String, Vec<Word>)> {
        match self {
            Asr::Parakeet(p, Kind::Streaming) => p.stream_feed(source, pcm),
            _ => bail!("this engine does not stream"),
        }
    }

    /// Streaming engines: flush and close one source's session at the end of an utterance.
    pub fn stream_finish(&mut self, source: usize) -> Result<(String, Vec<Word>)> {
        match self {
            Asr::Parakeet(p, Kind::Streaming) => p.stream_finish(source),
            _ => Ok(Default::default()),
        }
    }

    pub fn reset_streams(&mut self) {
        if let Asr::Parakeet(p, _) = self {
            p.reset_streams();
        }
    }
}
