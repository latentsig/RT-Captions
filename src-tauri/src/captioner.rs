//! Rolling-window streaming on top of a full-clip transcriber.
//!
//! Every `hop_s` of speech the last `window_s` of audio is re-transcribed. A word is
//! committed when two consecutive passes agree on it (LocalAgreement-2) and it ends at
//! least `margin_s` before the window edge, so a word cut by the edge is never committed
//! half-heard. When the window rolls, the cut snaps back to the start of the word that
//! straddles it, so that word is re-read whole. An energy gate skips inference in silence.

use crate::whistle::{Word, SAMPLE_RATE};
use serde::{Deserialize, Serialize};

const SR: f64 = SAMPLE_RATE as f64;
const PREROLL_S: f64 = 0.3;
const FRAME: usize = 480; // 30 ms

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(default)]
pub struct StreamConfig {
    pub window_s: f64,
    pub hop_s: f64,
    pub margin_s: f64,
    /// Trailing silence that ends an utterance and commits everything.
    pub silence_s: f64,
    /// A pending tail unchanged this long is committed even without a following word.
    pub finalize_s: f64,
    /// RMS level that counts as speech.
    // ponytail: fixed energy gate; swap for Silero VAD if noisy rooms trigger it constantly
    pub vad_threshold: f64,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self { window_s: 3.0, hop_s: 0.25, margin_s: 0.3, silence_s: 0.6, finalize_s: 2.0, vad_threshold: 0.008 }
    }
}

#[derive(Serialize, Clone, Debug, Default)]
pub struct Update {
    /// Newly committed words, absolute times from stream start.
    pub committed: Vec<Word>,
    /// Current unconfirmed tail.
    pub pending: String,
    pub end_of_utterance: bool,
}

pub struct Captioner {
    cfg: StreamConfig,
    buf: Vec<f32>,
    buf_start: usize, // absolute sample index of buf[0]
    since_pass: usize,
    silent: usize,
    active: bool,
    committed_until: f64,
    last_committed: String,
    prev: Vec<Word>,
    /// Stream time since which `prev` has read the same.
    prev_since: f64,
}

fn sec(n: usize) -> f64 {
    n as f64 / SR
}

fn norm(w: &str) -> String {
    w.chars().filter(|c| c.is_alphanumeric()).flat_map(char::to_lowercase).collect()
}

pub fn join(words: &[Word]) -> String {
    words.iter().map(|w| w.word.as_str()).collect::<Vec<_>>().join(" ")
}

impl Captioner {
    pub fn new(cfg: StreamConfig) -> Self {
        Self { cfg, buf: Vec::new(), buf_start: 0, since_pass: 0, silent: 0, active: false, committed_until: 0.0, last_committed: String::new(), prev: Vec::new(), prev_since: 0.0 }
    }

    fn buf_end(&self) -> f64 {
        sec(self.buf_start + self.buf.len())
    }

    fn drop_front(&mut self, n: usize) {
        let n = n.min(self.buf.len());
        self.buf.drain(..n);
        self.buf_start += n;
    }

    /// Append 16 kHz mono audio, judging speech by loudness. Cheap; never runs the model.
    pub fn feed(&mut self, samples: &[f32]) {
        for frame in samples.chunks(FRAME) {
            let rms = (frame.iter().map(|x| x * x).sum::<f32>() / frame.len() as f32).sqrt();
            self.feed_with(frame, rms as f64 >= self.cfg.vad_threshold);
        }
    }

    /// Append audio whose speech/no-speech verdict comes from a real VAD (Silero).
    pub fn feed_with(&mut self, samples: &[f32], speech: bool) {
        if speech {
            self.active = true;
            self.silent = 0;
        } else {
            self.silent += samples.len();
        }
        self.buf.extend_from_slice(samples);
        if self.active {
            self.since_pass += samples.len();
        } else {
            let keep = (PREROLL_S * SR) as usize;
            self.drop_front(self.buf.len().saturating_sub(keep));
        }
    }

    /// Hearing speech right now (inside an utterance).
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Forget the current utterance without committing it.
    pub fn discard(&mut self) {
        self.active = false;
        self.prev.clear();
        self.silent = 0;
        self.since_pass = 0;
        self.drop_front(self.buf.len());
    }

    fn remember(&mut self, committed: &[Word]) {
        if let Some(w) = committed.last() {
            self.committed_until = w.end;
            self.last_committed = w.word.clone();
        }
    }

    /// Run a pass if one is due. `transcribe` gets the window and returns clip-relative words.
    pub fn step(&mut self, transcribe: impl FnOnce(&[f32]) -> Vec<Word>) -> Option<Update> {
        if !self.active {
            return None;
        }
        let end_of_utterance = sec(self.silent) >= self.cfg.silence_s;
        if !end_of_utterance && sec(self.since_pass) < self.cfg.hop_s {
            return None;
        }
        self.since_pass = 0;
        let offset = sec(self.buf_start);
        let (committed_until, last) = (self.committed_until, norm(&self.last_committed));
        let mut words: Vec<Word> = transcribe(&self.buf)
            .into_iter()
            .map(|mut w| {
                w.start += offset;
                w.end += offset;
                w
            })
            .filter(|w| (w.start + w.end) / 2.0 > committed_until)
            .collect();
        // Timestamps drift a little between passes; drop a re-heard copy of the last committed word.
        if words.first().is_some_and(|w| norm(&w.word) == last && w.start < committed_until) {
            words.remove(0);
        }

        if end_of_utterance {
            self.remember(&words);
            self.prev.clear();
            self.active = false;
            self.silent = 0;
            self.drop_front(self.buf.len());
            return Some(Update { committed: words, pending: String::new(), end_of_utterance: true });
        }

        // Commit only words the model has heard past (it knows the next word), so punctuation
        // is not the sentence-final guess a window edge produces.
        let safe = self.buf_end() - self.cfg.margin_s;
        let mut n = words[..words.len().saturating_sub(1)]
            .iter()
            .zip(&self.prev)
            .take_while(|(a, b)| norm(&a.word) == norm(&b.word) && a.end <= safe)
            .count();

        let cut = self.buf_end() - self.cfg.window_s;
        if cut > offset {
            // Words wholly before the cut would fall out of view: commit them now.
            while n < words.len() && words[n].end <= cut {
                n += 1;
            }
            // Keep a straddling word whole by cutting at its start instead.
            let start = words.get(n).map_or(cut, |w| w.start.min(cut)).max(offset);
            self.drop_front(((start - offset) * SR) as usize);
        }

        let tail = &words[n..];
        let same = tail.len() == self.prev.len() && tail.iter().zip(&self.prev).all(|(a, b)| norm(&a.word) == norm(&b.word));
        if !same {
            self.prev_since = self.buf_end();
        } else if !tail.is_empty() && self.buf_end() - self.prev_since >= self.cfg.finalize_s {
            n = words.len(); // stable for long enough: nothing new is coming to confirm it
        }
        let committed = words[..n].to_vec();
        self.remember(&committed);
        self.prev = words[n..].to_vec();
        Some(Update { committed, pending: join(&self.prev), end_of_utterance: false })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Fake audio: sample i holds 1.0 + i*1e-6, so the fake model can recover absolute time.
    fn fake_model<'a>(script: &'a [(&'a str, f64, f64)]) -> impl Fn(&[f32]) -> Vec<Word> + 'a {
        move |buf: &[f32]| {
            let start = sec(((buf[0] - 1.0) * 1e6).round() as usize);
            let end = start + sec(buf.len());
            script
                .iter()
                // a word is "heard" if it is fully inside, or mostly inside at the left edge
                .filter(|(_, s, e)| *e <= end && (s + e) / 2.0 > start)
                .map(|(w, s, e)| Word { word: w.to_string(), start: s - start, end: e - start })
                .collect()
        }
    }

    fn run(window_s: f64) -> (Vec<String>, f64) {
        let script: Vec<(String, f64, f64)> =
            (0..60).map(|i| (format!("w{i}"), 0.1 + i as f64 * 0.31, 0.1 + i as f64 * 0.31 + 0.28)).collect();
        let script_ref: Vec<(&str, f64, f64)> = script.iter().map(|(w, s, e)| (w.as_str(), *s, *e)).collect();
        let model = fake_model(&script_ref);
        let mut c = Captioner::new(StreamConfig { window_s, ..Default::default() });
        let total = 20 * SAMPLE_RATE;
        let mut out = Vec::new();
        let mut worst_lag: f64 = 0.0;
        for chunk in (0..total).step_by(FRAME) {
            let samples: Vec<f32> = (chunk..chunk + FRAME).map(|i| 1.0 + i as f32 * 1e-6).collect();
            c.feed(&samples);
            if let Some(u) = c.step(&model) {
                for w in u.committed {
                    worst_lag = worst_lag.max(sec(chunk + FRAME) - w.end);
                    out.push(w.word);
                }
            }
        }
        c.feed(&vec![0.0; SAMPLE_RATE]); // silence flushes the tail
        while let Some(u) = c.step(&model) {
            out.extend(u.committed.into_iter().map(|w| w.word));
        }
        (out, worst_lag)
    }

    #[test]
    fn every_word_once_in_order() {
        for window in [3.0, 5.0] {
            let (out, lag) = run(window);
            let want: Vec<String> = (0..60).map(|i| format!("w{i}")).collect();
            assert_eq!(out, want, "window {window}");
            assert!(lag < 1.2, "window {window}: commit lag {lag:.2}s");
        }
    }

    #[test]
    fn stable_tail_commits_without_silence() {
        // model keeps hearing one trailing word over steady noise (no silence, no next word)
        let script = [("end.", 0.2, 0.6)];
        let model = fake_model(&script);
        let mut c = Captioner::new(StreamConfig::default());
        let mut committed = Vec::new();
        for chunk in (0..4 * SAMPLE_RATE).step_by(FRAME) {
            let samples: Vec<f32> = (chunk..chunk + FRAME).map(|i| 1.0 + i as f32 * 1e-6).collect();
            c.feed(&samples);
            if let Some(u) = c.step(&model) {
                committed.extend(u.committed.into_iter().map(|w| (w.word, sec(chunk))));
            }
        }
        assert_eq!(committed.len(), 1, "{committed:?}");
        assert!(committed[0].1 < 3.0, "committed at {:.2}s", committed[0].1);
    }

    #[test]
    fn silence_runs_no_passes() {
        let mut c = Captioner::new(StreamConfig::default());
        let mut calls = 0;
        for _ in 0..100 {
            c.feed(&[0.0; FRAME]);
            c.step(|_| {
                calls += 1;
                vec![]
            });
        }
        assert_eq!(calls, 0);
        assert!(c.buf.len() <= (PREROLL_S * SR) as usize);
    }
}
