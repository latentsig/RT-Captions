//! Host-side decoding of a hybrid model's CTC scores (`parakeet_capi_stream_ctc_rows`): prefix
//! beam search, hotword boosting, and a fuzzy spotter that turns near-miss spellings of listed
//! words ("jondice", "jhon dis") into the word itself.
//!
//! Text becomes final once every beam agrees on it; beams that disagree with the best one before
//! its last `TAIL` tokens are dropped, so only the last word or two can still change (shown as
//! pending). Hotwords are matched on text, so any segmentation the model picks earns the bonus;
//! the bonus is refunded when the walk leaves the list (or ends on an incomplete word).

use crate::asr::Word;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Log-prob bonus per matched token (1.75 was safe on the meeting clips: 2.5 caused deletions).
const BONUS: f32 = 1.75;
/// No bonus until this many letters match ("j" / "d" alone would flood the beam).
const MIN_CHARS: usize = 3;
/// Tokens at the end of the best hypothesis that may still change.
const TAIL: usize = 2;
/// Candidate tokens per frame (besides blank).
const TOPK: usize = 16;
/// FastConformer encoder frame: 8 x 10 ms mel frames.
const FRAME_S: f64 = 0.08;
const NEG: f32 = -1e30;

/// Tokenizer text and the words the model has as one whole piece (common words it already knows).
pub struct Vocab {
    text: Vec<String>,
    words: HashSet<String>,
}

impl Vocab {
    pub fn new(pieces: &[String]) -> Self {
        let text = pieces.iter().map(|p| p.replace('\u{2581}', " ")).collect();
        let words = pieces.iter().filter_map(|p| p.strip_prefix('\u{2581}')).filter(|w| !w.is_empty()).map(str::to_lowercase).collect();
        Self { text, words }
    }
}

/// The hotword list: " word" strings (leading space = word start) and all their prefixes.
pub struct Hotwords {
    words: HashSet<String>,
    prefixes: HashSet<String>,
    /// Single words of 5+ letters, for the fuzzy spotter.
    fuzzy: Vec<String>,
}

impl Hotwords {
    pub fn new(list: &[String]) -> Self {
        let words: HashSet<String> = list.iter().map(|w| norm(w)).filter(|w| !w.is_empty()).map(|w| format!(" {w}")).collect();
        let prefixes = words.iter().flat_map(|w| w.char_indices().skip(1).map(|(i, _)| w[..i].to_string()).chain([w.clone()])).collect();
        let fuzzy = words.iter().map(|w| w[1..].to_string()).filter(|w| !w.contains(' ') && w.chars().count() >= 5).collect();
        Self { words, prefixes, fuzzy }
    }

    pub fn is_empty(&self) -> bool {
        self.words.is_empty()
    }

    fn earn(&self, node: &str, piece: &str) -> f32 {
        let n = node.chars().count() - 1;
        if n < MIN_CHARS {
            return 0.0;
        }
        // The first qualifying piece also pays for the pieces before it.
        let before = n - piece.chars().count().min(n);
        BONUS * if before >= MIN_CHARS { 1.0 } else { (n as f32 / 3.0).round().max(1.0) }
    }

    /// Appends one piece to a hypothesis' walk: (node, provisional bonus, settled bonus change).
    fn step(&self, node: &str, earned: f32, piece: &str) -> (String, f32, f32) {
        if self.words.is_empty() {
            return (String::new(), 0.0, 0.0);
        }
        let mut settled = 0.0;
        if !node.is_empty() {
            let cont = format!("{node}{piece}");
            if self.prefixes.contains(&cont) {
                let e = earned + self.earn(&cont, piece);
                return (cont, e, 0.0);
            }
            if piece.starts_with(' ') && self.words.contains(node) {
                settled = earned; // a whole hotword just ended: keep its bonus (else: refund)
            }
        }
        if piece.starts_with(' ') && self.prefixes.contains(piece) {
            return (piece.to_string(), self.earn(piece, piece), settled);
        }
        (String::new(), 0.0, settled)
    }
}

/// Lowercase letters/digits/spaces only (as the scorer normalizes words).
fn norm(w: &str) -> String {
    w.to_lowercase().chars().filter(|c| c.is_alphanumeric() || *c == ' ').collect::<String>().split_whitespace().collect::<Vec<_>>().join(" ")
}

#[derive(Clone)]
struct Hyp {
    pb: f32,
    pnb: f32,
    node: String,
    prov: f32,
    settled: f32,
    /// Frame each token was first emitted at.
    frames: Vec<u32>,
}

impl Hyp {
    fn score(&self) -> f32 {
        lse(self.pb, self.pnb) + self.prov + self.settled
    }
}

fn lse(a: f32, b: f32) -> f32 {
    let (hi, lo) = if a > b { (a, b) } else { (b, a) };
    if lo <= NEG { hi } else { hi + (lo - hi).exp().ln_1p() }
}

/// One stream's decoder: beam -> final tokens -> words -> fuzzy spotter.
pub struct Decoder {
    vocab: Arc<Vocab>,
    hot: Arc<Hotwords>,
    width: usize,
    /// Undecided suffixes (relative to the final tokens) -> hypothesis.
    hyps: HashMap<Vec<i32>, Hyp>,
    last_final: Option<i32>,
    frame: u32,
    /// Final tokens of the word being built, with the frame of its first token.
    word: String,
    word_start: u32,
    word_end: u32,
    /// A misspelt fragment held back one word, in case it joins the next ("jhon" + "dis").
    held: Option<Word>,
}

impl Decoder {
    pub fn new(vocab: Arc<Vocab>, hot: Arc<Hotwords>, width: usize) -> Self {
        let mut hyps = HashMap::new();
        hyps.insert(Vec::new(), Hyp { pb: 0.0, pnb: NEG, node: String::new(), prov: 0.0, settled: 0.0, frames: Vec::new() });
        Self { vocab, hot, width: width.max(1), hyps, last_final: None, frame: 0, word: String::new(), word_start: 0, word_end: 0, held: None }
    }

    /// Decodes `rows` ([frames x classes] log-probs, blank last). Returns the words that became
    /// final and the text that may still change.
    pub fn feed(&mut self, rows: &[f32], classes: usize) -> (Vec<Word>, String) {
        let mut idx: Vec<usize> = Vec::with_capacity(classes);
        for row in rows.chunks_exact(classes) {
            self.step(row, &mut idx);
            self.frame += 1;
        }
        let mut out = Vec::new();
        self.settle(&mut out);
        (out, self.pending())
    }

    /// End of the stream: everything in the best hypothesis becomes final.
    pub fn finish(&mut self) -> Vec<Word> {
        let best = self.best().clone();
        let mut out = Vec::new();
        let h = &self.hyps[&best];
        let toks: Vec<(i32, u32)> = best.iter().copied().zip(h.frames.iter().copied()).collect();
        for (t, f) in toks {
            self.push_token(t, f, &mut out);
        }
        self.end_word(&mut out);
        if let Some(w) = self.held.take() {
            out.push(w);
        }
        *self = Decoder::new(self.vocab.clone(), self.hot.clone(), self.width);
        out
    }

    fn step(&mut self, row: &[f32], idx: &mut Vec<usize>) {
        let blank = row.len() - 1;
        idx.clear();
        idx.extend(0..blank);
        let k = TOPK.min(blank);
        idx.select_nth_unstable_by(k - 1, |&a, &b| row[b].total_cmp(&row[a]));
        let cand = &idx[..k];
        let mut next: HashMap<Vec<i32>, Hyp> = HashMap::with_capacity(self.width * (k + 1));
        for (pfx, h) in &self.hyps {
            let tot = lse(h.pb, h.pnb);
            let last = pfx.last().copied().or(self.last_final);
            let e = next.entry(pfx.clone()).or_insert_with(|| Hyp { pb: NEG, pnb: NEG, ..h.clone() });
            e.pb = lse(e.pb, tot + row[blank]);
            if let Some(l) = last {
                e.pnb = lse(e.pnb, h.pnb + row[l as usize]); // the last token, held over this frame
            }
            for &c in cand {
                let lp = row[c];
                if lp < -30.0 {
                    continue;
                }
                let t = c as i32;
                let (node, prov, d) = self.hot.step(&h.node, h.prov, &self.vocab.text[c]);
                let mut key = pfx.clone();
                key.push(t);
                let e = next.entry(key).or_insert_with(|| {
                    let mut frames = h.frames.clone();
                    frames.push(self.frame);
                    Hyp { pb: NEG, pnb: NEG, node, prov, settled: h.settled + d, frames }
                });
                // A repeat of the last token only starts a new token after a blank.
                let src = if Some(t) == last { h.pb } else { tot };
                e.pnb = lse(e.pnb, src + lp);
            }
        }
        let mut all: Vec<(Vec<i32>, Hyp)> = next.into_iter().collect();
        if all.len() > self.width {
            all.select_nth_unstable_by(self.width - 1, |a, b| b.1.score().total_cmp(&a.1.score()));
            all.truncate(self.width);
        }
        self.hyps = all.into_iter().collect();
    }

    fn best(&self) -> &Vec<i32> {
        self.hyps.iter().max_by(|a, b| a.1.score().total_cmp(&b.1.score())).map(|(k, _)| k).expect("beam is never empty")
    }

    /// Drops beams that disagree with the best one before its last TAIL tokens, then makes the
    /// prefix all beams share final.
    fn settle(&mut self, out: &mut Vec<Word>) {
        let best = self.best().clone();
        let keep = &best[..best.len().saturating_sub(TAIL)];
        self.hyps.retain(|p, _| p.starts_with(keep));
        let mut n = 0;
        while n < best.len() && self.hyps.keys().all(|p| p.len() > n && p[n] == best[n]) {
            n += 1;
        }
        if n == 0 {
            return;
        }
        let frames = self.hyps[&best].frames[..n].to_vec();
        for (&t, &f) in best[..n].iter().zip(&frames) {
            self.push_token(t, f, out);
        }
        self.last_final = Some(best[n - 1]);
        self.hyps = std::mem::take(&mut self.hyps).into_iter()
            .map(|(p, mut h)| { h.frames.drain(..n); (p[n..].to_vec(), h) })
            .collect();
    }

    fn pending(&self) -> String {
        let best = self.best();
        let tail: String = best.iter().map(|&t| self.vocab.text[t as usize].as_str()).collect();
        let held = self.held.as_ref().map(|w| format!(" {}", w.word)).unwrap_or_default();
        format!("{held}{}{tail}", if self.word.is_empty() { String::new() } else { format!(" {}", self.word) })
    }

    fn push_token(&mut self, t: i32, frame: u32, out: &mut Vec<Word>) {
        let vocab = self.vocab.clone();
        let text = &vocab.text[t as usize];
        if text.starts_with('<') && text.ends_with('>') {
            return; // speaker / language tags
        }
        if text.starts_with(' ') {
            self.end_word(out);
            self.word_start = frame;
        }
        self.word.push_str(text.trim_start());
        self.word_end = frame + 1;
    }

    fn end_word(&mut self, out: &mut Vec<Word>) {
        if self.word.is_empty() {
            return;
        }
        let w = Word { word: std::mem::take(&mut self.word), start: self.word_start as f64 * FRAME_S, end: self.word_end as f64 * FRAME_S };
        self.spot(w, out);
    }

    /// Fuzzy spotter: a misspelt word (not one the model knows whole) that is close to a listed
    /// word becomes it; a fragment that doesn't match alone waits one word to try the pair.
    fn spot(&mut self, w: Word, out: &mut Vec<Word>) {
        if let Some(prev) = self.held.take() {
            let pair = format!("{}{}", norm(&prev.word), norm(&w.word));
            if let Some(h) = self.close(&pair) {
                out.push(Word { word: h, start: prev.start, end: w.end });
                return;
            }
            out.push(prev);
        }
        let n = norm(&w.word);
        if self.hot.fuzzy.is_empty() || n.is_empty() || self.known(&n) {
            out.push(w);
        } else if let Some(h) = self.close(&n) {
            out.push(Word { word: h, ..w });
        } else {
            self.held = Some(w);
        }
    }

    fn known(&self, w: &str) -> bool {
        self.vocab.words.contains(w) || self.hot.fuzzy.iter().any(|h| h == w)
    }

    fn close(&self, cand: &str) -> Option<String> {
        self.hot.fuzzy.iter()
            .filter(|h| h.as_str() != cand && close(cand, h))
            .min_by_key(|h| lev(cand, h))
            .cloned()
    }
}

/// Same first letter, consonant-sound keys within one edit (exact for short keys), and at most
/// 0.65 x length letter edits.
fn close(cand: &str, h: &str) -> bool {
    if cand.chars().next() != h.chars().next() {
        return false;
    }
    let (kc, kh) = (skel(cand), skel(h));
    lev(&kc, &kh) <= usize::from(kh.chars().count() >= 4) && lev(cand, h) as f32 <= 0.65 * h.chars().count() as f32
}

/// Consonant-sound key: jaundice / jondice / jhondis / juondis -> "jnds".
fn skel(w: &str) -> String {
    let mut out = String::new();
    for (i, c) in w.chars().enumerate() {
        let c = match c { 'c' | 'z' => 's', 't' => 'd', c => c };
        if (i > 0 && "aeiouyhw".contains(c)) || out.ends_with(c) {
            continue;
        }
        out.push(c);
    }
    out
}

fn lev(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut d: Vec<usize> = (0..=b.len()).collect();
    for (i, x) in a.chars().enumerate() {
        let mut prev = d[0];
        d[0] = i + 1;
        for (j, &y) in b.iter().enumerate() {
            let cur = d[j + 1];
            d[j + 1] = (d[j + 1] + 1).min(d[j] + 1).min(prev + usize::from(x != y));
            prev = cur;
        }
    }
    d[b.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vocab() -> Arc<Vocab> {
        let p = ["\u{2581}ja", "und", "ice", "\u{2581}join", "ed", "\u{2581}the", "\u{2581}is", "<b>"];
        Arc::new(Vocab::new(&p.map(String::from)))
    }

    fn frame(probs: &[(usize, f32)]) -> Vec<f32> {
        let mut r = vec![1e-6f32; 8];
        for &(k, p) in probs {
            r[k] = p;
        }
        let s: f32 = r.iter().sum();
        r.iter().map(|p| (p / s).ln()).collect()
    }

    fn decode(hot: &[&str]) -> String {
        // "ja"+"und"+"ice" against a stronger "join"+"ed"; then "the".
        let rows: Vec<f32> = [
            frame(&[(0, 0.4), (3, 0.6)]), frame(&[(1, 0.4), (4, 0.6)]), frame(&[(2, 0.3), (7, 0.7)]),
            frame(&[(7, 0.9)]), frame(&[(5, 0.9)]), frame(&[(7, 0.9)]),
        ].concat();
        let mut d = Decoder::new(vocab(), Arc::new(Hotwords::new(&hot.iter().map(|s| s.to_string()).collect::<Vec<_>>())), 8);
        let (mut words, _) = d.feed(&rows, 8);
        words.extend(d.finish());
        words.iter().map(|w| w.word.as_str()).collect::<Vec<_>>().join(" ")
    }

    #[test]
    fn beam_follows_the_acoustics_without_hotwords() {
        assert_eq!(decode(&[]), "joined the");
    }

    #[test]
    fn a_hotword_wins_when_its_path_is_close() {
        assert_eq!(decode(&["jaundice"]), "jaundice the");
    }

    /// cargo test --release --lib decode_speed -- --ignored --nocapture
    #[test]
    #[ignore]
    fn decode_speed() {
        // A 4097-class model, 240 ms chunks (3 frames), random peaked scores, 60 s of audio.
        let pieces: Vec<String> = (0..4096).map(|i| if i % 3 == 0 { format!("\u{2581}w{i}") } else { format!("p{i}") }).collect();
        let vocab = Arc::new(Vocab::new(&pieces));
        let mut seed = 1u64;
        let mut rnd = || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1); (seed >> 33) as f32 / (1u64 << 31) as f32 };
        let frames = 750;
        let rows: Vec<f32> = (0..frames).flat_map(|_| {
            let mut r: Vec<f32> = (0..4097).map(|_| -12.0 - 6.0 * rnd()).collect();
            r[4096] = -0.3;
            for _ in 0..3 { r[(rnd() * 4096.0) as usize] = -1.5 - rnd(); }
            r
        }).collect();
        for n_hot in [0usize, 30, 1000, 10000] {
            let list: Vec<String> = (0..n_hot).map(|i| format!("hotword{i}")).collect();
            let hot = Arc::new(Hotwords::new(&list));
            for width in [1usize, 4, 8, 16] {
                let mut d = Decoder::new(vocab.clone(), hot.clone(), width);
                let t = std::time::Instant::now();
                for chunk in rows.chunks(3 * 4097) {
                    d.feed(chunk, 4097);
                }
                let ms = t.elapsed().as_secs_f64() * 1000.0 / (frames / 3) as f64;
                println!("hotwords {n_hot:>5}  beam {width:>2}: {ms:.3} ms per 240 ms chunk");
            }
        }
    }

    #[test]
    fn settling_with_an_empty_best_beam_keeps_waiting() {
        let mut d = Decoder::new(vocab(), Arc::new(Hotwords::new(&[])), 8);
        let (words, pending) = d.feed(&frame(&[(7, 0.7), (5, 0.3)]), 8);
        assert!(words.is_empty() && pending.is_empty());
    }

    #[test]
    fn an_unfinished_hotword_is_refunded() {
        let h = Hotwords::new(&["jaundice".into()]);
        let (n, e, s) = h.step("", 0.0, " jau");
        assert!(e > 0.0 && s == 0.0);
        let (_, e2, s2) = h.step(&n, e, " the");
        assert_eq!((e2, s2), (0.0, 0.0));
    }

    #[test]
    fn fuzzy_spotter_fixes_misspellings_but_not_known_words() {
        assert!(close("jondice", "jaundice") && close("jhondis", "jaundice") && close("juondis", "jaundice"));
        assert!(!close("jon", "jaundice") && !close("jeans", "jaundice") && !close("joined", "jaundice"));
        let p = ["\u{2581}this", "\u{2581}is", "\u{2581}dis", "<b>"].map(String::from);
        let mut d = Decoder::new(Arc::new(Vocab::new(&p)), Arc::new(Hotwords::new(&["jaundice".into()])), 4);
        let mut out = Vec::new();
        for w in ["this", "jondice", "jhon", "dis", "is"] {
            d.spot(Word { word: w.into(), start: 0.0, end: 0.0 }, &mut out);
        }
        let got: Vec<_> = out.iter().map(|w| w.word.as_str()).collect();
        assert_eq!(got, ["this", "jaundice", "jaundice", "is"]);
    }
}
