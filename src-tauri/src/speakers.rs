//! Live speaker labels. Streaming diarization only marks TURNS (stretches where someone speaks
//! alone, split by pauses); WHO a turn is comes from the people database: embed the turn's clean
//! speech at 2 / 3 / 6 s, cosine against every saved signature (-> a person), else against this
//! unlabelled voices kept from earlier meetings and this one (-> "Voice N", N is stable), else a new
//! "Voice N" saved to the database. The diarizer's slot numbers are
//! not used: they flip within one person and are shared by two (measured), so they cannot name anyone.

use crate::engines;
use crate::parakeet::{Diarizer, Encoder, Segment};
use crate::people::{self, cosine, now_secs, now_u64, PeopleDb, Who};
use crate::asr::SAMPLE_RATE;
use anyhow::{anyhow, bail, Result};
use serde::Serialize;
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

/// Clean audio kept per turn.
const TURN_CAP: usize = 12 * SAMPLE_RATE;
/// A turn is embedded once it has this much clean speech, each time over all of it so far.
// 2 s not 1.5: same-voice cosine on 1.5 s clips ran 0.50-0.78, on 2 s clips 0.64-0.82.
const STAGES_S: [f32; 3] = [2.0, 3.0, 6.0];
/// Signatures kept per unknown voice in memory (the database keeps `MAX_VOICE_SIGS`).
const UNKNOWN_SIGS: usize = 6;
/// Playback sample kept per voice.
const SAMPLE_CAP: usize = 8 * SAMPLE_RATE;
/// Frames (100 ms) without a lone speaker that end a turn.
// 0.8 s (was 0.5): AMI ES2002 excerpts, DER 35 -> 29 % (missed speech inside a speaker's own pauses). Someone
// taking over with no pause is only caught by the window check below, 3 s later.
const GAP_FRAMES: usize = 8;
/// A long turn is re-checked on its last `WINDOW` of speech after every further `WINDOW`; a window
/// that is another voice starts a new turn.
const WINDOW: usize = 3 * SAMPLE_RATE;
/// A new turn is provisionally the previous turn's voice if it starts within this many frames of it.
const CONTINUES_FRAMES: usize = 30;
/// See `Params::new_min`.
const NEW_MIN_S: f32 = 3.0;
/// Cosine to join a meeting's unnamed voice (people keep the settings threshold, 0.6); 0.5 made fewer stray voices.
const VOICE_THRESHOLD: f32 = 0.5;
/// See `Params::hang`.
const HANG_FRAMES: usize = 3;
/// Audio kept for slicing clean speech out once the diarizer catches up (it trails ~0.32 s+).
const HISTORY: usize = 30 * SAMPLE_RATE;
const FRAME: usize = SAMPLE_RATE / 10;
const MAX_SEGMENTS: usize = 64;
/// Label of a turn nobody has fingerprinted yet.
pub const PENDING: &str = "…";

/// The tunables above, overridable for evaluation with `RT_CAPTIONS_TUNE=gap=5,stages=2/3/6,window=3,
/// continues=30,thr_person=0.6,thr_voice=0.6,new_min=0,latency=3` (any subset). Defaults are the constants.
#[derive(Clone, Debug)]
pub struct Params {
    pub gap: usize,
    pub stages: [f32; 3],
    pub window: usize,
    pub continues: usize,
    pub thr_person: Option<f32>,
    pub thr_voice: Option<f32>,
    /// A first-guess embedding of less than this many seconds never creates a new "Voice N" (the turn keeps
    /// the previous voice's label until a longer embedding).
    pub new_min: f32,
    pub latency: i32,
    /// Evaluation only: frames after a turn's last speech that still carry its label in `Labeler::segments`
    /// (captions trail the speech by 0.3-0.5 s and take the latest turn's label).
    pub hang: usize,
}

impl Params {
    pub fn get() -> &'static Params {
        static P: std::sync::OnceLock<Params> = std::sync::OnceLock::new();
        P.get_or_init(|| Self::parse(&std::env::var("RT_CAPTIONS_TUNE").unwrap_or_default()))
    }

    pub fn parse(spec: &str) -> Params {
        let mut p = Params { gap: GAP_FRAMES, stages: STAGES_S, window: WINDOW, continues: CONTINUES_FRAMES, thr_person: None, thr_voice: None, new_min: NEW_MIN_S, latency: crate::parakeet::DIAR_LATENCY_DEFAULT, hang: HANG_FRAMES };
        for kv in spec.split(',').filter(|s| !s.is_empty()) {
            let (k, v) = kv.split_once('=').unwrap_or((kv, ""));
            let f = v.parse::<f32>().unwrap_or(0.0);
            match k {
                "gap" => p.gap = f as usize,
                "stages" => {
                    let s: Vec<f32> = v.split('/').filter_map(|x| x.parse().ok()).collect();
                    if s.len() == 3 {
                        p.stages = [s[0], s[1], s[2]];
                    }
                }
                "window" => p.window = (f * SAMPLE_RATE as f32) as usize,
                "continues" => p.continues = f as usize,
                "thr_person" => p.thr_person = Some(f),
                "thr_voice" => p.thr_voice = Some(f),
                "new_min" => p.new_min = f,
                "latency" => p.latency = f as i32,
                "hang" => p.hang = f as usize,
                _ => eprintln!("RT_CAPTIONS_TUNE: unknown key '{k}'"),
            }
        }
        p
    }
}

/// "Voice 3" for number 3.
pub fn voice_label(n: usize) -> String {
    format!("Voice {n}")
}

/// Appends to a buffer, keeping only the last `cap` samples.
pub fn push_capped(buf: &mut Vec<f32>, pcm: &[f32], cap: usize) {
    buf.extend_from_slice(pcm);
    let extra = buf.len().saturating_sub(cap);
    buf.drain(..extra);
}

/// One voice heard in this meeting, for the "Voices in this meeting" list.
#[derive(Clone, Debug, Serialize)]
pub struct VoiceInfo {
    /// "person:<id>" or "voice:<n>".
    pub key: String,
    pub label: String,
    /// Saved person this voice is.
    pub person: Option<String>,
    /// Best cosine any turn matched with.
    pub score: Option<f32>,
    pub turns: usize,
    pub seconds: f32,
}

/// A voice in this meeting: a saved person, or an unknown "Voice N" with the signatures heard.
struct Ident {
    uid: usize,
    person: Option<u64>,
    /// Unlabelled voice id ("Voice N") when `person` is none.
    number: usize,
    sigs: Vec<(Vec<f32>, f32)>,
    sample: Vec<f32>,
}

struct Turn {
    /// Clean speech of this turn (dropped once it ends).
    clean: Vec<f32>,
    heard: usize,
    stage: usize,
    ident: Option<usize>,
    score: f32,
    sig: Option<(Vec<f32>, f32)>,
    /// The previous turn's voice, if this one follows closely: the label until the first embedding.
    prov: Option<usize>,
    /// `heard` at the last window check.
    checked: usize,
    open: bool,
}

/// Turns and the voices they belong to. Pure: audio and embeddings come in, labels go out.
pub struct Roster {
    p: Params,
    thr_person: f32,
    thr_voice: f32,
    idents: Vec<Ident>,
    /// Turn id = index.
    turns: Vec<Turn>,
    cur: Option<usize>,
    /// Frames since the last speech.
    gap: usize,
    /// The voice of the latest resolved turn.
    last: Option<usize>,
    next_uid: usize,
    /// This meeting's id, stamped on the signatures it adds.
    pub meeting: String,
    /// Turns of unlabelled voices that ended and are not saved yet: (voice id, clean audio, seconds heard).
    finished: Vec<(u64, Vec<f32>, f32)>,
}

impl Roster {
    pub fn new(threshold: f32) -> Self {
        Self { p: Params::get().clone(), thr_person: Params::get().thr_person.unwrap_or(threshold), thr_voice: Params::get().thr_voice.unwrap_or(VOICE_THRESHOLD.min(threshold)), idents: Vec::new(), turns: Vec::new(), cur: None, gap: usize::MAX / 2, last: None, next_uid: 0, meeting: String::new(), finished: Vec::new() }
    }

    /// Ended turns of unlabelled voices, to save their seconds and audio.
    pub fn take_finished(&mut self) -> Vec<(u64, Vec<f32>, f32)> {
        std::mem::take(&mut self.finished)
    }

    /// The meeting is over: the open turn ends.
    pub fn finish(&mut self) {
        self.close_cur();
    }

    /// The open turn, if any.
    pub fn open_turn(&self) -> Option<usize> {
        self.cur
    }

    /// Unlabelled voice id of an unknown voice.
    pub fn voice_id(&self, uid: usize) -> Option<u64> {
        self.ident(uid).filter(|i| i.person.is_none()).map(|i| i.number as u64)
    }

    fn ident(&self, uid: usize) -> Option<&Ident> {
        self.idents.iter().find(|i| i.uid == uid)
    }

    fn ident_mut(&mut self, uid: usize) -> Option<&mut Ident> {
        self.idents.iter_mut().find(|i| i.uid == uid)
    }

    /// The open turn, or a new one.
    pub fn ensure(&mut self) -> usize {
        if let Some(c) = self.cur {
            return c;
        }
        self.new_turn(Vec::new(), 0)
    }

    fn new_turn(&mut self, clean: Vec<f32>, heard: usize) -> usize {
        let prov = self.last.filter(|_| self.gap < self.p.continues);
        self.turns.push(Turn { clean, heard, stage: 0, ident: None, score: 0.0, sig: None, prov, checked: heard, open: true });
        self.cur = Some(self.turns.len() - 1);
        self.turns.len() - 1
    }

    /// The turn captions belong to now: the open one, else the latest.
    pub fn current(&mut self) -> usize {
        match self.cur.or(self.turns.len().checked_sub(1)) {
            Some(t) => t,
            None => self.ensure(),
        }
    }

    fn close_cur(&mut self) {
        if let Some(c) = self.cur.take() {
            self.keep_sample(c);
            if let Some(id) = self.turns[c].ident.and_then(|u| self.voice_id(u)) {
                self.finished.push((id, self.turns[c].clean.clone(), self.turns[c].heard as f32 / SAMPLE_RATE as f32));
            }
            self.turns[c].open = false;
            self.turns[c].clean = Vec::new();
        }
    }

    /// The longest turn of a voice is its playback sample.
    fn keep_sample(&mut self, turn: usize) {
        let t = &self.turns[turn];
        let Some(uid) = t.ident else { return };
        let clip = t.clean[t.clean.len().saturating_sub(SAMPLE_CAP)..].to_vec();
        if let Some(i) = self.ident_mut(uid).filter(|i| clip.len() > i.sample.len()) {
            i.sample = clip;
        }
    }

    /// One diarized 100 ms frame; `speech` = someone is speaking alone in it.
    pub fn frame(&mut self, speech: bool, pcm: &[f32]) {
        if speech {
            let c = self.ensure();
            self.gap = 0;
            let t = &mut self.turns[c];
            push_capped(&mut t.clean, pcm, TURN_CAP);
            t.heard += pcm.len();
        } else {
            self.gap = self.gap.saturating_add(1);
            if self.gap >= self.p.gap {
                self.close_cur();
            }
        }
    }

    /// The open turn if it needs an embedding now, with the audio to embed: all its speech so far
    /// for the first three, then the last `WINDOW` of it every `WINDOW`.
    pub fn due(&self) -> Option<(usize, &[f32])> {
        let c = self.cur?;
        let t = &self.turns[c];
        match self.p.stages.get(t.stage) {
            Some(s) => (t.heard >= (*s * SAMPLE_RATE as f32) as usize).then_some((c, t.clean.as_slice())),
            None => (t.heard >= t.checked + self.p.window).then(|| (c, &t.clean[t.clean.len() - self.p.window..])),
        }
    }

    /// A window check of an established turn: a different voice starts a new turn from that window.
    /// Returns the turn the window belongs to.
    pub fn check_window(&mut self, turn: usize, v: Vec<f32>, db: &mut PeopleDb) -> usize {
        self.turns[turn].checked = self.turns[turn].heard;
        let own = self.turns[turn].ident.and_then(|u| self.ident(u));
        let same = match db.identify(&v, self.thr_person, self.thr_voice) {
            Some((Who::Person(pid), _)) => own.is_some_and(|i| i.person == Some(pid)),
            _ => own.is_some_and(|i| i.person.is_none() && i.sigs.iter().any(|s| cosine(&s.0, &v) >= self.thr_voice)),
        };
        if same {
            return turn;
        }
        let w = self.p.window;
        let audio = self.turns[turn].clean[self.turns[turn].clean.len() - w..].to_vec();
        self.close_cur();
        let n = self.new_turn(audio, w);
        self.resolve(n, v, w as f32 / SAMPLE_RATE as f32, db);
        n
    }

    /// Records an embedding of turn `turn` (`secs` of audio): saved person, else an unlabelled voice
    /// (kept from before or heard in this meeting), else a new "Voice N" in `db`. The turn's earlier
    /// embedding is replaced.
    pub fn resolve(&mut self, turn: usize, v: Vec<f32>, secs: f32, db: &mut PeopleDb) {
        let old = self.turns[turn].ident.take();
        let mut reuse = None;
        if let (Some(u), Some(sig)) = (old, self.turns[turn].sig.take()) {
            if let Some(id) = self.voice_id(u) {
                db.retract_voice_sig(id, &sig.0);
                reuse = Some(id); // a lone first guess keeps its number if nothing else matches
            }
            if let Some(i) = self.ident_mut(u) {
                i.sigs.retain(|s| s.0 != sig.0);
            }
        }
        self.gc();
        let hit = db.identify(&v, self.thr_person, self.thr_voice);
        if hit.is_none() && old.is_none() && secs < self.p.new_min {
            self.turns[turn].stage += 1; // too short a sample to start a new voice from: wait for a longer one
            return;
        }
        let (uid, score) = match hit {
            Some((Who::Person(pid), c)) => (self.person_ident(pid), c),
            Some((Who::Voice(id), c)) => (self.voice_ident(id as usize), c),
            None => {
                let id = reuse.filter(|r| db.unlabelled.iter().any(|u| u.id == *r && u.signatures.is_empty())).unwrap_or_else(|| db.alloc_voice_id());
                (self.voice_ident(id as usize), 1.0)
            }
        };
        let i = self.ident_mut(uid).unwrap();
        i.sigs.push((v.clone(), secs));
        if i.person.is_none() {
            let extra = i.sigs.len().saturating_sub(UNKNOWN_SIGS);
            i.sigs.drain(..extra);
            let id = i.number as u64;
            db.add_voice_sig(id, v.clone(), secs, &self.meeting, now_u64());
        }
        db.prune_empty();
        let t = &mut self.turns[turn];
        t.ident = Some(uid);
        t.score = score;
        t.sig = Some((v, secs));
        t.stage += 1;
        if t.stage >= self.p.stages.len() {
            t.checked = t.heard;
        }
        self.last = Some(uid);
        self.keep_sample(turn);
    }

    fn person_ident(&mut self, pid: u64) -> usize {
        match self.idents.iter().find(|i| i.person == Some(pid)) {
            Some(i) => i.uid,
            None => self.push_ident(Some(pid), 0),
        }
    }

    /// This meeting's ident for unlabelled voice `n`.
    fn voice_ident(&mut self, n: usize) -> usize {
        match self.idents.iter().find(|i| i.person.is_none() && i.number == n) {
            Some(i) => i.uid,
            None => self.push_ident(None, n),
        }
    }

    fn push_ident(&mut self, person: Option<u64>, number: usize) -> usize {
        self.next_uid += 1;
        self.idents.push(Ident { uid: self.next_uid, person, number, sigs: Vec::new(), sample: Vec::new() });
        self.next_uid
    }

    /// Forgets voices no turn belongs to any more (a first guess that a later embedding corrected).
    fn gc(&mut self) {
        let used: Vec<usize> = self.turns.iter().filter_map(|t| t.ident).collect();
        self.idents.retain(|i| used.contains(&i.uid));
    }

    /// Has this turn had all its first embeddings (so only window checks remain)?
    pub fn established(&self, turn: usize) -> bool {
        self.turns[turn].stage >= self.p.stages.len()
    }

    pub fn score(&self, turn: usize) -> f32 {
        self.turns[turn].score
    }

    /// Gives up on a turn's embeddings.
    pub fn skip(&mut self, turn: usize) {
        self.turns[turn].stage = self.p.stages.len();
        self.turns[turn].checked = self.turns[turn].heard;
    }

    pub fn label(&self, turn: usize, db: &PeopleDb) -> String {
        let t = &self.turns[turn];
        t.ident.or(t.prov).and_then(|u| self.ident(u)).map_or_else(|| PENDING.to_owned(), |i| self.ident_label(i, db))
    }

    fn ident_label(&self, i: &Ident, db: &PeopleDb) -> String {
        match i.person {
            Some(p) => db.people.iter().find(|x| x.id == p).map_or_else(|| PENDING.to_owned(), |x| x.name.clone()),
            None => voice_label(i.number),
        }
    }

    fn key(i: &Ident) -> String {
        i.person.map_or_else(|| format!("voice:{}", i.number), |p| format!("person:{p}"))
    }

    /// "turn:<id>", "person:<id>" or "voice:<n>" -> the voice.
    fn find(&self, key: &str) -> Option<usize> {
        if let Some(t) = key.strip_prefix("turn:") {
            return self.turns.get(t.parse::<usize>().ok()?)?.ident;
        }
        self.idents.iter().find(|i| Self::key(i) == key).map(|i| i.uid)
    }

    pub fn voices(&self, db: &PeopleDb) -> Vec<VoiceInfo> {
        self.idents.iter().map(|i| {
            let mine = || self.turns.iter().filter(|t| t.ident == Some(i.uid));
            VoiceInfo {
                key: Self::key(i),
                label: self.ident_label(i, db),
                person: i.person.and_then(|_| Some(self.ident_label(i, db))),
                score: mine().map(|t| t.score).max_by(f32::total_cmp),
                turns: mine().count(),
                seconds: mine().map(|t| t.heard).sum::<usize>() as f32 / SAMPLE_RATE as f32,
            }
        }).collect()
    }

    /// Up to 8 s of the voice's longest turn.
    pub fn sample(&self, key: &str) -> Vec<f32> {
        let Some(uid) = self.find(key) else { return Vec::new() };
        let cur = self.turns.iter().filter(|t| t.ident == Some(uid) && t.open).map(|t| &t.clean);
        let mut best = self.ident(uid).map(|i| &i.sample).filter(|s| !s.is_empty());
        for c in cur {
            if best.map_or(true, |b| c.len() > b.len()) {
                best = Some(c);
            }
        }
        best.map(|b| b[b.len().saturating_sub(SAMPLE_CAP)..].to_vec()).unwrap_or_default()
    }

    /// Signatures to save when the user names a voice: the turn's own latest, or the voice's last three.
    pub fn signatures_for(&self, key: &str) -> Option<(usize, Vec<(Vec<f32>, f32)>)> {
        let uid = self.find(key)?;
        let own = key.strip_prefix("turn:").and_then(|t| self.turns.get(t.parse::<usize>().ok()?)?.sig.clone());
        let sigs = own.map(|s| vec![s]).unwrap_or_else(|| {
            let all = &self.ident(uid).unwrap().sigs;
            all[all.len().saturating_sub(3)..].to_vec()
        });
        (!sigs.is_empty()).then_some((uid, sigs))
    }

    /// The voice `uid` is saved person `pid` now (merging with that person's voice if already here).
    pub fn set_person(&mut self, uid: usize, pid: u64) {
        let twin = self.idents.iter().find(|i| i.person == Some(pid) && i.uid != uid).map(|i| i.uid);
        if let Some(t) = twin {
            let gone = self.idents.iter().position(|i| i.uid == uid).unwrap();
            let gone = self.idents.remove(gone);
            let keep = self.ident_mut(t).unwrap();
            keep.sigs.extend(gone.sigs);
            if gone.sample.len() > keep.sample.len() {
                keep.sample = gone.sample;
            }
            for x in self.turns.iter_mut().filter(|x| x.ident == Some(uid)) {
                x.ident = Some(t);
            }
            for x in self.turns.iter_mut().filter(|x| x.prov == Some(uid)) {
                x.prov = Some(t);
            }
            if self.last == Some(uid) {
                self.last = Some(t);
            }
        } else if let Some(i) = self.ident_mut(uid) {
            i.person = Some(pid);
        }
    }

    /// The people file changed: people that are gone become unknown voices, and unknown voices
    /// that now match a saved person become that person.
    pub fn reload(&mut self, db: &mut PeopleDb) {
        for k in 0..self.idents.len() {
            if self.idents[k].person.is_some_and(|p| !db.people.iter().any(|x| x.id == p)) {
                let id = db.alloc_voice_id();
                for (v, secs) in &self.idents[k].sigs {
                    db.add_voice_sig(id, v.clone(), *secs, &self.meeting, now_u64());
                }
                self.idents[k].person = None;
                self.idents[k].number = id as usize;
            }
        }
        let hits: Vec<(usize, u64)> = self.idents.iter().filter(|i| i.person.is_none())
            .filter_map(|i| i.sigs.iter().filter_map(|s| db.best_match(&s.0)).filter(|b| b.2 >= self.thr_person)
                .max_by(|a, b| a.2.total_cmp(&b.2)).map(|b| (i.uid, b.0))).collect();
        for (uid, pid) in hits {
            self.set_person(uid, pid);
        }
    }
}

/// One diarized 100 ms frame, for `Labeler::segments` (evaluation).
struct FrameRec {
    speech: bool,
    /// The open turn after this frame.
    turn: Option<usize>,
    /// The label the captions had for it at the time.
    live: String,
}

pub struct Labeler {
    diar: Diarizer,
    enc: Encoder,
    db: PeopleDb,
    encoder_file: String,
    meeting: String,
    roster: Roster,
    /// Samples fed since the session started.
    total: usize,
    /// Audio from `hist_start` on, to slice clean speech from.
    hist: Vec<f32>,
    hist_start: usize,
    /// Diarized samples already sorted into turns.
    sorted: usize,
    ended: Vec<Segment>,
    open: Vec<Segment>,
    /// Wall time of the latest `feed`.
    pub last_feed_ms: f64,
    /// Per-frame record, kept only when asked (`record`).
    rec: Option<Vec<FrameRec>>,
}

impl Labeler {
    /// Needs the diarization and speaker models.
    pub fn load(threshold: f64) -> Result<Self> {
        let Some(diar_model) = engines::installed(engines::get_any("diarization")?) else {
            bail!("diarization model not downloaded (run: rt-captions download diarization)");
        };
        let spk = engines::get_any("speakers")?;
        let Some(spk_model) = engines::installed(spk) else {
            bail!("speaker model not downloaded (run: rt-captions download speakers)");
        };
        let meeting = now_secs();
        let mut roster = Roster::new(threshold as f32);
        roster.meeting = meeting.clone();
        Ok(Self { diar: Diarizer::load(&diar_model, Params::get().latency)?, enc: Encoder::load(&spk_model)?, db: PeopleDb::load_for(spk.file),
                  encoder_file: spk.file.to_owned(), meeting, roster, total: 0,
                  hist: Vec::new(), hist_start: 0, sorted: 0, ended: Vec::new(), open: Vec::new(), last_feed_ms: 0.0, rec: None })
    }

    pub fn feed(&mut self, pcm: &[f32]) -> Result<()> {
        let t = std::time::Instant::now();
        self.total += pcm.len();
        self.hist.extend_from_slice(pcm);
        let extra = self.hist.len().saturating_sub(HISTORY);
        self.hist.drain(..extra);
        self.hist_start += extra;
        let done = self.diar.feed(pcm)?;
        self.ended.extend(done);
        let extra = self.ended.len().saturating_sub(MAX_SEGMENTS);
        self.ended.drain(..extra);
        self.open = self.diar.active();
        self.sort_turns();
        self.fingerprint();
        self.save_finished();
        self.last_feed_ms = t.elapsed().as_secs_f64() * 1000.0;
        Ok(())
    }

    /// Files each diarized 100 ms frame into the turns (lone speaker) or the gaps between them.
    fn sort_turns(&mut self) {
        let diarized = ((self.diar.time() * SAMPLE_RATE as f64) as usize).min(self.total);
        while self.sorted + FRAME <= diarized {
            let a = self.sorted;
            self.sorted += FRAME;
            if a < self.hist_start {
                if let Some(rec) = &mut self.rec {
                    rec.push(FrameRec { speech: false, turn: None, live: String::new() });
                }
                continue;
            }
            let mid = (a + FRAME / 2) as f64 / SAMPLE_RATE as f64;
            let speech = lone_speaker(self.ended.iter().chain(&self.open), mid).is_some();
            let from = a - self.hist_start;
            self.roster.frame(speech, &self.hist[from..from + FRAME]);
            if let Some(rec) = &mut self.rec {
                let turn = self.roster.open_turn();
                let live = turn.map_or(String::new(), |t| self.roster.label(t, &self.db));
                rec.push(FrameRec { speech, turn, live });
            }
        }
    }

    /// Seconds the diarizer trails the audio fed (its latency mode's chunk delay).
    pub fn diar_lag_s(&self) -> f64 {
        self.total as f64 / SAMPLE_RATE as f64 - self.diar.time()
    }

    /// Keeps a per-frame record so `segments` can say who the labeller thought spoke when.
    pub fn record(&mut self) {
        self.rec = Some(Vec::new());
    }

    /// Hypothesis segments (start s, end s, label) of everything labelled so far: speech frames plus the
    /// short pauses inside a turn (captions get the open turn in those too). `live` = the label each
    /// frame had when it was heard, else the label its turn ended up with.
    pub fn segments(&self, live: bool) -> Vec<(f64, f64, String)> {
        let Some(rec) = &self.rec else { return Vec::new() };
        let mut next: Option<usize> = None;
        let mut labels: Vec<Option<String>> = vec![None; rec.len()];
        for (i, f) in rec.iter().enumerate().rev() {
            if f.speech {
                next = f.turn;
            }
            let Some(t) = f.turn.filter(|t| f.speech || next == Some(*t)) else { continue };
            labels[i] = Some(if live { f.live.clone() } else { self.roster.label(t, &self.db) });
        }
        let (mut last, mut age) = (None::<String>, usize::MAX);
        for l in labels.iter_mut() {
            match (&*l, &last) {
                (Some(x), _) => (last, age) = (Some(x.clone()), 0),
                (None, Some(x)) if age < Params::get().hang => {
                    *l = Some(x.clone());
                    age += 1;
                }
                _ => {}
            }
        }
        let mut out: Vec<(f64, f64, String)> = Vec::new();
        for (i, l) in labels.into_iter().enumerate() {
            let Some(l) = l else { continue };
            let (a, b) = (i as f64 * 0.1, (i + 1) as f64 * 0.1);
            match out.last_mut() {
                Some(last) if last.2 == l && (last.1 - a).abs() < 1e-6 => last.1 = b,
                _ => out.push((a, b, l)),
            }
        }
        out
    }

    /// The open turn gets its next signature when it has enough speech.
    fn fingerprint(&mut self) {
        let Some((turn, audio)) = self.roster.due() else { return };
        let secs = audio.len() as f32 / SAMPLE_RATE as f32;
        match self.enc.embed(audio) {
            Ok(v) if self.roster.established(turn) => {
                let t = self.roster.check_window(turn, v, &mut self.db);
                if t != turn {
                    // the window's speech now belongs to the new turn
                    let mut left = Params::get().window / FRAME;
                    for f in self.rec.iter_mut().flatten().rev().filter(|f| f.speech) {
                        if left == 0 || f.turn != Some(turn) {
                            break;
                        }
                        f.turn = Some(t);
                        left -= 1;
                    }
                }
                crate::engine::debug_log(format_args!("turn {turn} window: {}", if t == turn { "same voice".into() } else { format!("new turn {t}: {} (cosine {:.2})", self.roster.label(t, &self.db), self.roster.score(t)) }));
            }
            Ok(v) => {
                self.roster.resolve(turn, v, secs, &mut self.db);
                crate::engine::debug_log(format_args!("turn {turn} ({secs:.1} s): {} (cosine {:.2})", self.roster.label(turn, &self.db), self.roster.score(turn)));
            }
            Err(e) => {
                eprintln!("speaker signature: {e:#}");
                self.roster.skip(turn); // do not retry this turn
            }
        }
    }

    /// Turns that ended: their seconds and audio go to the voices' samples, and the database is saved
    /// (once per turn, not per frame).
    fn save_finished(&mut self) {
        let fin = self.roster.take_finished();
        if fin.is_empty() {
            return;
        }
        let (dir, now) = (people::voices_dir(), now_u64());
        for (id, audio, heard) in fin {
            if let Err(e) = self.db.hear(&dir, id, &audio, heard, now) {
                eprintln!("voice sample: {e:#}");
            }
        }
        if let Err(e) = self.db.save() {
            eprintln!("saving voices: {e:#}");
        }
    }

    /// The meeting ended: close the open turn and save.
    pub fn finish(&mut self) {
        self.roster.finish();
        self.save_finished();
    }

    /// Who is speaking at the caption edge now: (turn id, label).
    pub fn current(&mut self) -> Option<(usize, String)> {
        let turn = self.roster.current();
        Some((turn, self.roster.label(turn, &self.db)))
    }

    /// Names a voice (a "turn:<id>", "person:<id>" or "voice:<n>" key) for this meeting and saves its
    /// signatures under `name` in the people database.
    pub fn assign(&mut self, key: &str, name: &str) -> Result<()> {
        let name = name.trim();
        if name.is_empty() {
            bail!("assign: empty name");
        }
        let Some((uid, sigs)) = self.roster.signatures_for(key) else { bail!("not enough speech from this voice yet") };
        // (the app tells us when it edits the file: `reload_people`)
        let named = self.roster.voice_id(uid).map(|id| self.db.label_voice_files(&people::voices_dir(), id, name, false)).transpose()?.flatten();
        let pid = match named {
            Some(pid) => pid,
            None => sigs.into_iter().fold(0, |_, (v, secs)| self.db.assign(name, v, &self.meeting, secs)),
        };
        self.db.save()?;
        self.roster.set_person(uid, pid);
        Ok(())
    }

    /// The people database changed on disk: reload it and re-match this meeting's voices.
    pub fn reload_people(&mut self) {
        self.db = PeopleDb::load_for(&self.encoder_file);
        self.roster.reload(&mut self.db);
    }

    pub fn voices(&self) -> Vec<VoiceInfo> {
        self.roster.voices(&self.db)
    }

    pub fn sample(&self, key: &str) -> Vec<f32> {
        self.roster.sample(key)
    }
}

/// Writes segments as an RTTM file (one SPEAKER line each).
pub fn write_rttm(path: &std::path::Path, name: &str, segs: &[(f64, f64, String)]) -> Result<()> {
    let mut out = String::new();
    for (a, b, who) in segs {
        out.push_str(&format!("SPEAKER {name} 1 {a:.2} {:.2} <NA> <NA> {} <NA> <NA>\n", b - a, who.replace(' ', "_")));
    }
    Ok(std::fs::write(path, out)?)
}

/// The one speaker whose segments cover `t`, if exactly one does.
fn lone_speaker<'a>(segs: impl Iterator<Item = &'a Segment>, t: f64) -> Option<usize> {
    let mut who = None;
    for s in segs.filter(|s| s.start <= t && t <= s.end) {
        match who {
            Some(w) if w != s.speaker => return None,
            _ => who = Some(s.speaker),
        }
    }
    who
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(speaker: usize, start: f64, end: f64) -> Segment {
        Segment { speaker, start, end }
    }

    /// Unit vector at `deg` degrees: cosine of two of them is cos(difference).
    fn at(deg: f32) -> Vec<f32> {
        let r = deg.to_radians();
        vec![r.cos(), r.sin()]
    }

    fn feed(r: &mut Roster, speech: bool, frames: usize) {
        for _ in 0..frames {
            r.frame(speech, &[0.1; FRAME]);
        }
    }

    /// A turn of `frames` speech frames, resolved to the voice at `deg`, then a closing pause.
    fn turn(r: &mut Roster, db: &mut PeopleDb, deg: f32, frames: usize) -> usize {
        feed(r, true, frames);
        let t = r.current();
        r.resolve(t, at(deg), frames as f32 / 10.0, db);
        feed(r, false, GAP_FRAMES);
        t
    }

    #[test]
    fn lone_speaker_needs_exactly_one() {
        let segs = [seg(0, 0.0, 2.0), seg(1, 1.0, 3.0)];
        assert_eq!(lone_speaker(segs.iter(), 0.5), Some(0));
        assert_eq!(lone_speaker(segs.iter(), 1.5), None);
        assert_eq!(lone_speaker(segs.iter(), 2.5), Some(1));
        assert_eq!(lone_speaker(segs.iter(), 9.0), None);
    }

    #[test]
    fn turns_split_on_a_long_pause_only() {
        let mut r = Roster::new(0.6);
        feed(&mut r, true, 10);
        feed(&mut r, false, GAP_FRAMES - 1);
        feed(&mut r, true, 10); // short pause: same turn, whoever the diarizer's slots say
        assert_eq!(r.turns.len(), 1);
        feed(&mut r, false, GAP_FRAMES);
        assert!(!r.turns[0].open);
        feed(&mut r, true, 10);
        assert_eq!(r.turns.len(), 2);
        assert_eq!(r.turns[0].heard, 20 * FRAME);
        assert_eq!(r.current(), 1, "captions get the open turn");
        feed(&mut r, false, GAP_FRAMES);
        assert_eq!(r.current(), 1, "and the latest one in a pause");
    }

    #[test]
    fn turn_audio_is_capped() {
        let mut r = Roster::new(0.6);
        feed(&mut r, true, 200);
        assert_eq!(r.turns[0].clean.len(), TURN_CAP);
        assert_eq!(r.turns[0].heard, 200 * FRAME);
    }

    #[test]
    fn embeddings_are_due_at_each_stage_then_by_window() {
        let mut r = Roster::new(0.6);
        let mut db = PeopleDb::default();
        feed(&mut r, true, 19);
        assert!(r.due().is_none());
        feed(&mut r, true, 1);
        let (t, audio) = r.due().unwrap();
        assert_eq!(audio.len(), 20 * FRAME);
        r.resolve(t, at(0.0), 2.0, &mut db);
        assert!(r.due().is_none());
        feed(&mut r, true, 10);
        r.resolve(0, at(0.0), 3.0, &mut db);
        feed(&mut r, true, 30);
        r.resolve(0, at(0.0), 6.0, &mut db);
        assert!(r.established(0) && r.due().is_none());
        feed(&mut r, true, 30);
        assert_eq!(r.due().unwrap().1.len(), WINDOW, "then the last 3 s, every 3 s");
    }

    #[test]
    fn a_window_of_another_voice_starts_a_new_turn() {
        let mut r = Roster::new(0.6);
        let mut db = PeopleDb::default();
        feed(&mut r, true, 60);
        for secs in [2.0, 3.0, 6.0] {
            r.resolve(0, at(0.0), secs, &mut db);
        }
        feed(&mut r, true, 30);
        assert_eq!(r.check_window(0, at(5.0), &mut db), 0, "same voice: no split");
        feed(&mut r, true, 30);
        let n = r.check_window(0, at(100.0), &mut db);
        assert_eq!(n, 1);
        assert_eq!((r.label(0, &db), r.label(1, &db)), ("Voice 1".into(), "Voice 2".into()));
        assert_eq!(r.current(), 1);
        assert!(!r.turns[0].open && r.turns[1].heard == WINDOW);
    }

    #[test]
    fn unknown_voices_cluster_and_number_in_order_created() {
        let mut r = Roster::new(0.6);
        let mut db = PeopleDb::default();
        for deg in [0.0, 90.0, 5.0] {
            turn(&mut r, &mut db, deg, 30);
        }
        assert_eq!(r.label(0, &db), "Voice 1");
        assert_eq!(r.label(1, &db), "Voice 2");
        assert_eq!(r.label(2, &db), "Voice 1", "close to the first voice");
        assert_eq!(r.voices(&db).iter().find(|v| v.key == "voice:1").unwrap().turns, 2);
    }

    #[test]
    fn saved_person_beats_unknown() {
        let mut r = Roster::new(0.6);
        let mut db = PeopleDb::default();
        let ann = db.assign("Ann", at(0.0), "m", 5.0);
        turn(&mut r, &mut db, 10.0, 30);
        turn(&mut r, &mut db, 120.0, 30);
        assert_eq!(r.label(0, &db), "Ann");
        assert_eq!(r.label(1, &db), "Voice 1");
        assert_eq!(r.voices(&db).iter().find(|v| v.person.is_some()).unwrap().key, format!("person:{ann}"));
    }

    #[test]
    fn a_later_embedding_replaces_the_first_guess() {
        let mut r = Roster::new(0.6);
        let mut db = PeopleDb::default();
        turn(&mut r, &mut db, 0.0, 30);
        feed(&mut r, true, 20);
        r.resolve(1, at(80.0), 2.0, &mut db); // Voice 2
        feed(&mut r, true, 10);
        r.resolve(1, at(2.0), 3.0, &mut db); // really the first voice
        assert_eq!(r.label(1, &db), "Voice 1");
        assert_eq!(r.voices(&db).len(), 1, "the wrongly created voice is gone");
    }

    #[test]
    fn provisional_label_is_the_previous_voice_if_it_follows_closely() {
        let mut r = Roster::new(0.6);
        let mut db = PeopleDb::default();
        feed(&mut r, true, 3);
        assert_eq!(r.label(0, &db), PENDING, "nothing known yet");
        feed(&mut r, true, 27);
        r.resolve(0, at(0.0), 3.0, &mut db);
        feed(&mut r, false, GAP_FRAMES);
        feed(&mut r, true, 3);
        assert_eq!(r.label(1, &db), "Voice 1", "a pause, then more speech: probably the same person");
        feed(&mut r, false, CONTINUES_FRAMES);
        feed(&mut r, true, 3);
        assert_eq!(r.label(2, &db), PENDING, "long silence: unknown until fingerprinted");
    }

    #[test]
    fn assigning_makes_the_voice_a_person_and_merges() {
        let mut r = Roster::new(0.6);
        let mut db = PeopleDb::default();
        turn(&mut r, &mut db, 0.0, 30);
        turn(&mut r, &mut db, 90.0, 30);
        let (uid, sigs) = r.signatures_for("turn:0").unwrap();
        let pid = db.assign("Ann", sigs[0].0.clone(), "m", sigs[0].1);
        r.set_person(uid, pid);
        assert_eq!(r.label(0, &db), "Ann");
        assert_eq!(r.label(1, &db), "Voice 2");
        // the second voice is also Ann: merges into one person
        let (uid2, sigs2) = r.signatures_for("voice:2").unwrap();
        db.assign("ann", sigs2[0].0.clone(), "m", 1.5);
        r.set_person(uid2, pid);
        assert_eq!(r.voices(&db).len(), 1);
        assert_eq!((r.label(0, &db), r.label(1, &db)), ("Ann".into(), "Ann".into()));
        // deleting her turns the voice back into an unknown
        db.delete_person(pid);
        r.reload(&mut db);
        assert!(r.label(0, &db).starts_with("Voice "), "an unknown again, with a fresh never-reused number");
    }

    #[test]
    fn clean_audio_keeps_the_last_samples() {
        let mut buf = Vec::new();
        push_capped(&mut buf, &vec![1.0; 100], 100);
        push_capped(&mut buf, &vec![2.0; 10], 100);
        assert_eq!(buf.len(), 100);
        assert_eq!((buf[0], *buf.last().unwrap()), (1.0, 2.0));
    }
}

enum Job {
    Audio(Vec<f32>),
    Assign(String, String),
    ListVoices(Sender<Vec<VoiceInfo>>),
    VoiceSample(String, Sender<Vec<f32>>),
    ReloadPeople,
}

/// The labeler on its own thread: diarization runs alongside speech recognition, so it never
/// delays a caption. Captions read the latest "who is speaking" without waiting.
pub struct LabelerThread {
    tx: Option<Sender<Job>>,
    now: Arc<Mutex<Option<(usize, String)>>>,
    worker: Option<JoinHandle<()>>,
}

impl LabelerThread {
    pub fn spawn(threshold: f64) -> Result<Self> {
        let (tx, rx) = channel::<Job>();
        let (ready_tx, ready_rx) = channel();
        let now = Arc::new(Mutex::new(None));
        let shared = now.clone();
        let worker = std::thread::Builder::new().name("speakers".into()).spawn(move || {
            crate::engine::realtime_priority();
            let mut l = match Labeler::load(threshold) {
                Ok(l) => {
                    let _ = ready_tx.send(Ok(()));
                    l
                }
                Err(e) => {
                    let _ = ready_tx.send(Err(e));
                    return;
                }
            };
            let l = &mut l;
            // evaluation: RT_CAPTIONS_RTTM_OUT=<file> writes the labels of the whole session as an RTTM
            let rttm_out = std::env::var_os("RT_CAPTIONS_RTTM_OUT").map(std::path::PathBuf::from);
            if rttm_out.is_some() {
                l.record();
            }
            while let Ok(job) = rx.recv() {
                match job {
                    Job::Audio(pcm) => {
                        if let Err(e) = l.feed(&pcm) {
                            eprintln!("speaker labels: {e:#}");
                            return;
                        }
                        *shared.lock().unwrap_or_else(|e| e.into_inner()) = l.current();
                    }
                    Job::Assign(key, name) => {
                        match l.assign(&key, &name) {
                            Ok(()) => crate::engine::debug_log(format_args!("{key} is {name} (signature saved)")),
                            Err(e) => eprintln!("assign voice: {e:#}"),
                        }
                        *shared.lock().unwrap_or_else(|e| e.into_inner()) = l.current();
                    }
                    Job::ListVoices(reply) => {
                        let _ = reply.send(l.voices());
                    }
                    Job::VoiceSample(key, reply) => {
                        let _ = reply.send(l.sample(&key));
                    }
                    Job::ReloadPeople => {
                        l.reload_people();
                        *shared.lock().unwrap_or_else(|e| e.into_inner()) = l.current();
                    }
                }
            }
            l.finish();
            if let Some(p) = rttm_out {
                if let Err(e) = write_rttm(&p, "live", &l.segments(false)) {
                    eprintln!("writing {}: {e:#}", p.display());
                }
            }
        })?;
        ready_rx.recv().map_err(|_| anyhow!("speaker thread exited"))??;
        Ok(Self { tx: Some(tx), now, worker: Some(worker) })
    }

    fn send(&self, job: Job) {
        if let Some(tx) = &self.tx {
            let _ = tx.send(job);
        }
    }

    pub fn feed(&self, pcm: &[f32]) {
        self.send(Job::Audio(pcm.to_vec()));
    }

    /// Names a voice ("turn:<id>", "person:<id>" or "voice:<n>") and saves its signatures under `name`.
    pub fn assign(&self, key: String, name: String) {
        self.send(Job::Assign(key, name));
    }

    pub fn reload_people(&self) {
        self.send(Job::ReloadPeople);
    }

    /// Asks for the voices heard this meeting; the answer arrives on `reply` (the caller waits, we don't).
    pub fn voices(&self, reply: Sender<Vec<VoiceInfo>>) {
        self.send(Job::ListVoices(reply));
    }

    pub fn sample(&self, key: String, reply: Sender<Vec<f32>>) {
        self.send(Job::VoiceSample(key, reply));
    }

    /// The speaker of a caption with words in it: (turn id, label).
    pub fn current(&self) -> Option<(usize, String)> {
        self.now.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }
}

impl Drop for LabelerThread {
    fn drop(&mut self) {
        // Close the queue, then wait: the models must be freed before the engine's own teardown.
        self.tx.take();
        if let Some(w) = self.worker.take() {
            let _ = w.join();
        }
    }
}
