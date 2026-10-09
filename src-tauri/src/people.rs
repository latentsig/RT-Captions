//! The people database: named voice signatures (speaker-encoder vectors), matched by plain
//! cosine. A person matches when ANY of their signatures is close. Pure data, no FFI.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use crate::whistle::SAMPLE_RATE;
use std::path::{Path, PathBuf};

/// Signatures kept per person; the oldest is dropped past this.
pub const MAX_SIGNATURES: usize = 20;
/// Signatures kept per unlabelled voice.
pub const MAX_VOICE_SIGS: usize = 10;
/// Seconds of audio kept per unlabelled voice.
pub const SAMPLE_MAX_S: f32 = 60.0;
/// An unlabelled voice not heard for this long is forgotten, audio included.
pub const EXPIRE_SECS: u64 = 30 * 24 * 3600;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Signature {
    pub id: u64,
    pub vec: Vec<f32>,
    /// Unix seconds, as a string.
    pub added: String,
    /// The meeting (session start, unix seconds) it was heard in.
    pub meeting: String,
    pub seconds: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Person {
    pub id: u64,
    pub name: String,
    pub signatures: Vec<Signature>,
}

/// A voice heard in meetings that nobody has named yet. Its audio sample is `voices/<id>.wav`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Unlabelled {
    /// Stable, never reused.
    pub id: u64,
    pub signatures: Vec<Signature>,
    /// Unix seconds.
    pub first_seen: u64,
    pub last_seen: u64,
    pub meetings: Vec<String>,
    pub seconds_heard: f32,
    pub sample_secs: f32,
}

impl Unlabelled {
    /// Whole turns only: room left under the cap, and a turn under 2 s only as the very first sample.
    pub fn accepts(&self, secs: f32) -> bool {
        self.sample_secs + secs <= SAMPLE_MAX_S && (secs >= 2.0 || self.sample_secs == 0.0)
    }
}

/// Who a voice vector is closest to.
#[derive(Clone, Debug, PartialEq)]
pub enum Who {
    Person(u64),
    Voice(u64),
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PeopleDb {
    /// File name of the speaker encoder that made the vectors; another encoder's vectors are not comparable.
    #[serde(default)]
    pub encoder: String,
    #[serde(default)]
    pub people: Vec<Person>,
    #[serde(default)]
    pub unlabelled: Vec<Unlabelled>,
    /// Next unlabelled voice id (ids are never reused, even after a discard).
    #[serde(default)]
    pub next_voice_id: u64,
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let (na, nb) = (a.iter().map(|x| x * x).sum::<f32>().sqrt(), b.iter().map(|x| x * x).sum::<f32>().sqrt());
    if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na * nb) }
}

pub fn now_secs() -> String {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs()).to_string()
}

pub fn now_u64() -> u64 {
    now_secs().parse().unwrap_or(0)
}

/// 16 kHz mono 16-bit WAV bytes.
pub fn wav_bytes(pcm: &[f32]) -> Result<Vec<u8>> {
    let spec = hound::WavSpec { channels: 1, sample_rate: SAMPLE_RATE as u32, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
    let mut cur = std::io::Cursor::new(Vec::new());
    let mut w = hound::WavWriter::new(&mut cur, spec)?;
    for x in pcm {
        w.write_sample((x.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)?;
    }
    w.finalize()?;
    Ok(cur.into_inner())
}

/// Samples of one of our own wav files (empty if missing).
pub fn read_wav(p: &Path) -> Vec<f32> {
    let Ok(mut r) = hound::WavReader::open(p) else { return Vec::new() };
    r.samples::<i16>().filter_map(Result::ok).map(|s| s as f32 / i16::MAX as f32).collect()
}

fn write_wav(p: &Path, pcm: &[f32]) -> Result<()> {
    std::fs::create_dir_all(p.parent().context("voices path")?)?;
    Ok(std::fs::write(p, wav_bytes(pcm)?)?)
}

pub fn voices_dir() -> PathBuf {
    voices_dir_of(&path())
}

fn voices_dir_of(db: &Path) -> PathBuf {
    db.parent().unwrap_or(Path::new("")).join("voices")
}

pub fn sample_path(dir: &Path, id: u64) -> PathBuf {
    dir.join(format!("{id}.wav"))
}

/// `RT_CAPTIONS_DATA=<dir>` keeps people.json and voices/ there instead (evaluation runs on a scratch database).
pub fn path() -> PathBuf {
    if let Some(d) = std::env::var_os("RT_CAPTIONS_DATA") {
        return PathBuf::from(d).join("people.json");
    }
    dirs::data_local_dir().unwrap_or_default().join(crate::settings::IDENTIFIER).join("people.json")
}

impl PeopleDb {
    /// Reads the file; unlabelled voices unseen for 30 days are dropped here, with their audio.
    pub fn load_from(p: &Path) -> Self {
        let mut db: Self = std::fs::read_to_string(p).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default();
        for id in db.expire(now_u64(), EXPIRE_SECS) {
            let _ = std::fs::remove_file(sample_path(&voices_dir_of(p), id));
        }
        db
    }

    pub fn save_to(&self, p: &Path) -> Result<()> {
        std::fs::create_dir_all(p.parent().context("people path")?)?;
        // write-then-rename so a reader never sees a half-written file
        let tmp = p.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_string(self)?)?;
        std::fs::rename(&tmp, p).with_context(|| format!("writing {}", p.display()))
    }

    /// The saved file as is (whatever encoder made it).
    pub fn load() -> Self {
        Self::load_from(&path())
    }

    pub fn save(&self) -> Result<()> {
        self.save_to(&path())
    }

    /// For use with `encoder`: a DB made by another encoder is set aside (people.json.old) and
    /// an empty one starts, since its vectors mean nothing here.
    pub fn load_for(encoder: &str) -> Self {
        let mut db = Self::load();
        if !db.encoder.is_empty() && db.encoder != encoder {
            eprintln!("people.json was made by {}, not {encoder}: ignoring it (kept as people.json.old)", db.encoder);
            let _ = std::fs::rename(path(), path().with_extension("json.old"));
            db = Self::default();
        }
        db.encoder = encoder.to_owned();
        db
    }

    fn next_id(&self) -> u64 {
        let max = self.people.iter().flat_map(|p| std::iter::once(p.id).chain(p.signatures.iter().map(|s| s.id))).max();
        max.map_or(1, |m| m + 1)
    }

    /// The closest signature over everyone: (person id, name, cosine). The caller applies the threshold.
    pub fn best_match(&self, v: &[f32]) -> Option<(u64, String, f32)> {
        self.people.iter()
            .flat_map(|p| p.signatures.iter().filter(|s| s.vec.len() == v.len()).map(move |s| (p, cosine(&s.vec, v))))
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(p, c)| (p.id, p.name.clone(), c))
    }

    /// Adds a signature to the person called `name` (case-insensitive), or makes them. Returns the person id.
    pub fn assign(&mut self, name: &str, vec: Vec<f32>, meeting: &str, seconds: f32) -> u64 {
        let name = name.trim();
        let sig = Signature { id: self.next_id(), vec, added: now_secs(), meeting: meeting.to_owned(), seconds };
        match self.people.iter().position(|p| p.name.to_lowercase() == name.to_lowercase()) {
            Some(i) => {
                let p = &mut self.people[i];
                p.signatures.push(sig);
                let extra = p.signatures.len().saturating_sub(MAX_SIGNATURES);
                p.signatures.drain(..extra);
                p.id
            }
            None => {
                let id = sig.id + 1;
                self.people.push(Person { id, name: name.to_owned(), signatures: vec![sig] });
                id
            }
        }
    }

    /// Next unlabelled voice id; never handed out twice.
    pub fn alloc_voice_id(&mut self) -> u64 {
        let top = self.unlabelled.iter().map(|u| u.id).max().unwrap_or(0);
        self.next_voice_id = self.next_voice_id.max(top + 1).max(1);
        self.next_voice_id += 1;
        self.next_voice_id - 1
    }

    /// The closest person, else the closest unlabelled voice, at or above `threshold`.
    pub fn identify(&self, v: &[f32], people_threshold: f32, voice_threshold: f32) -> Option<(Who, f32)> {
        if let Some((pid, _, c)) = self.best_match(v).filter(|b| b.2 >= people_threshold) {
            return Some((Who::Person(pid), c));
        }
        self.unlabelled.iter()
            .flat_map(|u| u.signatures.iter().filter(|s| s.vec.len() == v.len()).map(move |s| (u.id, cosine(&s.vec, v))))
            .filter(|b| b.1 >= voice_threshold).max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(id, c)| (Who::Voice(id), c))
    }

    /// Adds a signature to unlabelled voice `id` (made if missing) heard in `meeting`.
    pub fn add_voice_sig(&mut self, id: u64, vec: Vec<f32>, secs: f32, meeting: &str, now: u64) {
        self.next_voice_id = self.next_voice_id.max(id + 1);
        let i = match self.unlabelled.iter().position(|u| u.id == id) {
            Some(i) => i,
            None => {
                self.unlabelled.push(Unlabelled { id, first_seen: now, ..Default::default() });
                self.unlabelled.len() - 1
            }
        };
        let u = &mut self.unlabelled[i];
        u.signatures.push(Signature { id: 0, vec, added: now.to_string(), meeting: meeting.to_owned(), seconds: secs });
        let extra = u.signatures.len().saturating_sub(MAX_VOICE_SIGS);
        u.signatures.drain(..extra);
        u.last_seen = now;
        if !u.meetings.iter().any(|m| m == meeting) {
            u.meetings.push(meeting.to_owned());
        }
    }

    /// Takes a signature back (a later embedding replaced it).
    pub fn retract_voice_sig(&mut self, id: u64, vec: &[f32]) {
        if let Some(u) = self.unlabelled.iter_mut().find(|u| u.id == id) {
            u.signatures.retain(|s| s.vec != vec);
        }
    }

    /// Forgets voices that never got a signature kept or audio (a first guess that was corrected).
    pub fn prune_empty(&mut self) {
        self.unlabelled.retain(|u| !u.signatures.is_empty() || u.sample_secs > 0.0 || u.seconds_heard > 0.0);
    }

    /// Drops unlabelled voices last heard more than `max_age` seconds before `now`; returns their ids.
    pub fn expire(&mut self, now: u64, max_age: u64) -> Vec<u64> {
        let gone: Vec<u64> = self.unlabelled.iter().filter(|u| now.saturating_sub(u.last_seen) > max_age).map(|u| u.id).collect();
        self.unlabelled.retain(|u| !gone.contains(&u.id));
        gone
    }

    /// Gives an unlabelled voice's signatures to the person called `name` (made if new); the voice is gone.
    /// Returns the person id and the removed voice.
    pub fn label_voice(&mut self, id: u64, name: &str) -> Option<(u64, Unlabelled)> {
        let name = name.trim();
        let i = self.unlabelled.iter().position(|u| u.id == id).filter(|_| !name.is_empty())?;
        if self.unlabelled[i].signatures.is_empty() {
            return None;
        }
        let u = self.unlabelled.remove(i);
        let mut pid = 0;
        for s in &u.signatures {
            pid = self.assign(name, s.vec.clone(), &s.meeting, s.seconds);
        }
        Some((pid, u))
    }

    /// Folds voice `other` into `keep`; returns `other` (its audio is the caller's to fold in).
    pub fn merge_voices(&mut self, keep: u64, other: u64) -> Option<Unlabelled> {
        if keep == other || !self.unlabelled.iter().any(|u| u.id == keep) {
            return None;
        }
        let o = self.unlabelled.remove(self.unlabelled.iter().position(|u| u.id == other)?);
        let k = self.unlabelled.iter_mut().find(|u| u.id == keep)?;
        k.signatures.extend(o.signatures.iter().cloned());
        let extra = k.signatures.len().saturating_sub(MAX_VOICE_SIGS);
        k.signatures.drain(..extra);
        for m in &o.meetings {
            if !k.meetings.contains(m) {
                k.meetings.push(m.clone());
            }
        }
        k.seconds_heard += o.seconds_heard;
        k.first_seen = k.first_seen.min(o.first_seen);
        k.last_seen = k.last_seen.max(o.last_seen);
        Some(o)
    }

    pub fn discard_voice(&mut self, id: u64) -> bool {
        let n = self.unlabelled.len();
        self.unlabelled.retain(|u| u.id != id);
        self.unlabelled.len() != n
    }

    pub fn delete_person(&mut self, id: u64) -> bool {
        let n = self.people.len();
        self.people.retain(|p| p.id != id);
        self.people.len() != n
    }

    pub fn delete_signature(&mut self, person: u64, sig: u64) -> bool {
        let Some(p) = self.people.iter_mut().find(|p| p.id == person) else { return false };
        let n = p.signatures.len();
        p.signatures.retain(|s| s.id != sig);
        let gone = p.signatures.len() != n;
        self.people.retain(|p| !p.signatures.is_empty()); // a person with no voice is nothing to match
        gone
    }

    pub fn rename_person(&mut self, id: u64, name: &str) -> bool {
        let name = name.trim();
        if name.is_empty() {
            return false;
        }
        self.people.iter_mut().find(|p| p.id == id).map(|p| p.name = name.to_owned()).is_some()
    }

    pub fn find(&self, name: &str) -> Option<&Person> {
        self.people.iter().find(|p| p.name.to_lowercase() == name.trim().to_lowercase())
    }
}

impl PeopleDb {
    /// A finished turn of voice `id`: counts its seconds and, while the sample has room, keeps its audio.
    pub fn hear(&mut self, dir: &Path, id: u64, audio: &[f32], heard_secs: f32, now: u64) -> Result<()> {
        let Some(u) = self.unlabelled.iter_mut().find(|u| u.id == id) else { return Ok(()) };
        u.last_seen = now;
        u.seconds_heard += heard_secs;
        let secs = audio.len() as f32 / SAMPLE_RATE as f32;
        if u.accepts(secs) {
            let mut all = read_wav(&sample_path(dir, id));
            all.extend_from_slice(audio);
            write_wav(&sample_path(dir, id), &all)?;
            u.sample_secs = all.len() as f32 / SAMPLE_RATE as f32;
        }
        Ok(())
    }

    /// Names voice `id`. Its audio is deleted, or with `keep_sample` moved to `person-<id>.wav`
    /// (ponytail: replacing an older kept sample of that person). Returns the person id.
    pub fn label_voice_files(&mut self, dir: &Path, id: u64, name: &str, keep_sample: bool) -> Result<Option<u64>> {
        let Some((pid, _)) = self.label_voice(id, name) else { return Ok(None) };
        let from = sample_path(dir, id);
        if keep_sample && from.exists() {
            std::fs::rename(&from, dir.join(format!("person-{pid}.wav")))?;
        } else {
            let _ = std::fs::remove_file(&from);
        }
        Ok(Some(pid))
    }

    /// Merges `other` into `keep`, joining their audio up to the cap.
    pub fn merge_voices_files(&mut self, dir: &Path, keep: u64, other: u64) -> Result<bool> {
        let Some(_) = self.merge_voices(keep, other) else { return Ok(false) };
        let (mut a, b) = (read_wav(&sample_path(dir, keep)), read_wav(&sample_path(dir, other)));
        if (a.len() + b.len()) as f32 <= SAMPLE_MAX_S * SAMPLE_RATE as f32 && !b.is_empty() {
            a.extend(b);
            write_wav(&sample_path(dir, keep), &a)?;
        }
        let _ = std::fs::remove_file(sample_path(dir, other));
        if let Some(k) = self.unlabelled.iter_mut().find(|u| u.id == keep) {
            k.sample_secs = read_wav(&sample_path(dir, keep)).len() as f32 / SAMPLE_RATE as f32;
        }
        Ok(true)
    }

    pub fn discard_voice_files(&mut self, dir: &Path, id: u64) -> bool {
        let _ = std::fs::remove_file(sample_path(dir, id));
        self.discard_voice(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(x: f32, y: f32) -> Vec<f32> {
        let n = (x * x + y * y).sqrt();
        vec![x / n, y / n]
    }

    #[test]
    fn cosine_basics() {
        assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 0.0]), 0.0);
    }

    #[test]
    fn best_match_picks_closest_signature_across_people() {
        let mut db = PeopleDb::default();
        db.assign("Ann", unit(1.0, 0.0), "m1", 5.0);
        db.assign("Bob", unit(0.0, 1.0), "m1", 5.0);
        let (_, name, c) = db.best_match(&unit(0.1, 1.0)).unwrap();
        assert_eq!(name, "Bob");
        assert!(c > 0.99);
        assert!(PeopleDb::default().best_match(&[1.0]).is_none());
    }

    #[test]
    fn person_matches_through_either_signature() {
        let mut db = PeopleDb::default();
        db.assign("Ann", unit(1.0, 0.0), "m1", 5.0);
        db.assign("ann", unit(0.0, 1.0), "m2", 5.0);
        assert_eq!(db.people.len(), 1);
        assert_eq!(db.best_match(&unit(1.0, 0.05)).unwrap().1, "Ann");
        assert_eq!(db.best_match(&unit(0.05, 1.0)).unwrap().1, "Ann");
    }

    #[test]
    fn below_threshold_still_reports_the_score() {
        let mut db = PeopleDb::default();
        db.assign("Ann", unit(1.0, 0.0), "m1", 5.0);
        let (_, _, c) = db.best_match(&unit(1.0, 1.0)).unwrap();
        assert!(c > 0.6 && c < 0.8, "{c}"); // the caller compares to its threshold and says "unknown"
    }

    #[test]
    fn assign_merges_by_name_and_ids_are_unique() {
        let mut db = PeopleDb::default();
        let a = db.assign("Ann", unit(1.0, 0.0), "m1", 5.0);
        let b = db.assign("Bob", unit(0.0, 1.0), "m1", 5.0);
        assert_eq!(db.assign(" ANN ", unit(1.0, 0.1), "m2", 4.0), a);
        assert_ne!(a, b);
        let mut ids: Vec<u64> = db.people.iter().flat_map(|p| std::iter::once(p.id).chain(p.signatures.iter().map(|s| s.id))).collect();
        let n = ids.len();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), n);
        assert_eq!(db.find("ann").unwrap().signatures.len(), 2);
    }

    #[test]
    fn edit_operations() {
        let mut db = PeopleDb::default();
        let a = db.assign("Ann", unit(1.0, 0.0), "m1", 5.0);
        db.assign("Ann", unit(1.0, 0.2), "m2", 5.0);
        let sig = db.people[0].signatures[0].id;
        assert!(db.rename_person(a, "Anna"));
        assert!(!db.rename_person(a, "  "));
        assert!(db.delete_signature(a, sig));
        assert_eq!(db.people[0].signatures.len(), 1);
        let last = db.people[0].signatures[0].id;
        assert!(db.delete_signature(a, last));
        assert!(db.people.is_empty(), "no signatures left, no person");
        let b = db.assign("Bob", unit(0.0, 1.0), "m", 1.0);
        assert!(db.delete_person(b) && !db.delete_person(b));
    }

    #[test]
    fn json_round_trip() {
        let mut db = PeopleDb { encoder: "enc.gguf".into(), ..Default::default() };
        db.assign("Ann", unit(1.0, 0.0), "m1", 5.5);
        let p = std::env::temp_dir().join(format!("rtc-people-{}.json", std::process::id()));
        db.save_to(&p).unwrap();
        assert_eq!(PeopleDb::load_from(&p), db);
        let _ = std::fs::remove_file(&p);
        assert_eq!(PeopleDb::load_from(&p), PeopleDb::default());
    }

    #[test]
    fn caps_signatures_dropping_the_oldest() {
        let mut db = PeopleDb::default();
        for i in 0..MAX_SIGNATURES + 5 {
            db.assign("Ann", unit(1.0, i as f32), &i.to_string(), 1.0);
        }
        let sigs = &db.people[0].signatures;
        assert_eq!(sigs.len(), MAX_SIGNATURES);
        assert_eq!(sigs[0].meeting, "5");
    }

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("rtc-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn identify_matches_people_then_unlabelled_voices() {
        let mut db = PeopleDb::default();
        db.assign("Ann", unit(1.0, 0.0), "m", 5.0);
        db.add_voice_sig(7, unit(0.0, 1.0), 3.0, "m", 100);
        assert_eq!(db.identify(&unit(1.0, 0.1), 0.6, 0.6).unwrap().0, Who::Person(db.people[0].id));
        assert_eq!(db.identify(&unit(0.1, 1.0), 0.6, 0.6).unwrap().0, Who::Voice(7));
        assert!(db.identify(&unit(-1.0, -1.0), 0.6, 0.6).is_none());
    }

    #[test]
    fn voice_ids_are_never_reused() {
        let mut db = PeopleDb::default();
        let a = db.alloc_voice_id();
        db.add_voice_sig(a, unit(1.0, 0.0), 1.0, "m", 1);
        db.discard_voice(a);
        assert!(db.alloc_voice_id() > a);
        db.add_voice_sig(50, unit(1.0, 0.0), 1.0, "m", 1);
        assert!(db.alloc_voice_id() > 50);
    }

    #[test]
    fn unlabelled_voices_expire_after_30_days_and_old_files_load() {
        let mut db = PeopleDb::default();
        db.add_voice_sig(1, unit(1.0, 0.0), 1.0, "m", 1_000);
        db.add_voice_sig(2, unit(0.0, 1.0), 1.0, "m", 1_000 + EXPIRE_SECS);
        assert_eq!(db.expire(1_000 + EXPIRE_SECS + 1, EXPIRE_SECS), vec![1]);
        assert_eq!(db.unlabelled.len(), 1);
        let old: PeopleDb = serde_json::from_str(r#"{"encoder":"e","people":[]}"#).unwrap();
        assert!(old.unlabelled.is_empty());
    }

    #[test]
    fn load_deletes_expired_audio() {
        let dir = tmp("expire");
        let p = dir.join("people.json");
        let mut db = PeopleDb::default();
        db.add_voice_sig(3, unit(1.0, 0.0), 1.0, "m", 5); // heard in 1970
        db.hear(&dir.join("voices"), 3, &[0.1; 16_000], 1.0, 5).unwrap();
        db.save_to(&p).unwrap();
        assert!(sample_path(&dir.join("voices"), 3).exists());
        assert!(PeopleDb::load_from(&p).unlabelled.is_empty());
        assert!(!sample_path(&dir.join("voices"), 3).exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn sample_keeps_whole_turns_up_to_60_s() {
        let dir = tmp("sample");
        let mut db = PeopleDb::default();
        db.add_voice_sig(1, unit(1.0, 0.0), 1.0, "m", 1);
        for _ in 0..4 {
            db.hear(&dir, 1, &vec![0.1; 12 * 16_000], 12.0, 2).unwrap();
        }
        assert!((db.unlabelled[0].sample_secs - 48.0).abs() < 0.01);
        db.hear(&dir, 1, &vec![0.1; 16_000], 1.0, 2).unwrap(); // under 2 s once there is a sample: skipped
        db.hear(&dir, 1, &vec![0.1; 13 * 16_000], 13.0, 2).unwrap(); // would pass 60 s: skipped
        assert!((db.unlabelled[0].sample_secs - 48.0).abs() < 0.01);
        assert_eq!(read_wav(&sample_path(&dir, 1)).len(), 48 * 16_000);
        assert!((db.unlabelled[0].seconds_heard - 62.0).abs() < 0.01, "but all heard time counts");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn labelling_moves_signatures_to_a_person_and_the_audio_goes() {
        let dir = tmp("label");
        let mut db = PeopleDb::default();
        let ann = db.assign("Ann", unit(1.0, 0.0), "m0", 5.0);
        db.add_voice_sig(4, unit(1.0, 0.2), 3.0, "m1", 1);
        db.add_voice_sig(4, unit(1.0, 0.3), 3.0, "m2", 2);
        db.hear(&dir, 4, &[0.1; 32_000], 2.0, 3).unwrap();
        assert_eq!(db.label_voice_files(&dir, 4, " ann ", false).unwrap(), Some(ann));
        assert_eq!(db.find("Ann").unwrap().signatures.len(), 3);
        assert!(db.unlabelled.is_empty() && !sample_path(&dir, 4).exists());
        // keep_sample moves the wav to the new person
        db.add_voice_sig(5, unit(0.0, 1.0), 3.0, "m1", 1);
        db.hear(&dir, 5, &[0.1; 32_000], 2.0, 3).unwrap();
        let bob = db.label_voice_files(&dir, 5, "Bob", true).unwrap().unwrap();
        assert!(dir.join(format!("person-{bob}.wav")).exists() && !sample_path(&dir, 5).exists());
        assert_eq!(db.label_voice_files(&dir, 99, "Zed", false).unwrap(), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn merging_and_discarding_voices() {
        let dir = tmp("merge");
        let mut db = PeopleDb::default();
        db.add_voice_sig(1, unit(1.0, 0.0), 1.0, "m1", 10);
        db.add_voice_sig(2, unit(1.0, 0.1), 1.0, "m2", 20);
        db.hear(&dir, 1, &[0.1; 32_000], 2.0, 10).unwrap();
        db.hear(&dir, 2, &[0.1; 48_000], 3.0, 20).unwrap();
        assert!(db.merge_voices_files(&dir, 1, 2).unwrap());
        let k = &db.unlabelled[0];
        assert_eq!((db.unlabelled.len(), k.signatures.len(), k.meetings.len()), (1, 2, 2));
        assert_eq!((k.first_seen, k.last_seen), (10, 20));
        assert!((k.sample_secs - 5.0).abs() < 0.01 && (k.seconds_heard - 5.0).abs() < 0.01);
        assert!(!sample_path(&dir, 2).exists());
        assert!(!db.merge_voices_files(&dir, 1, 1).unwrap());
        assert!(db.discard_voice_files(&dir, 1) && db.unlabelled.is_empty() && !sample_path(&dir, 1).exists());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
