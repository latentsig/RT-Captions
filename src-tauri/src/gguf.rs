//! Reads what a model file says about itself (GGUF header only: key/values and tensor types,
//! never the weights), so the engine picker can describe models it has no catalogue entry for.

use anyhow::{bail, Result};
use std::collections::HashMap;
use std::io::{BufReader, Read};
use std::path::Path;

#[derive(Debug, Clone, PartialEq)]
enum Val {
    Int(i64),
    Str(String),
    Strs(Vec<String>),
    Other,
}

#[derive(Debug, Default, Clone)]
pub struct Meta {
    /// A speech-to-text model (not a diarization / speaker / VAD model).
    pub is_asr: bool,
    /// Hybrid model with a CTC head (decoded with CTC by default).
    pub ctc: bool,
    /// Cache-aware streaming (fed piece by piece) rather than offline.
    pub streaming: bool,
    /// Future audio a streaming model waits for, in ms.
    pub lookahead_ms: Option<u32>,
    /// Dominant weight type, e.g. "q8_0", "q4_k", "f16", "ternary".
    pub quant: Option<String>,
    /// Languages the model can be prompted with (multilingual prompt models only).
    pub languages: Vec<String>,
}

fn rd<const N: usize>(r: &mut impl Read) -> Result<[u8; N]> {
    let mut b = [0u8; N];
    r.read_exact(&mut b)?;
    Ok(b)
}
fn u32_(r: &mut impl Read) -> Result<u32> { Ok(u32::from_le_bytes(rd(r)?)) }
fn u64_(r: &mut impl Read) -> Result<u64> { Ok(u64::from_le_bytes(rd(r)?)) }

fn string(r: &mut impl Read) -> Result<String> {
    let n = u64_(r)?;
    if n > 1 << 20 {
        bail!("implausible string length {n}");
    }
    let mut b = vec![0u8; n as usize];
    r.read_exact(&mut b)?;
    Ok(String::from_utf8_lossy(&b).into_owned())
}

/// Reads one value of GGUF type `t`; keeps ints, strings and string arrays, skips the rest.
fn value(r: &mut impl Read, t: u32) -> Result<Val> {
    Ok(match t {
        0 | 1 | 7 => Val::Int(rd::<1>(r)?[0] as i8 as i64), // u8, i8, bool
        2 => Val::Int(u16::from_le_bytes(rd(r)?) as i64),
        3 => Val::Int(i16::from_le_bytes(rd(r)?) as i64),
        4 => Val::Int(u32_(r)? as i64),
        5 => Val::Int(i32::from_le_bytes(rd(r)?) as i64),
        6 => { rd::<4>(r)?; Val::Other } // f32
        8 => Val::Str(string(r)?),
        9 => {
            let (et, n) = (u32_(r)?, u64_(r)?);
            if et == 8 {
                Val::Strs((0..n).map(|_| string(r)).collect::<Result<_>>()?)
            } else {
                for _ in 0..n {
                    value(r, et)?;
                }
                Val::Other
            }
        }
        10 | 11 => Val::Int(u64_(r)? as i64),
        12 => { rd::<8>(r)?; Val::Other } // f64
        _ => bail!("unknown GGUF value type {t}"),
    })
}

/// ggml tensor type -> short name (the ones speech models use).
fn type_name(t: u32) -> String {
    match t {
        0 => "f32", 1 => "f16", 2 => "q4_0", 3 => "q4_1", 6 => "q5_0", 7 => "q5_1", 8 => "q8_0", 9 => "q8_1",
        10 => "q2_k", 11 => "q3_k", 12 => "q4_k", 13 => "q5_k", 14 => "q6_k", 15 => "q8_k", 30 => "bf16",
        24 => "i8", 34 => "tq1_0", 35 => "tq2_0",
        other => return format!("type {other}"),
    }
    .into()
}

/// Bytes per element, roughly (for weighting which type dominates).
fn bytes_per(t: u32) -> f64 {
    match t {
        0 => 4.0, 1 | 30 => 2.0, 8 | 9 | 15 => 1.06, 2 | 3 | 12 => 0.56, 6 | 7 | 13 => 0.69, 14 => 0.82,
        10 | 11 => 0.4, 34 | 35 => 0.25, _ => 1.0,
    }
}

pub fn read(path: &Path) -> Result<Meta> {
    let mut r = BufReader::new(std::fs::File::open(path)?);
    if &rd::<4>(&mut r)? != b"GGUF" {
        bail!("not a GGUF file");
    }
    let version = u32_(&mut r)?;
    if version < 2 {
        bail!("GGUF v{version} is too old");
    }
    let (n_tensors, n_kv) = (u64_(&mut r)?, u64_(&mut r)?);
    let mut kv = HashMap::new();
    for _ in 0..n_kv {
        let key = string(&mut r)?;
        let t = u32_(&mut r)?;
        kv.insert(key, value(&mut r, t)?);
    }
    // Which non-f32 tensor type holds most of the weights = the quantization.
    let mut weight: HashMap<u32, f64> = HashMap::new();
    for _ in 0..n_tensors {
        string(&mut r)?;
        let dims = u32_(&mut r)?;
        let elems: f64 = (0..dims).map(|_| u64_(&mut r).map(|d| d as f64)).product::<Result<f64>>()?;
        let t = u32_(&mut r)?;
        u64_(&mut r)?; // offset
        if t != 0 {
            *weight.entry(t).or_default() += elems * bytes_per(t);
        }
    }
    let quant = weight.into_iter().max_by(|a, b| a.1.total_cmp(&b.1)).map(|(t, _)| type_name(t));

    let int = |k: &str| match kv.get(k) { Some(Val::Int(v)) => Some(*v), _ => None };
    let has_prefix = |p: &str| kv.keys().any(|k| k.starts_with(p));
    let head = match kv.get("parakeet.arch") { Some(Val::Str(s)) => s.clone(), _ => String::new() };
    let arch = match kv.get("general.architecture") { Some(Val::Str(s)) => s.clone(), _ => String::new() };
    let streaming = has_prefix("parakeet.streaming.");
    // Encoder frame = subsampling x hop (8 x 10 ms = 80 ms for FastConformer).
    let frame_ms = int("parakeet.encoder.subsampling_factor").unwrap_or(8)
        * int("parakeet.preprocessor.hop_length").unwrap_or(160) * 1000
        / int("parakeet.preprocessor.sample_rate").unwrap_or(16_000);
    let languages = match kv.get("parakeet.prompt.dictionary.keys") {
        // keys repeat each language in several spellings (en, en-US, ...): keep the short codes
        Some(Val::Strs(keys)) => {
            let mut v: Vec<String> = keys.iter().filter(|k| k.len() == 2).cloned().collect();
            v.dedup();
            v
        }
        _ => Vec::new(),
    };
    Ok(Meta {
        is_asr: arch == "parakeet" && !has_prefix("parakeet.diar.") && !has_prefix("voicedetect.") && !has_prefix("silero"),
        ctc: head.ends_with("_ctc"),
        streaming,
        lookahead_ms: streaming.then(|| int("parakeet.encoder.att_context_right")).flatten().map(|r| (r * frame_ms) as u32),
        // Moondream's packed ternary weights are stored as i8 tensors plus this marker.
        quant: if has_prefix("parakeet.ternary.") { Some("ternary".into()) } else { quant },
        languages,
    })
}

/// Ids of tokens that write non-Latin script: Devanagari pieces, plus the byte-fallback tokens
/// for non-ASCII bytes (otherwise Devanagari could still be spelled byte by byte). Suppressing
/// these locks an English/Hindi model to English (Latin) output.
pub fn non_latin_tokens(path: &Path) -> Result<Vec<i32>> {
    let mut r = BufReader::new(std::fs::File::open(path)?);
    if &rd::<4>(&mut r)? != b"GGUF" {
        bail!("not a GGUF file");
    }
    u32_(&mut r)?;
    let (_, n_kv) = (u64_(&mut r)?, u64_(&mut r)?);
    for _ in 0..n_kv {
        let key = string(&mut r)?;
        let t = u32_(&mut r)?;
        let v = value(&mut r, t)?;
        if key == "parakeet.tokenizer.pieces" {
            let Val::Strs(pieces) = v else { bail!("tokenizer pieces are not strings") };
            let byte = |p: &str| p.strip_prefix("<0x").and_then(|h| h.strip_suffix('>')).and_then(|h| u8::from_str_radix(h, 16).ok());
            return Ok(pieces.iter().enumerate()
                .filter(|(_, p)| p.chars().any(|c| ('\u{0900}'..='\u{097F}').contains(&c)) || byte(p).is_some_and(|b| b >= 0x80))
                .map(|(i, _)| i as i32)
                .collect());
        }
    }
    Ok(Vec::new())
}

/// "q8_0" from a file name like "my-model-q8_0.gguf", when the tensors don't say.
pub fn quant_from_name(file: &str) -> Option<String> {
    let lower = file.to_lowercase();
    ["q4_0", "q4_1", "q4_k", "q5_0", "q5_1", "q5_k", "q6_k", "q8_0", "f16", "bf16", "f32"]
        .into_iter()
        .find(|q| lower.contains(q))
        .map(String::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_a_minimal_gguf() {
        // header: magic, v3, 1 tensor, 3 kv
        let mut b = b"GGUF".to_vec();
        b.extend(3u32.to_le_bytes());
        b.extend(1u64.to_le_bytes());
        b.extend(3u64.to_le_bytes());
        let s = |b: &mut Vec<u8>, v: &str| { b.extend((v.len() as u64).to_le_bytes()); b.extend(v.as_bytes()); };
        s(&mut b, "general.architecture"); b.extend(8u32.to_le_bytes()); s(&mut b, "parakeet");
        s(&mut b, "parakeet.streaming.valid_out_len"); b.extend(5u32.to_le_bytes()); b.extend(4i32.to_le_bytes());
        s(&mut b, "parakeet.encoder.att_context_right"); b.extend(5u32.to_le_bytes()); b.extend(6i32.to_le_bytes());
        // tensor: name, 2 dims (512 x 512), type q8_0, offset 0
        s(&mut b, "enc.w"); b.extend(2u32.to_le_bytes()); b.extend(512u64.to_le_bytes()); b.extend(512u64.to_le_bytes());
        b.extend(8u32.to_le_bytes()); b.extend(0u64.to_le_bytes());
        let p = std::env::temp_dir().join("rt_captions_gguf_test.gguf");
        std::fs::write(&p, b).unwrap();
        let m = read(&p).unwrap();
        assert!(m.is_asr && m.streaming);
        assert_eq!(m.lookahead_ms, Some(480));
        assert_eq!(m.quant.as_deref(), Some("q8_0"));
        assert_eq!(quant_from_name("my-asr-la6-q8_0.gguf").as_deref(), Some("q8_0"));
    }
}
