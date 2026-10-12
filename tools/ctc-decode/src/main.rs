//! Offline scorer for the app's CTC decoder (src-tauri/src/hotwords.rs, compiled in unchanged).
//!
//! ctc-decode --vocab pieces.txt [--beam 8] [--blank-penalty 0.5] [--hotwords a,b|@file]
//!            [--chunk FRAMES] [--classes 2048] <file.npy|.f32|.f16|dir|@list>...
//! Input: CTC LOGITS [T, V], blank last; .npy (<f4/<f2, C order) or raw little-endian .f32/.f16.
//! Output: one "<file>\t<text>" line per input on stdout.

#[path = "../../../src-tauri/src/hotwords.rs"]
#[allow(dead_code)]
mod hotwords;
mod asr {
    pub struct Word {
        pub word: String,
        #[allow(dead_code)]
        pub start: f64,
        #[allow(dead_code)]
        pub end: f64,
    }
}

use hotwords::{Decoder, Hotwords, Vocab};
use std::path::{Path, PathBuf};
use std::sync::Arc;

type Res<T> = Result<T, String>;

fn f16(h: u16) -> f32 {
    let (s, e, m) = ((h >> 15) as u32, ((h >> 10) & 0x1f) as u32, (h & 0x3ff) as u32);
    let bits = match e {
        0 if m == 0 => s << 31,
        0 => return if s == 1 { -1.0 } else { 1.0 } * m as f32 * 2f32.powi(-24),
        31 => (s << 31) | 0x7f80_0000 | (m << 13),
        _ => (s << 31) | ((e + 112) << 23) | (m << 13),
    };
    f32::from_bits(bits)
}

fn floats(b: &[u8], half: bool) -> Vec<f32> {
    if half {
        b.chunks_exact(2).map(|c| f16(u16::from_le_bytes([c[0], c[1]]))).collect()
    } else {
        b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    }
}

/// (values, classes)
fn load(p: &Path, classes: usize) -> Res<(Vec<f32>, usize)> {
    let b = std::fs::read(p).map_err(|e| format!("{}: {e}", p.display()))?;
    match p.extension().and_then(|e| e.to_str()) {
        Some("npy") => {
            if !b.starts_with(b"\x93NUMPY") {
                return Err(format!("{}: not a .npy", p.display()));
            }
            let (len, off) = if b[6] == 1 { (u16::from_le_bytes([b[8], b[9]]) as usize, 10) } else { (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12) };
            let hdr = std::str::from_utf8(&b[off..off + len]).map_err(|e| e.to_string())?;
            let half = if hdr.contains("'<f2'") { true } else if hdr.contains("'<f4'") { false } else { return Err(format!("{}: need <f4 or <f2: {hdr}", p.display())) };
            if hdr.contains("'fortran_order': True") {
                return Err(format!("{}: fortran order", p.display()));
            }
            let shape = hdr.split("'shape': (").nth(1).and_then(|s| s.split(')').next()).ok_or("no shape")?;
            let dims: Vec<usize> = shape.split(',').filter_map(|d| d.trim().parse().ok()).collect();
            let v = *dims.last().ok_or("empty shape")?;
            Ok((floats(&b[off + len..], half), v))
        }
        Some("f32") => Ok((floats(&b, false), classes)),
        Some("f16") => Ok((floats(&b, true), classes)),
        _ => Err(format!("{}: want .npy, .f32 or .f16", p.display())),
    }
}

/// Same scores the engine hands the app: log-softmax, then the blank penalty on the blank log-prob.
fn log_probs(x: &mut [f32], classes: usize, pen: f32) {
    for row in x.chunks_exact_mut(classes) {
        let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let z = m + row.iter().map(|v| (v - m).exp()).sum::<f32>().ln();
        row.iter_mut().for_each(|v| *v -= z);
        row[classes - 1] -= pen;
    }
}

fn decode(p: &Path, vocab: &Arc<Vocab>, hot: &Arc<Hotwords>, beam: usize, pen: f32, chunk: usize, classes: usize) -> Res<String> {
    let (mut x, v) = load(p, classes)?;
    if x.len() % v != 0 {
        return Err(format!("{}: {} values not a multiple of {v} classes", p.display(), x.len()));
    }
    log_probs(&mut x, v, pen);
    let mut d = Decoder::new(vocab.clone(), hot.clone(), beam);
    let mut words = Vec::new();
    let step = if chunk == 0 { x.len().max(1) } else { chunk * v };
    for c in x.chunks(step) {
        words.extend(d.feed(c, v).0);
    }
    words.extend(d.finish());
    Ok(words.iter().map(|w| w.word.as_str()).collect::<Vec<_>>().join(" "))
}

fn inputs(args: &[String]) -> Res<Vec<PathBuf>> {
    let mut out = Vec::new();
    for a in args {
        if let Some(list) = a.strip_prefix('@') {
            let t = std::fs::read_to_string(list).map_err(|e| format!("{list}: {e}"))?;
            out.extend(t.lines().map(str::trim).filter(|l| !l.is_empty()).map(PathBuf::from));
        } else if Path::new(a).is_dir() {
            let mut v: Vec<PathBuf> = std::fs::read_dir(a).map_err(|e| e.to_string())?.filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| matches!(p.extension().and_then(|e| e.to_str()), Some("npy" | "f32" | "f16"))).collect();
            v.sort();
            out.extend(v);
        } else {
            out.push(PathBuf::from(a));
        }
    }
    Ok(out)
}

fn main() -> Res<()> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut opt = |k: &str| args.iter().position(|a| a == k).map(|i| { let v = args.remove(i + 1); args.remove(i); v });
    let vocab_path = opt("--vocab").ok_or("--vocab pieces.txt (one SentencePiece piece per line, id order, no blank)")?;
    let num = |v: Option<String>, d: f64| v.map(|s| s.parse::<f64>().map_err(|e| format!("{s}: {e}"))).transpose().map(|v| v.unwrap_or(d));
    let beam = num(opt("--beam"), 8.0)? as usize;
    let pen = num(opt("--blank-penalty"), 0.5)? as f32;
    let chunk = num(opt("--chunk"), 0.0)? as usize;
    let classes = num(opt("--classes"), 2048.0)? as usize;
    let hot: Vec<String> = match opt("--hotwords") {
        Some(h) if h.starts_with('@') => std::fs::read_to_string(&h[1..]).map_err(|e| e.to_string())?.lines().map(String::from).collect(),
        Some(h) => h.split(',').map(String::from).collect(),
        None => Vec::new(),
    };
    let pieces: Vec<String> = std::fs::read_to_string(&vocab_path).map_err(|e| format!("{vocab_path}: {e}"))?
        .lines().map(|l| l.split('\t').next().unwrap_or("").to_string()).collect();
    let (vocab, hot) = (Arc::new(Vocab::new(&pieces)), Arc::new(Hotwords::new(&hot)));
    let files = inputs(&args)?;
    if files.is_empty() {
        return Err("no input files".into());
    }
    // Files are independent: one worker per core pulling the next index; printed in input order.
    let n = std::thread::available_parallelism().map_or(1, |n| n.get()).min(files.len());
    let next = std::sync::atomic::AtomicUsize::new(0);
    let mut res: Vec<(usize, Res<String>)> = std::thread::scope(|s| {
        let workers: Vec<_> = (0..n).map(|_| s.spawn(|| {
            let mut got = Vec::new();
            loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let Some(f) = files.get(i) else { break };
                got.push((i, decode(f, &vocab, &hot, beam, pen, chunk, classes)));
            }
            got
        })).collect();
        workers.into_iter().flat_map(|w| w.join().unwrap()).collect()
    });
    res.sort_by_key(|r| r.0);
    let mut failed = 0;
    for (f, (_, r)) in files.iter().zip(res) {
        match r {
            Ok(t) => println!("{}\t{t}", f.display()),
            Err(e) => { eprintln!("error: {e}"); failed += 1; }
        }
    }
    if failed > 0 { Err(format!("{failed} file(s) failed")) } else { Ok(()) }
}
