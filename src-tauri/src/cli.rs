//! `rt-captions` command line: everything the GUI can do, scriptable by people and agents.

use crate::captioner::{Captioner, StreamConfig};
use crate::dsp::Dsp;
use crate::engine::{self, Event};
use crate::settings::{self, Settings};
use crate::asr::Asr;
use crate::engines::{self, Kind};
use crate::whistle::{Word, SAMPLE_RATE};
use anyhow::{bail, Context, Result};
use std::time::{Duration, Instant};

const HELP: &str = "rt-captions - real-time on-device captions

USAGE
  rt-captions                          open the app (overlay + settings)
  rt-captions settings                 print all settings as JSON
  rt-captions get <key>                print one setting, e.g. audio.stream.window_s
  rt-captions set <key> <value>...     change settings (key value pairs); a running app applies them live
  rt-captions reset                    restore default settings
  rt-captions listen [--source mic|system|both] [--seconds N] [--json]
                                       caption live audio to stdout, no window
  rt-captions transcribe <file.wav> [--json] [--window S] [--hop S] [--no-rnnoise]
                                       run a recording through the live pipeline
  rt-captions bench <file.wav> --ref <text|@file.txt> [--window 3,4,5] [--hop S] [--rnnoise|--no-rnnoise]
                                       accuracy (WER), latency and CPU per setting
  rt-captions engines                  list speech engines and which are downloaded
  rt-captions download <engine>        download an engine's model (e.g. ultra)
  rt-captions speed <file.wav> [--engine E] [--lens 2,3]
                                       per-call model time on clips of each length (rolling engines)
  rt-captions doctor                   engine, model, devices and settings paths
  rt-captions install-cli              put rt-captions on your PATH
  rt-captions uninstall-cli            take it off again

KEYS (rt-captions settings shows all)
  audio.engine  nemotron-q4|nemotron-q8|ultra|redux|whistle
  audio.mode  low|balanced|performance   audio.language  en|hi (Nemotron)
  audio.source  mic|system|both       audio.rnnoise  true|false        audio.keywords  \"Claude\\nFathom\"
  audio.stream.window_s  3            audio.stream.hop_s  0.25          audio.stream.finalize_s  2
  audio.stream.vad_threshold  0.008   overlay  true|false              auto_hide  true|false
  click_through  true|false           style.size  30   style.lines  3   style.fg  \"#ffffff\"
  style.bg  \"#101014\"   style.opacity  0.72   style.font  \"Segoe UI\"   style.hide_after_s  5
";

pub fn run(args: Vec<String>) -> Result<()> {
    let flag = |k: &str| args.iter().any(|a| a == k);
    let opt = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    match args.first().map(String::as_str) {
        None => launch_app(),
        Some("-h" | "--help" | "help") => Ok(print!("{HELP}")),
        Some("-V" | "--version") => Ok(println!("rt-captions {}", env!("CARGO_PKG_VERSION"))),
        Some("settings") => Ok(println!("{}", serde_json::to_string_pretty(&settings::load())?)),
        Some("get") => {
            let key = args.get(1).context("usage: rt-captions get <key>")?;
            Ok(println!("{}", settings::get(&settings::load(), key)?))
        }
        Some("set") => {
            let pairs = &args[1..];
            if pairs.is_empty() || pairs.len() % 2 != 0 {
                bail!("usage: rt-captions set <key> <value> [<key> <value>...]");
            }
            let mut s = settings::load();
            for kv in pairs.chunks(2) {
                s = settings::set(&s, &kv[0], &kv[1])?;
            }
            settings::save(&s)?;
            for kv in pairs.chunks(2) {
                println!("{} = {}", kv[0], settings::get(&s, &kv[0])?);
            }
            Ok(())
        }
        Some("reset") => {
            settings::save(&Settings::default())?;
            Ok(println!("settings reset: {}", settings::path().display()))
        }
        Some("listen") => listen(opt("--source"), opt("--seconds").map(|s| s.parse()).transpose()?, flag("--json")),
        Some("transcribe") => {
            let path = args.get(1).context("usage: rt-captions transcribe <file.wav>")?;
            let mut cfg = settings::load().audio.stream;
            apply_stream_flags(&mut cfg, &opt("--window"), &opt("--hop"))?;
            let rnnoise = rnnoise_flag(&args);
            let mut asr = load_asr(&opt)?;
            let run = run_file(&mut asr, path, &cfg, rnnoise)?;
            if flag("--json") {
                println!("{}", serde_json::to_string_pretty(&serde_json::json!({ "text": run.text(), "words": run.words, "lag_s": run.commit_lag }))?);
            } else {
                println!("{}", run.text());
            }
            Ok(())
        }
        Some("bench") => bench(&args, &opt),
        Some("doctor") => doctor(),
        Some("engines") => {
            let cur = settings::load().audio.engine;
            for e in engines::ENGINES {
                let mark = if e.id == cur { "*" } else { " " };
                let state = if engines::installed(e).is_some() { "downloaded" } else { "not downloaded" };
                println!("{mark} {:12} {:28} {:>5} MB  {:9}  {:15} {}", e.id, e.name, e.bytes / 1_000_000,
                         format!("{:?}", e.kind).to_lowercase(), state, e.blurb);
            }
            Ok(())
        }
        Some("download") => {
            let e = engines::get(args.get(1).context("usage: rt-captions download <engine>")?)?;
            let mut last = u64::MAX;
            let path = engines::download(e, |done, total| {
                let pct = done * 100 / total.max(1);
                if pct != last {
                    eprint!("
{} {pct}%   ", e.name);
                    last = pct;
                }
            })?;
            eprintln!();
            Ok(println!("{}", path.display()))
        }
        Some("speed") => speed(&args, &opt),
        Some("install-cli") => install_cli(true),
        Some("uninstall-cli") => install_cli(false),
        Some(other) => bail!("unknown command '{other}'. Try rt-captions --help"),
    }
}

fn rnnoise_flag(args: &[String]) -> bool {
    if args.iter().any(|a| a == "--no-rnnoise") {
        false
    } else {
        args.iter().any(|a| a == "--rnnoise") || settings::load().audio.rnnoise
    }
}

fn apply_stream_flags(cfg: &mut StreamConfig, window: &Option<String>, hop: &Option<String>) -> Result<()> {
    if let Some(w) = window {
        cfg.window_s = w.parse()?;
    }
    if let Some(h) = hop {
        cfg.hop_s = h.parse()?;
    }
    Ok(())
}

fn exe_dir() -> Result<std::path::PathBuf> {
    Ok(std::env::current_exe()?.parent().context("exe dir")?.to_path_buf())
}

fn launch_app() -> Result<()> {
    let name = if cfg!(windows) { "rt-captions-app.exe" } else { "rt-captions-app" };
    let app = exe_dir()?.join(name);
    std::process::Command::new(&app).spawn().with_context(|| format!("starting {}", app.display()))?;
    Ok(())
}

fn listen(source: Option<String>, seconds: Option<f64>, json: bool) -> Result<()> {
    use std::io::Write;
    let mut audio = settings::load().audio;
    if let Some(s) = source {
        audio.source = s;
    }
    let labeled = audio.source == "both";
    let (tx, rx) = std::sync::mpsc::channel();
    let engine = engine::spawn(move |ev| {
        let _ = tx.send(ev);
    });
    engine.send(audio)?;
    eprintln!("listening (ctrl+c to stop)...");
    let deadline = seconds.map(|s| Instant::now() + Duration::from_secs_f64(s));
    let mut last_source = "";
    let mut out = std::io::stdout();
    loop {
        let wait = deadline.map_or(Duration::from_secs(3600), |d| d.saturating_duration_since(Instant::now()));
        if wait.is_zero() {
            break;
        }
        match rx.recv_timeout(wait) {
            Ok(Event::Caption(c)) if json => writeln!(out, "{}", serde_json::to_string(&c)?)?,
            Ok(Event::Caption(c)) => {
                if !c.committed.is_empty() {
                    if labeled && c.source != last_source {
                        write!(out, "\n[{}] ", if c.source == "mic" { "You" } else { "Them" })?;
                        last_source = c.source;
                    }
                    write!(out, "{} ", c.committed)?;
                }
                if c.end_of_utterance {
                    writeln!(out)?;
                    last_source = "";
                }
                out.flush()?;
            }
            Ok(Event::Status(s)) if !s.is_empty() => bail!(s),
            Ok(Event::Status(_)) => {}
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => break,
            Err(_) => bail!("engine stopped"),
        }
    }
    writeln!(out)?;
    Ok(())
}

pub struct FileRun {
    pub words: Vec<Word>,
    pub passes_ms: Vec<f64>,
    /// Seconds from a word's end to it being committed, with passes taking real time.
    pub commit_lag: Vec<f64>,
    pub busy_s: f64,
    pub duration_s: f64,
}

impl FileRun {
    pub fn text(&self) -> String {
        crate::captioner::join(&self.words)
    }
}

/// Feeds a WAV through dsp + captioner as if it arrived live: audio comes in on a
/// simulated clock, and every pass holds the engine for its real wall time.
/// Engine for file commands: --engine/--threads, else the saved settings.
fn load_asr(opt: &dyn Fn(&str) -> Option<String>) -> Result<Asr> {
    let s = settings::load().audio;
    let engine = opt("--engine").unwrap_or(s.engine.clone());
    let threads = opt("--threads").map(|t| t.parse()).transpose()?.unwrap_or(engine::effective_threads(&s));
    let mut asr = Asr::load(&engine, threads)?;
    asr.set_keywords(&s.keywords);
    asr.set_language(&opt("--language").unwrap_or(s.language.clone()));
    Ok(asr)
}

pub fn run_file(whistle: &mut Asr, path: &str, cfg: &StreamConfig, rnnoise: bool) -> Result<FileRun> {
    let mut reader = hound::WavReader::open(path).with_context(|| format!("opening {path}"))?;
    let spec = reader.spec();
    let raw: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = (1u32 << (spec.bits_per_sample - 1)) as f32;
            reader.samples::<i32>().map(|s| s.map(|s| s as f32 / scale)).collect::<Result<_, _>>()?
        }
    };
    let ch = spec.channels as usize;
    let mono: Vec<f32> = raw.chunks(ch).map(|f| f.iter().sum::<f32>() / ch as f32).collect();
    let audio = Dsp::new(spec.sample_rate as usize, rnnoise).process(mono);
    let duration_s = audio.len() as f64 / SAMPLE_RATE as f64;

    let mut run = FileRun { words: Vec::new(), passes_ms: Vec::new(), commit_lag: Vec::new(), busy_s: 0.0, duration_s };
    if whistle.kind() == Kind::Streaming {
        // Streaming: 160 ms blocks arrive in real time; each feed holds the engine for its wall time.
        whistle.reset_streams();
        let mut clock = 0.0f64;
        let block = 2560;
        let pieces = audio.chunks(block).count();
        for (k, piece) in audio.chunks(block).chain(std::iter::once(&[0.0f32; 19_200][..])).enumerate() {
            clock = clock.max(((k + 1) * block).min(audio.len() + 19_200) as f64 / SAMPLE_RATE as f64);
            let t = Instant::now();
            let (_, mut words) = whistle.stream_feed(0, piece)?;
            if k == pieces {
                words.extend(whistle.stream_finish(0)?.1);
            }
            let el = t.elapsed().as_secs_f64();
            run.passes_ms.push(el * 1000.0);
            run.busy_s += el;
            clock += el;
            for w in words {
                run.commit_lag.push(clock - w.end);
                run.words.push(w);
            }
        }
        return Ok(run);
    }
    let mut cap = Captioner::new(cfg.clone());
    let (mut clock, mut fed, tick) = (0.0f64, 0usize, 160usize);
    while fed < audio.len() || cap.is_active() {
        let avail = ((clock * SAMPLE_RATE as f64) as usize).min(audio.len());
        if avail > fed {
            cap.feed(&audio[fed..avail]);
            fed = avail;
        } else if fed >= audio.len() {
            cap.feed(&[0.0; 160]); // trailing silence flushes the tail
        }
        let t = Instant::now();
        let mut ran = false;
        let update = cap.step(|b| {
            ran = true;
            whistle.transcribe(b).unwrap_or_default()
        });
        if ran {
            let el = t.elapsed().as_secs_f64();
            run.passes_ms.push(el * 1000.0);
            run.busy_s += el;
            clock += el;
        } else {
            clock += tick as f64 / SAMPLE_RATE as f64;
        }
        if let Some(u) = update {
            for w in u.committed {
                run.commit_lag.push(clock - w.end);
                run.words.push(w);
            }
        }
    }
    Ok(run)
}

pub fn norm_words(s: &str) -> Vec<String> {
    s.split_whitespace()
        .map(|w| w.chars().filter(|c| c.is_alphanumeric()).flat_map(char::to_lowercase).collect::<String>())
        .filter(|w| !w.is_empty())
        .collect()
}

pub fn wer(reference: &[String], hyp: &[String]) -> f64 {
    let mut row: Vec<usize> = (0..=hyp.len()).collect();
    for (i, r) in reference.iter().enumerate() {
        let mut prev = row[0];
        row[0] = i + 1;
        for (j, h) in hyp.iter().enumerate() {
            let cur = row[j + 1];
            row[j + 1] = (prev + (r != h) as usize).min(row[j] + 1).min(cur + 1);
            prev = cur;
        }
    }
    row[hyp.len()] as f64 / reference.len().max(1) as f64
}

fn pct(v: &[f64], p: f64) -> f64 {
    let mut v = v.to_vec();
    v.sort_by(f64::total_cmp);
    v.get(((v.len().max(1) - 1) as f64 * p).round() as usize).copied().unwrap_or(0.0)
}

fn bench(args: &[String], opt: &dyn Fn(&str) -> Option<String>) -> Result<()> {
    let path = args.get(1).context("usage: rt-captions bench <file.wav> --ref <text|@file>")?;
    let reference = match opt("--ref") {
        Some(r) if r.starts_with('@') => Some(std::fs::read_to_string(&r[1..])?),
        r => r,
    };
    let reference = reference.map(|r| norm_words(&r));
    let rnnoise = rnnoise_flag(args);
    let base = settings::load().audio.stream;
    let windows: Vec<String> = opt("--window").map_or(vec![base.window_s.to_string()], |w| w.split(',').map(String::from).collect());
    let mut asr = load_asr(opt)?;
    println!("{path}  engine={}  rnnoise={rnnoise}", opt("--engine").unwrap_or(settings::load().audio.engine));
    let streaming = asr.kind() == Kind::Streaming;
    if streaming {
        println!("(streaming engine: window/hop do not apply; 'pass' is one 160 ms feed)");
    }
    println!("{:>7} {:>6} {:>14} {:>16} {:>6} {:>7}", "window", "hop", "pass p50/p95", "solid p50/p95", "busy", "WER");
    for w in windows {
        let mut cfg = base.clone();
        apply_stream_flags(&mut cfg, &Some(w), &opt("--hop"))?;
        let run = run_file(&mut asr, path, &cfg, rnnoise)?;
        let wer = reference.as_ref().map_or("-".into(), |r| format!("{:.1}%", 100.0 * wer(r, &norm_words(&run.text()))));
        println!(
            "{:>6}s {:>5}s {:>6.0}/{:<4.0}ms {:>6.2}/{:<5.2}s {:>5.0}% {:>7}",
            cfg.window_s, cfg.hop_s, pct(&run.passes_ms, 0.5), pct(&run.passes_ms, 0.95),
            pct(&run.commit_lag, 0.5), pct(&run.commit_lag, 0.95), 100.0 * run.busy_s / run.duration_s, wer
        );
        if args.iter().any(|a| a == "--text") {
            println!("  {}", run.text());
        }
    }
    Ok(())
}

/// Per-call latency of the model on clips of several lengths (what a live pass costs).
fn speed(args: &[String], opt: &dyn Fn(&str) -> Option<String>) -> Result<()> {
    let path = args.get(1).context("usage: rt-captions speed <file.wav> [--engine E] [--threads N] [--lens 2,3,4]")?;
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let raw: Vec<f32> = reader.samples::<i16>().map(|s| s.map(|s| s as f32 / 32768.0)).collect::<Result<_, _>>()?;
    let mono: Vec<f32> = raw.chunks(spec.channels as usize).map(|f| f.iter().sum::<f32>() / f.len() as f32).collect();
    let audio = Dsp::new(spec.sample_rate as usize, false).process(mono);
    let mut asr = load_asr(opt)?;
    let lens: Vec<f64> = opt("--lens").unwrap_or("2,3,4,5".into()).split(',').map(|l| l.parse()).collect::<Result<_, _>>()?;
    asr.transcribe(&audio[..SAMPLE_RATE * 3])?; // warm-up
    for len in lens {
        let n = (len * SAMPLE_RATE as f64) as usize;
        let mut ms: Vec<f64> = (0..30)
            .map(|i| {
                let start = (i * SAMPLE_RATE * 7) % (audio.len() - n);
                let t = Instant::now();
                asr.transcribe(&audio[start..start + n]).unwrap();
                t.elapsed().as_secs_f64() * 1000.0
            })
            .collect();
        ms.sort_by(f64::total_cmp);
        println!("{len:>4}s clip: median {:>5.0} ms  p95 {:>5.0} ms", ms[15], ms[28]);
    }
    Ok(())
}

fn doctor() -> Result<()> {
    println!("rt-captions {}", env!("CARGO_PKG_VERSION"));
    println!("settings: {}", settings::path().display());
    println!("models:   {}", engines::models_dir().display());
    let id = settings::load().audio.engine;
    println!("engine:   {id}");
    let t = Instant::now();
    match Asr::load(&id, 0) {
        Ok(_) => println!("loads OK in {:.0} ms", t.elapsed().as_secs_f64() * 1000.0),
        Err(e) => println!("NOT READY - {e:#}"),
    }
    for d in crate::audio::describe() {
        println!("{d}");
    }
    Ok(())
}

fn install_cli(install: bool) -> Result<()> {
    let dir = exe_dir()?;
    #[cfg(windows)]
    {
        // User PATH via .NET so we don't truncate it the way `setx` does.
        let d = dir.display().to_string().replace('\'', "''");
        let script = if install {
            format!("$p=[Environment]::GetEnvironmentVariable('Path','User'); if (-not (($p -split ';') -contains '{d}')) {{ [Environment]::SetEnvironmentVariable('Path', (($p.TrimEnd(';'), '{d}') -join ';').TrimStart(';'), 'User') }}")
        } else {
            format!("$p=[Environment]::GetEnvironmentVariable('Path','User'); [Environment]::SetEnvironmentVariable('Path', (($p -split ';' | Where-Object {{ $_ -and $_ -ne '{d}' }}) -join ';'), 'User')")
        };
        let ok = std::process::Command::new("powershell").args(["-NoProfile", "-Command", &script]).status()?.success();
        if !ok {
            bail!("could not update the user PATH");
        }
        println!("{} {} {} PATH. Open a new terminal to use `rt-captions`.", if install { "Added" } else { "Removed" }, dir.display(), if install { "to" } else { "from" });
    }
    #[cfg(not(windows))]
    {
        let exe = std::env::current_exe()?;
        let home = std::path::PathBuf::from(std::env::var("HOME")?);
        let candidates = [std::path::PathBuf::from("/usr/local/bin"), home.join(".local/bin")];
        if install {
            for bin in candidates {
                let _ = std::fs::create_dir_all(&bin);
                let link = bin.join("rt-captions");
                let _ = std::fs::remove_file(&link);
                if std::os::unix::fs::symlink(&exe, &link).is_ok() {
                    println!("Linked {} -> {}", link.display(), exe.display());
                    return Ok(());
                }
            }
            bail!("no writable bin directory; run: sudo ln -s '{}' /usr/local/bin/rt-captions", exe.display());
        }
        for bin in candidates {
            let _ = std::fs::remove_file(bin.join("rt-captions"));
        }
        println!("Removed rt-captions links");
        let _ = dir;
    }
    Ok(())
}
