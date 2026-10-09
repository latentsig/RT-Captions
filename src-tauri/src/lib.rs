pub mod asr;
pub mod captioner;
pub mod cli;
pub mod dsp;
pub mod engine;
pub mod engines;
pub mod gguf;
pub mod parakeet;
pub mod people;
pub mod settings;
pub mod speakers;
mod audio;

use engine::{EngineHandle, Event};
use settings::Settings;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};
use tauri::menu::{CheckMenuItem, Menu, MenuItem, PredefinedMenuItem};
use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
use tauri::{AppHandle, Emitter, Manager, State, Wry};

struct AppState {
    settings: Mutex<Settings>,
    engine: Mutex<EngineHandle>,
    status: Arc<Mutex<String>>,
    overlay_item: CheckMenuItem<Wry>,
}

/// Push settings everywhere they matter. Called for UI saves, CLI edits and tray toggles.
fn apply(app: &AppHandle, new: Settings) {
    let state = app.state::<AppState>();
    let mut cur = state.settings.lock().unwrap();
    if cur.audio != new.audio {
        let _ = state.engine.lock().unwrap().send(new.audio.clone());
    }
    if let Some(overlay) = app.get_webview_window("overlay") {
        let _ = overlay.set_ignore_cursor_events(new.click_through);
    }
    let _ = state.overlay_item.set_checked(new.overlay);
    let _ = app.emit("settings", &new);
    *cur = new;
}

/// RT_CAPTIONS_DEBUG: the overlay reports each caption once painted (timing probe).
#[tauri::command]
fn debug_enabled() -> bool {
    std::env::var_os("RT_CAPTIONS_DEBUG").is_some()
}

#[tauri::command]
fn debug_ack(text: String) {
    engine::debug_log(format_args!("overlay painted: {text:?}"));
}

#[tauri::command]
fn get_settings(state: State<AppState>) -> Settings {
    state.settings.lock().unwrap().clone()
}

#[tauri::command]
fn get_status(state: State<AppState>) -> String {
    state.status.lock().unwrap().clone()
}

#[tauri::command]
fn save_settings(app: AppHandle, settings: Settings) -> Result<(), String> {
    settings::save(&settings).map_err(|e| format!("{e:#}"))?;
    apply(&app, settings);
    Ok(())
}

#[tauri::command]
fn list_engines() -> Vec<engines::Model> {
    engines::list()
}

#[derive(serde::Serialize, Clone)]
struct DownloadProgress {
    id: &'static str,
    done: u64,
    total: u64,
    finished: bool,
    error: Option<String>,
}

/// Downloads an engine's model in the background, reporting "download" events; when it
/// lands, the engine thread is poked so a waiting session loads it.
#[tauri::command]
fn download_engine(app: AppHandle, id: String) -> Result<(), String> {
    let e = engines::get_any(&id).map_err(|e| format!("{e:#}"))?;
    std::thread::spawn(move || {
        let progress = |done: u64, total: u64| {
            let _ = app.emit("download", DownloadProgress { id: e.id, done, total, finished: false, error: None });
            let _ = app.emit("status", format!("Downloading {} speech model… {}%", e.name, done * 100 / total.max(1)));
        };
        let result = engines::download(e, progress);
        let error = result.as_ref().err().map(|err| format!("{err:#}"));
        let _ = app.emit("download", DownloadProgress { id: e.id, done: e.bytes, total: e.bytes, finished: true, error });
        if result.is_ok() {
            let state = app.state::<AppState>();
            let audio = state.settings.lock().unwrap().audio.clone();
            let _ = state.engine.lock().unwrap().send(audio);
        }
    });
    Ok(())
}

/// Names a voice of this meeting and saves its signature, so later meetings recognise it.
#[tauri::command]
fn assign_voice(state: State<AppState>, key: String, name: String) {
    state.engine.lock().unwrap().assign_voice(key, name);
}

/// Old name of `assign_voice`, taking a caption's speaker (turn id).
#[tauri::command]
fn rename_speaker(state: State<AppState>, slot: usize, name: String) {
    assign_voice(state, format!("turn:{slot}"), name);
}

/// Voiceprint glyph data: the mean of `sigs`, downsampled to 24 values (mean of each 1/24 chunk).
fn print_of<'a>(sigs: impl Iterator<Item = &'a people::Signature>) -> Vec<f32> {
    let mut mean: Vec<f32> = Vec::new();
    let mut n = 0f32;
    for s in sigs.filter(|s| !s.vec.is_empty()) {
        if mean.is_empty() {
            mean = vec![0.0; s.vec.len()];
        }
        if s.vec.len() != mean.len() {
            continue;
        }
        for (m, x) in mean.iter_mut().zip(&s.vec) {
            *m += x;
        }
        n += 1.0;
    }
    if n == 0.0 {
        return Vec::new();
    }
    let len = mean.len();
    (0..24).map(|i| {
        let chunk = &mean[i * len / 24..((i + 1) * len / 24).max(i * len / 24 + 1).min(len)];
        chunk.iter().sum::<f32>() / chunk.len().max(1) as f32 / n
    }).collect()
}

#[derive(serde::Serialize)]
struct LiveVoice {
    #[serde(flatten)]
    info: speakers::VoiceInfo,
    /// Voiceprint (24 values), empty when no signature is saved for this voice yet.
    print: Vec<f32>,
}

#[tauri::command]
fn list_voices(state: State<AppState>) -> Vec<LiveVoice> {
    let voices = state.engine.lock().unwrap().list_voices();
    let db = people::PeopleDb::load();
    voices.into_iter().map(|info| {
        let print = match info.key.split_once(':') {
            Some(("person", id)) => id.parse::<u64>().ok().and_then(|id| db.people.iter().find(|p| p.id == id)).map(|p| print_of(p.signatures.iter())),
            Some(("voice", id)) => id.parse::<u64>().ok().and_then(|id| db.unlabelled.iter().find(|u| u.id == id)).map(|u| print_of(u.signatures.iter())),
            _ => None,
        };
        LiveVoice { info, print: print.unwrap_or_default() }
    }).collect()
}

/// A voice's playback sample as 16 kHz mono 16-bit WAV bytes.
#[tauri::command]
fn voice_sample(state: State<AppState>, key: String) -> Result<Vec<u8>, String> {
    let pcm = state.engine.lock().unwrap().voice_sample(key);
    people::wav_bytes(&pcm).map_err(|e| format!("{e:#}"))
}

#[derive(serde::Serialize)]
struct SigInfo {
    id: u64,
    added: String,
    meeting: String,
    seconds: f32,
}

#[derive(serde::Serialize)]
struct PersonInfo {
    id: u64,
    name: String,
    signatures: Vec<SigInfo>,
    /// Voiceprint glyph data: 24 values.
    print: Vec<f32>,
}

/// Saved people, without their vectors.
#[tauri::command]
fn list_people() -> Vec<PersonInfo> {
    people::PeopleDb::load().people.into_iter().map(|p| PersonInfo {
        id: p.id,
        print: print_of(p.signatures.iter()),
        name: p.name,
        signatures: p.signatures.into_iter().map(|s| SigInfo { id: s.id, added: s.added, meeting: s.meeting, seconds: s.seconds }).collect(),
    }).collect()
}

/// Edits the people file, then tells the speaker thread to re-read it.
fn edit_people(state: &State<AppState>, edit: impl FnOnce(&mut people::PeopleDb) -> bool) -> Result<(), String> {
    let mut db = people::PeopleDb::load();
    if edit(&mut db) {
        db.save().map_err(|e| format!("{e:#}"))?;
        state.engine.lock().unwrap().reload_people();
    }
    Ok(())
}

#[tauri::command]
fn delete_person(state: State<AppState>, id: u64) -> Result<(), String> {
    edit_people(&state, |db| db.delete_person(id))
}

#[tauri::command]
fn delete_signature(state: State<AppState>, person_id: u64, signature_id: u64) -> Result<(), String> {
    edit_people(&state, |db| db.delete_signature(person_id, signature_id))
}

#[tauri::command]
fn rename_person(state: State<AppState>, id: u64, name: String) -> Result<(), String> {
    edit_people(&state, |db| db.rename_person(id, &name))
}

#[derive(serde::Serialize)]
struct BestMatch {
    name: String,
    score: f32,
}

#[derive(serde::Serialize)]
struct InboxVoice {
    id: u64,
    first_seen: u64,
    last_seen: u64,
    /// Number of meetings.
    meetings: usize,
    seconds_heard: f32,
    sample_secs: f32,
    best_match: Option<BestMatch>,
    /// Voiceprint glyph data: 24 values.
    print: Vec<f32>,
}

/// Voices nobody has named yet, newest first, each with the saved person it sounds most like.
#[tauri::command]
fn list_inbox() -> Vec<InboxVoice> {
    let db = people::PeopleDb::load();
    let mut v: Vec<InboxVoice> = db.unlabelled.iter().map(|u| InboxVoice {
        id: u.id,
        first_seen: u.first_seen,
        last_seen: u.last_seen,
        meetings: u.meetings.len(),
        seconds_heard: u.seconds_heard,
        sample_secs: u.sample_secs,
        print: print_of(u.signatures.iter()),
        best_match: u.signatures.iter().filter_map(|s| db.best_match(&s.vec)).max_by(|a, b| a.2.total_cmp(&b.2)).map(|(_, name, score)| BestMatch { name, score }),
    }).collect();
    v.sort_by(|a, b| b.last_seen.cmp(&a.last_seen));
    v
}

/// An unlabelled voice's audio sample (16 kHz mono WAV bytes; empty if none).
#[tauri::command]
fn inbox_sample(id: u64) -> Vec<u8> {
    std::fs::read(people::sample_path(&people::voices_dir(), id)).unwrap_or_default()
}

/// Edits the people file like `edit_people`, for edits that can fail.
fn edit_people_try(state: &State<AppState>, edit: impl FnOnce(&mut people::PeopleDb) -> anyhow::Result<bool>) -> Result<(), String> {
    let mut db = people::PeopleDb::load();
    if edit(&mut db).map_err(|e| format!("{e:#}"))? {
        db.save().map_err(|e| format!("{e:#}"))?;
        state.engine.lock().unwrap().reload_people();
    }
    Ok(())
}

/// Names an unlabelled voice: its signatures go to the person called `name` (made if new).
#[tauri::command]
fn label_voice(state: State<AppState>, id: u64, name: String, keep_sample: bool) -> Result<(), String> {
    edit_people_try(&state, |db| Ok(db.label_voice_files(&people::voices_dir(), id, &name, keep_sample)?.is_some()))
}

#[tauri::command]
fn merge_voices(state: State<AppState>, keep: u64, other: u64) -> Result<(), String> {
    edit_people_try(&state, |db| db.merge_voices_files(&people::voices_dir(), keep, other))
}

#[tauri::command]
fn discard_voice(state: State<AppState>, id: u64) -> Result<(), String> {
    edit_people(&state, |db| db.discard_voice_files(&people::voices_dir(), id))
}

/// Voice vector of `pcm` (16 kHz mono) with the speaker encoder, loaded for the occasion (~1 s), and
/// the encoder's file name.
fn embed_pcm(pcm: &[f32]) -> anyhow::Result<(Vec<f32>, &'static str)> {
    let spk = engines::get_any("speakers")?;
    let Some(model) = engines::installed(spk) else { anyhow::bail!("speaker model not downloaded") };
    anyhow::ensure!(pcm.len() >= 2 * asr::SAMPLE_RATE, "need at least 2 seconds of speech");
    Ok((parakeet::Encoder::load(&model)?.embed(pcm)?, spk.file))
}

/// Saves `pcm` as a signature of the person called `name`, then tells the speaker thread.
fn add_signature(state: &State<AppState>, name: &str, pcm: &[f32]) -> Result<(), String> {
    let name = name.trim();
    if name.is_empty() {
        return Err("Enter a name first.".into());
    }
    let (vec, file) = embed_pcm(pcm).map_err(|e| format!("{e:#}"))?;
    let mut db = people::PeopleDb::load_for(file);
    db.assign(name, vec, &people::now_secs(), pcm.len() as f32 / asr::SAMPLE_RATE as f32);
    db.save().map_err(|e| format!("{e:#}"))?;
    state.engine.lock().unwrap().reload_people();
    Ok(())
}

/// Records the default microphone for `seconds` (its own stream; the caption engine may hold the mic
/// too) and saves the voice under `name`. Emits "record-level" {level 0..1, left seconds} while recording.
#[tauri::command(async)]
fn record_signature(app: AppHandle, state: State<AppState>, name: String, seconds: u32) -> Result<(), String> {
    let seconds = seconds.clamp(2, 60);
    let buf = Arc::new(Mutex::new(Vec::<f32>::new()));
    let sink = buf.clone();
    let (_capture, rate) = audio::start("mic", move |c| sink.lock().unwrap().extend(c)).map_err(|e| format!("{e:#}"))?;
    let start = std::time::Instant::now();
    let mut seen = 0;
    while start.elapsed().as_secs_f32() < seconds as f32 {
        std::thread::sleep(Duration::from_millis(80));
        let b = buf.lock().unwrap();
        let recent = &b[seen.max(b.len().saturating_sub(rate / 10))..];
        let rms = (recent.iter().map(|x| x * x).sum::<f32>() / recent.len().max(1) as f32).sqrt();
        seen = b.len();
        let _ = app.emit("record-level", serde_json::json!({ "level": (rms * 8.0).min(1.0), "left": seconds as f32 - start.elapsed().as_secs_f32() }));
    }
    drop(_capture);
    let raw = std::mem::take(&mut *buf.lock().unwrap());
    let pcm = dsp::Dsp::new(rate, false).process(raw);
    add_signature(&state, &name, &pcm)
}

/// Saves the voice in a WAV file (any rate or channel count; the first minute is used) under `name`.
#[tauri::command(async)]
fn import_signature(state: State<AppState>, name: String, bytes: Vec<u8>) -> Result<(), String> {
    let mut pcm = dsp::decode_wav_16k(&bytes).map_err(|e| format!("Not a WAV file I can read: {e:#}"))?;
    pcm.truncate(60 * asr::SAMPLE_RATE);
    add_signature(&state, &name, &pcm)
}

#[tauri::command]
fn quit_app(app: AppHandle) {
    app.exit(0);
}

#[tauri::command]
fn default_settings() -> Settings {
    Settings::default()
}

fn show_settings(app: &AppHandle) {
    if let Some(w) = app.get_webview_window("settings") {
        let _ = w.show();
        let _ = w.unminimize();
        let _ = w.set_focus();
    }
}

/// The CLI edits the settings file; poll its mtime (1 stat per second) and apply changes live.
fn watch_settings(app: AppHandle) {
    std::thread::spawn(move || {
        let mtime = || std::fs::metadata(settings::path()).and_then(|m| m.modified()).ok();
        let mut seen: Option<SystemTime> = mtime();
        loop {
            std::thread::sleep(Duration::from_secs(1));
            let now = mtime();
            if now != seen {
                seen = now;
                let new = settings::load();
                if *app.state::<AppState>().settings.lock().unwrap() != new {
                    apply(&app, new);
                }
            }
        }
    });
}

fn shutdown_engine(app: &AppHandle) {
    if let Some(state) = app.try_state::<AppState>() {
        state.engine.lock().unwrap().shutdown();
    }
}

pub fn run() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_single_instance::init(|app, _, _| show_settings(app)))
        .setup(|app| {
            // Captions are live work: stop macOS (App Nap) from slowing the app while another app
            // is in front. Held for the app's lifetime; the Mac may still sleep when idle.
            #[cfg(target_os = "macos")]
            {
                use objc2_foundation::{NSActivityOptions, NSProcessInfo, NSString};
                let options = NSActivityOptions::UserInitiatedAllowingIdleSystemSleep | NSActivityOptions::LatencyCritical;
                let token = NSProcessInfo::processInfo().beginActivityWithOptions_reason(options, &NSString::from_str("Live captions"));
                std::mem::forget(token);
            }
            let settings = settings::load();
            let click_through = settings.click_through;

            let status = Arc::new(Mutex::new(String::new()));
            let (handle, last) = (app.handle().clone(), status.clone());
            let engine = engine::spawn(move |ev| {
                let _ = match ev {
                    Event::Caption(c) => handle.emit("caption", c),
                    Event::Status(s) => {
                        *last.lock().unwrap() = s.clone();
                        handle.emit("status", s)
                    }
                };
            });
            let _ = engine.send(settings.audio.clone());

            let overlay_item = CheckMenuItem::with_id(app, "overlay", "Show captions", true, settings.overlay, None::<&str>)?;
            let menu = Menu::with_items(app, &[
                &overlay_item,
                &MenuItem::with_id(app, "settings", "Settings…", true, None::<&str>)?,
                &PredefinedMenuItem::separator(app)?,
                &MenuItem::with_id(app, "quit", "Quit RT Captions", true, None::<&str>)?,
            ])?;
            app.manage(AppState { settings: Mutex::new(settings), engine: Mutex::new(engine), status, overlay_item });

            TrayIconBuilder::with_id("tray")
                .icon(app.default_window_icon().cloned().expect("icon"))
                .tooltip("RT Captions")
                .menu(&menu)
                .show_menu_on_left_click(false)
                .on_menu_event(|app, e| match e.id().as_ref() {
                    "overlay" => {
                        let mut s = app.state::<AppState>().settings.lock().unwrap().clone();
                        s.overlay = !s.overlay;
                        let _ = settings::save(&s);
                        apply(app, s);
                    }
                    "settings" => show_settings(app),
                    "quit" => app.exit(0),
                    _ => {}
                })
                .on_tray_icon_event(|tray, e| {
                    if let TrayIconEvent::Click { button: MouseButton::Left, button_state: MouseButtonState::Up, .. } = e {
                        show_settings(tray.app_handle());
                    }
                })
                .build(app)?;

            // Windows are created here, after state is managed, so their first invoke can't race it.
            for cfg in app.config().app.windows.clone() {
                let w = tauri::WebviewWindowBuilder::from_config(app.handle(), &cfg)?.build()?;
                if cfg.label == "overlay" {
                    w.set_ignore_cursor_events(click_through)?;
                    if let Some(m) = w.current_monitor()? {
                        let (ms, ws) = (m.size(), w.outer_size()?);
                        let x = m.position().x + (ms.width as i32 - ws.width as i32) / 2;
                        let y = m.position().y + ms.height as i32 - ws.height as i32 - (ms.height as i32 / 10);
                        w.set_position(tauri::PhysicalPosition::new(x, y))?;
                    }
                }
            }
            watch_settings(app.handle().clone());
            // First launch: fetch the selected engine's model so captions can start.
            let engine = app.state::<AppState>().settings.lock().unwrap().audio.engine.clone();
            if engines::get(&engine).map_or(false, |e| engines::installed(e).is_none()) {
                let _ = download_engine(app.handle().clone(), engine);
            }
            Ok(())
        })
        .on_window_event(|window, event| {
            // Close means quit, after the window confirms it; "Minimize to tray" is its own button.
            if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                if window.label() == "settings" {
                    api.prevent_close();
                    let _ = window.emit_to("settings", "confirm-quit", ());
                }
            }
        })
        .invoke_handler(tauri::generate_handler![get_settings, get_status, save_settings, default_settings, quit_app, list_engines, download_engine, rename_speaker, assign_voice, list_voices, voice_sample, list_inbox, inbox_sample, label_voice, merge_voices, discard_voice, record_signature, import_signature, list_people, delete_person, delete_signature, rename_person, debug_enabled, debug_ack])
        .build(tauri::generate_context!())
        .expect("error while running RT Captions");
    app.run(|app, event| {
        if matches!(event, tauri::RunEvent::ExitRequested { .. } | tauri::RunEvent::Exit) {
            shutdown_engine(app);
        }
    });
}
