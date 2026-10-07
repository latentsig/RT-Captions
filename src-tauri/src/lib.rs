pub mod asr;
pub mod captioner;
pub mod cli;
pub mod dsp;
pub mod engine;
pub mod engines;
pub mod parakeet;
pub mod settings;
pub mod whistle;
mod audio;

use engine::{AudioSettings, Event};
use settings::Settings;
use std::sync::{mpsc::Sender, Arc, Mutex};
use std::time::{Duration, SystemTime};
use tauri::menu::{CheckMenuItem, Menu, MenuItem, PredefinedMenuItem};
use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
use tauri::{AppHandle, Emitter, Manager, State, Wry};

struct AppState {
    settings: Mutex<Settings>,
    engine: Mutex<Sender<AudioSettings>>,
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

#[derive(serde::Serialize)]
struct EngineInfo {
    #[serde(flatten)]
    engine: engines::Engine,
    installed: bool,
}

#[tauri::command]
fn list_engines() -> Vec<EngineInfo> {
    engines::ENGINES.iter().map(|e| EngineInfo { engine: *e, installed: engines::installed(e).is_some() }).collect()
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
    let e = engines::get(&id).map_err(|e| format!("{e:#}"))?;
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

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_single_instance::init(|app, _, _| show_settings(app)))
        .setup(|app| {
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
        .invoke_handler(tauri::generate_handler![get_settings, get_status, save_settings, default_settings, quit_app, list_engines, download_engine])
        .run(tauri::generate_context!())
        .expect("error while running RT Captions");
}
