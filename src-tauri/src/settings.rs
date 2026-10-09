//! One settings file shared by the app, the settings window and the CLI. The running app
//! watches it, so `rt-captions set ...` takes effect live.

use crate::engine::AudioSettings;
use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::PathBuf;

pub const IDENTIFIER: &str = "org.intelehealth.rtcaptions";

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
#[serde(default)]
pub struct Settings {
    pub audio: AudioSettings,
    /// Captions overlay on/off (tray toggle).
    pub overlay: bool,
    /// Clicks pass through the overlay to windows below it.
    pub click_through: bool,
    /// Hide the overlay until words are heard, and again after `style.hide_after_s` of quiet.
    pub auto_hide: bool,
    /// Owned by the overlay UI: font, size, fg, bg, opacity, lines, hide_after_s, mic_color, system_color.
    pub style: Value,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            audio: AudioSettings::default(),
            overlay: true,
            click_through: false,
            auto_hide: false,
            style: serde_json::json!({ "font": "Segoe UI Variable Text, Segoe UI, -apple-system, system-ui, sans-serif",
                                       "size": 30, "fg": "#ffffff", "bg": "#101014", "opacity": 0.72,
                                       "lines": 3, "hide_after_s": 5,
                                       "mic_color": "#fff3b0", "system_color": "#b8f4ff" }),
        }
    }
}

pub fn path() -> PathBuf {
    dirs::config_dir().unwrap_or_default().join(IDENTIFIER).join("settings.json")
}

pub fn load() -> Settings {
    let mut s: Settings = std::fs::read_to_string(path()).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default();
    // Files from older versions lack newer style keys: fill them from the defaults.
    if let (Value::Object(have), Value::Object(defaults)) = (&mut s.style, Settings::default().style) {
        for (k, v) in defaults {
            have.entry(k).or_insert(v);
        }
    }
    s
}

pub fn save(s: &Settings) -> Result<()> {
    let p = path();
    std::fs::create_dir_all(p.parent().unwrap())?;
    // write-then-rename so the watcher never reads a half-written file
    let tmp = p.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_string_pretty(s)?)?;
    std::fs::rename(&tmp, &p).with_context(|| format!("writing {}", p.display()))
}

/// Dotted-path read, e.g. `audio.stream.window_s`.
pub fn get(s: &Settings, key: &str) -> Result<Value> {
    let mut v = serde_json::to_value(s)?;
    for k in key.split('.').filter(|k| !k.is_empty()) {
        v = v.get(k).cloned().ok_or_else(|| anyhow!("unknown setting '{key}'"))?;
    }
    Ok(v)
}

/// Dotted-path write. `raw` is parsed as JSON, falling back to a plain string.
/// The result must still be a valid Settings with the same value type, or nothing changes.
pub fn set(s: &Settings, key: &str, raw: &str) -> Result<Settings> {
    let mut root = serde_json::to_value(s)?;
    let new: Value = serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.into()));
    let mut slot = &mut root;
    for k in key.split('.') {
        slot = slot.get_mut(k).ok_or_else(|| anyhow!("unknown setting '{key}'"))?;
    }
    let same_type = std::mem::discriminant(slot) == std::mem::discriminant(&new)
        || (slot.is_number() && new.is_number());
    if !same_type {
        bail!("'{key}' expects {}, got {new}", type_name(slot));
    }
    *slot = new;
    Ok(serde_json::from_value(root)?)
}

fn type_name(v: &Value) -> &'static str {
    match v {
        Value::Bool(_) => "true/false",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        _ => "an object",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dotted_get_set() {
        let s = Settings::default();
        let s = set(&s, "audio.stream.window_s", "4").unwrap();
        assert_eq!(get(&s, "audio.stream.window_s").unwrap(), serde_json::json!(4.0));
        let s = set(&s, "audio.source", "system").unwrap();
        assert_eq!(s.audio.source, "system");
        assert!(set(&s, "audio.rnnoise", "\"yes\"").is_err());
        assert!(set(&s, "audio.nope", "1").is_err());
    }

    #[test]
    fn defaults_keep_captions_visible_and_feed_all_audio() {
        let s = Settings::default();
        assert!(!s.auto_hide);
        assert!(!s.audio.rnnoise);
        assert!(!s.audio.skip_silence);
    }
}
