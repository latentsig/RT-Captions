//! Audio capture: microphone, or system output (WASAPI loopback on Windows, ScreenCaptureKit on macOS).

use anyhow::{bail, Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample, SizedSample, Stream};

/// A running capture; dropping it stops it (the fields are only held for that).
#[allow(dead_code)]
pub enum Capture {
    Cpal(Stream),
    #[cfg(target_os = "macos")]
    System(sck::SystemStream),
    File(FilePlayer),
}

/// RT_CAPTIONS_FAKE_MIC=<16 kHz mono wav>: the mic source plays this file in real time (then
/// silence), so the live pipeline can be tested without speakers and a room.
pub struct FilePlayer(std::sync::Arc<std::sync::atomic::AtomicBool>);

impl Drop for FilePlayer {
    fn drop(&mut self) {
        self.0.store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

fn play_file(path: &std::ffi::OsStr, mut sink: impl FnMut(Vec<f32>) + Send + 'static) -> Result<(Capture, usize)> {
    let mut r = hound::WavReader::open(path).with_context(|| format!("opening {}", path.to_string_lossy()))?;
    let spec = r.spec();
    let pcm: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => r.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => r.samples::<i32>().map(|s| s.map(|v| v as f32 / (1i64 << (spec.bits_per_sample - 1)) as f32)).collect::<Result<_, _>>()?,
    };
    let ch = spec.channels as usize;
    let mono: Vec<f32> = pcm.chunks(ch).map(|f| f.iter().sum::<f32>() / ch as f32).collect();
    let rate = spec.sample_rate as usize;
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let flag = stop.clone();
    std::thread::spawn(move || {
        let chunk = rate / 100; // 10 ms, like a sound card
        let start = std::time::Instant::now();
        for i in 0.. {
            if flag.load(std::sync::atomic::Ordering::Relaxed) {
                break;
            }
            let a = i * chunk;
            sink(if a < mono.len() { mono[a..(a + chunk).min(mono.len())].to_vec() } else { vec![0.0; chunk] });
            let due = start + std::time::Duration::from_millis(10 * (i as u64 + 1));
            std::thread::sleep(due.saturating_duration_since(std::time::Instant::now()));
        }
    });
    Ok((Capture::File(FilePlayer(stop)), rate))
}

/// Starts capture; hands mono f32 chunks at the returned sample rate to `sink`.
pub fn start(source: &str, sink: impl FnMut(Vec<f32>) + Send + 'static) -> Result<(Capture, usize)> {
    #[cfg(target_os = "macos")]
    if source == "system" {
        return sck::start(sink);
    }
    if let (Some(path), "mic") = (std::env::var_os("RT_CAPTIONS_FAKE_MIC"), source) {
        return play_file(&path, sink);
    }
    let host = cpal::default_host();
    let (device, config) = match source {
        "mic" => {
            let d = host.default_input_device().context("no microphone found")?;
            let c = d.default_input_config()?;
            (d, c)
        }
        // cpal turns an input stream on an output device into WASAPI loopback.
        "system" if cfg!(windows) => {
            let d = host.default_output_device().context("no output device found")?;
            let c = d.default_output_config()?;
            (d, c)
        }
        "system" => bail!("system audio is not supported on this OS"),
        other => bail!("unknown audio source '{other}'"),
    };
    let rate = config.sample_rate().0 as usize;
    let channels = config.channels() as usize;
    let stream = match config.sample_format() {
        cpal::SampleFormat::F32 => build::<f32>(&device, &config.into(), channels, sink)?,
        cpal::SampleFormat::I16 => build::<i16>(&device, &config.into(), channels, sink)?,
        cpal::SampleFormat::U16 => build::<u16>(&device, &config.into(), channels, sink)?,
        cpal::SampleFormat::I32 => build::<i32>(&device, &config.into(), channels, sink)?,
        f => bail!("unsupported sample format {f:?}"),
    };
    stream.play()?;
    Ok((Capture::Cpal(stream), rate))
}

fn build<T: SizedSample>(device: &cpal::Device, config: &cpal::StreamConfig, channels: usize, mut sink: impl FnMut(Vec<f32>) + Send + 'static) -> Result<Stream>
where
    f32: FromSample<T>,
{
    Ok(device.build_input_stream(
        config,
        move |data: &[T], _| {
            let mono = data.chunks(channels).map(|f| f.iter().map(|s| f32::from_sample(*s)).sum::<f32>() / channels as f32).collect();
            sink(mono);
        },
        |e| eprintln!("audio stream error: {e}"),
        None,
    )?)
}

/// Averages per-channel buffers (non-interleaved) into mono.
#[cfg(any(target_os = "macos", test))]
fn mix_mono(chans: &[Vec<f32>]) -> Vec<f32> {
    match chans {
        [] => Vec::new(),
        [one] => one.clone(),
        _ => {
            let n = chans.iter().map(Vec::len).min().unwrap_or(0);
            (0..n).map(|i| chans.iter().map(|c| c[i]).sum::<f32>() / chans.len() as f32).collect()
        }
    }
}

#[cfg(target_os = "macos")]
#[path = "sck.rs"]
mod sck;

/// Default device names, for `rt-captions doctor`.
pub fn describe() -> Vec<String> {
    let host = cpal::default_host();
    let name = |d: Option<cpal::Device>| d.and_then(|d| d.name().ok()).unwrap_or_else(|| "none".into());
    vec![format!("microphone: {}", name(host.default_input_device())), format!("system output: {}", name(host.default_output_device()))]
}

#[cfg(test)]
mod tests {
    use super::mix_mono;

    #[test]
    fn mixes_channels_to_mono() {
        assert_eq!(mix_mono(&[vec![1.0, 0.0], vec![0.0, 1.0]]), vec![0.5, 0.5]);
        assert_eq!(mix_mono(&[vec![0.25, -0.5]]), vec![0.25, -0.5]);
        assert!(mix_mono(&[]).is_empty());
    }
}
