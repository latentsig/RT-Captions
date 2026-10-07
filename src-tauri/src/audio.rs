//! Audio capture: microphone, or system output via WASAPI loopback (Windows).

use anyhow::{bail, Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample, SizedSample, Stream};

/// Starts capture; hands mono f32 chunks at the returned sample rate to `sink`.
pub fn start(source: &str, sink: impl FnMut(Vec<f32>) + Send + 'static) -> Result<(Stream, usize)> {
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
        // ponytail: macOS system audio needs ScreenCaptureKit. Next phase.
        "system" => bail!("system audio on macOS is not in this build yet"),
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
    Ok((stream, rate))
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

/// Default device names, for `rt-captions doctor`.
pub fn describe() -> Vec<String> {
    let host = cpal::default_host();
    let name = |d: Option<cpal::Device>| d.and_then(|d| d.name().ok()).unwrap_or_else(|| "none".into());
    vec![format!("microphone: {}", name(host.default_input_device())), format!("system output: {}", name(host.default_output_device()))]
}
