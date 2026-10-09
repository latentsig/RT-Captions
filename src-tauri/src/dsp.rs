//! Device audio -> 16 kHz mono, optionally through RNNoise (which works at 48 kHz).

use crate::whistle::SAMPLE_RATE;
use nnnoiseless::DenoiseState;
use rubato::{FftFixedIn, Resampler};

struct Rs {
    r: FftFixedIn<f32>,
    pending: Vec<f32>,
}

impl Rs {
    fn new(from: usize, to: usize) -> Option<Self> {
        (from != to).then(|| Self { r: FftFixedIn::new(from, to, from / 100, 1, 1).expect("resampler"), pending: Vec::new() })
    }

    fn run(rs: &mut Option<Self>, input: Vec<f32>) -> Vec<f32> {
        let Some(s) = rs else { return input };
        s.pending.extend(input);
        let mut out = Vec::new();
        while s.pending.len() >= s.r.input_frames_next() {
            let n = s.r.input_frames_next();
            out.extend(&s.r.process(&[&s.pending[..n]], None).expect("resample")[0]);
            s.pending.drain(..n);
        }
        out
    }
}

/// Any WAV (rate, channels, int or float) -> 16 kHz mono.
pub fn decode_wav_16k(bytes: &[u8]) -> anyhow::Result<Vec<f32>> {
    let mut r = hound::WavReader::new(std::io::Cursor::new(bytes))?;
    let spec = r.spec();
    let pcm: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => r.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => r.samples::<i32>().map(|s| s.map(|v| v as f32 / (1i64 << (spec.bits_per_sample - 1)) as f32)).collect::<Result<_, _>>()?,
    };
    let ch = spec.channels.max(1) as usize;
    let mono: Vec<f32> = pcm.chunks(ch).map(|f| f.iter().sum::<f32>() / ch as f32).collect();
    Ok(Dsp::new(spec.sample_rate as usize, false).process(mono))
}

pub struct Dsp {
    to_mid: Option<Rs>,
    denoise: Option<(Box<DenoiseState<'static>>, Vec<f32>)>,
    to_16k: Option<Rs>,
}

impl Dsp {
    pub fn new(input_rate: usize, rnnoise: bool) -> Self {
        let mid = if rnnoise { 48_000 } else { input_rate };
        Self {
            to_mid: Rs::new(input_rate, mid),
            denoise: rnnoise.then(|| (DenoiseState::new(), Vec::new())),
            to_16k: Rs::new(mid, SAMPLE_RATE),
        }
    }

    pub fn process(&mut self, input: Vec<f32>) -> Vec<f32> {
        let mut x = Rs::run(&mut self.to_mid, input);
        if let Some((state, pending)) = &mut self.denoise {
            pending.extend(x.iter().map(|s| s * 32767.0));
            let frames = pending.len() / DenoiseState::FRAME_SIZE;
            x = vec![0.0; frames * DenoiseState::FRAME_SIZE];
            for (inp, out) in pending.chunks_exact(DenoiseState::FRAME_SIZE).zip(x.chunks_exact_mut(DenoiseState::FRAME_SIZE)) {
                state.process_frame(out, inp);
            }
            pending.drain(..frames * DenoiseState::FRAME_SIZE);
            x.iter_mut().for_each(|s| *s /= 32767.0);
        }
        Rs::run(&mut self.to_16k, x)
    }
}
