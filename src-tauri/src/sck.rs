//! macOS system audio via ScreenCaptureKit (objc2 bindings; no Swift build needed).

use super::{mix_mono, Capture};
use anyhow::{anyhow, Context, Result};
use block2::RcBlock;
use dispatch2::{DispatchQueue, DispatchQueueAttr};
use objc2::rc::{Allocated, Retained};
use objc2::runtime::ProtocolObject;
use objc2::{define_class, msg_send, AnyThread, DefinedClass};
use objc2_core_audio_types::AudioBufferList;
use objc2_core_foundation::CFRetained;
use objc2_core_media::{CMBlockBuffer, CMSampleBuffer, CMTime, CMTimeFlags};
use objc2_foundation::{NSArray, NSError, NSObject, NSObjectProtocol};
use objc2_screen_capture_kit::*;
use std::ptr::{null_mut, NonNull};
use std::sync::mpsc::channel;
use std::sync::Mutex;
use std::time::Duration;

const HINT: &str = "allow RT Captions (or your terminal, for the CLI) in System Settings > Privacy & Security > Screen & System Audio Recording, then restart it";
const WAIT: Duration = Duration::from_secs(5);

type Sink = Mutex<Box<dyn FnMut(Vec<f32>) + Send>>;

/// The running stream; dropping it stops capture.
pub struct SystemStream {
    stream: Retained<SCStream>,
    _out: Retained<AudioOut>,
}

impl Drop for SystemStream {
    fn drop(&mut self) {
        unsafe { self.stream.stopCaptureWithCompletionHandler(None) };
    }
}

define_class!(
    #[unsafe(super(NSObject))]
    #[name = "RTCaptionsAudioOut"]
    #[ivars = Sink]
    struct AudioOut;

    unsafe impl NSObjectProtocol for AudioOut {}

    unsafe impl SCStreamOutput for AudioOut {
        #[unsafe(method(stream:didOutputSampleBuffer:ofType:))]
        fn did_output(&self, _stream: &SCStream, buf: &CMSampleBuffer, ty: SCStreamOutputType) {
            if ty != SCStreamOutputType::Audio {
                return;
            }
            let mono = mix_mono(&read_channels(buf));
            if !mono.is_empty() {
                if let Ok(mut sink) = self.ivars().lock() {
                    (*sink)(mono);
                }
            }
        }
    }
);

/// Each AudioBuffer of the sample buffer as f32 samples (one per channel when non-interleaved).
fn read_channels(buf: &CMSampleBuffer) -> Vec<Vec<f32>> {
    unsafe {
        let mut need = 0usize;
        if buf.audio_buffer_list_with_retained_block_buffer(&mut need, null_mut(), 0, None, None, 0, null_mut()) != 0 || need == 0 {
            return Vec::new();
        }
        let mut mem = vec![0u64; need.div_ceil(8)]; // 8-byte aligned storage for the list
        let list = mem.as_mut_ptr() as *mut AudioBufferList;
        let mut block: *mut CMBlockBuffer = null_mut();
        if buf.audio_buffer_list_with_retained_block_buffer(null_mut(), list, need, None, None, 0, &mut block) != 0 {
            return Vec::new();
        }
        let n = (*list).mNumberBuffers as usize;
        let bufs = std::slice::from_raw_parts((*list).mBuffers.as_ptr(), n);
        let chans = bufs
            .iter()
            .filter(|b| !b.mData.is_null())
            .map(|b| std::slice::from_raw_parts(b.mData as *const u8, b.mDataByteSize as usize).chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
            .collect();
        // The samples live in the retained block buffer; release it only after copying.
        if let Some(b) = NonNull::new(block) {
            drop(CFRetained::from_raw(b));
        }
        chans
    }
}

struct Handoff<T>(T);
// ponytail: handed from the completion queue to the waiting thread once, never used concurrently.
unsafe impl<T> Send for Handoff<T> {}

fn err_text(e: *mut NSError) -> Option<String> {
    unsafe { e.as_ref().map(|e| e.localizedDescription().to_string()) }
}

pub fn start(sink: impl FnMut(Vec<f32>) + Send + 'static) -> Result<(Capture, usize)> {
    let (tx, rx) = channel();
    let done = RcBlock::new(move |c: *mut SCShareableContent, e: *mut NSError| {
        let display = unsafe { c.as_ref().and_then(|c| c.displays().firstObject()) };
        let _ = tx.send(Handoff(display.ok_or_else(|| err_text(e))));
    });
    unsafe { SCShareableContent::getShareableContentWithCompletionHandler(&done) };
    let display = match rx.recv_timeout(WAIT).map(|h| h.0) {
        Ok(Ok(d)) => d,
        Ok(Err(e)) => return Err(anyhow!("{HINT}: {}", e.unwrap_or_else(|| "no display found".into()))),
        Err(_) => return Err(anyhow!("{HINT}: timed out asking for screen content")),
    };
    unsafe {
        let filter = SCContentFilter::initWithDisplay_excludingWindows(SCContentFilter::alloc(), &display, &NSArray::new());
        // ponytail: SCK needs a display stream even for audio only; keep it a 2x2 frame at 1 fps.
        let cfg = SCStreamConfiguration::new();
        cfg.setWidth(2);
        cfg.setHeight(2);
        cfg.setMinimumFrameInterval(CMTime { value: 1, timescale: 1, flags: CMTimeFlags::Valid, epoch: 0 });
        cfg.setCapturesAudio(true);
        cfg.setSampleRate(48000);
        cfg.setChannelCount(1);
        cfg.setExcludesCurrentProcessAudio(true);
        let stream = SCStream::initWithFilter_configuration_delegate(SCStream::alloc(), &filter, &cfg, None);
        let out: Allocated<AudioOut> = AudioOut::alloc();
        let out: Retained<AudioOut> = msg_send![super(out.set_ivars(Mutex::new(Box::new(sink) as Box<dyn FnMut(Vec<f32>) + Send>))), init];
        let queue = DispatchQueue::new("rt-captions.system-audio", DispatchQueueAttr::SERIAL);
        stream.addStreamOutput_type_sampleHandlerQueue_error(ProtocolObject::from_ref(&*out), SCStreamOutputType::Audio, Some(&queue)).map_err(|e| anyhow!("{e}")).context("add audio output")?;
        let (tx, rx) = channel();
        let started = RcBlock::new(move |e: *mut NSError| {
            let _ = tx.send(err_text(e));
        });
        stream.startCaptureWithCompletionHandler(Some(&started));
        match rx.recv_timeout(WAIT) {
            Ok(None) => {}
            Ok(Some(e)) => return Err(anyhow!("{HINT}: {e}")),
            Err(_) => return Err(anyhow!("{HINT}: timed out starting capture")),
        }
        Ok((Capture::System(SystemStream { stream, _out: out }), 48000))
    }
}
