//! # ZRTL Audio Plugin
//!
//! Provides audio loading and processing for machine learning workloads.
//! Essential for speech recognition, audio classification, and music ML models.
//!
//! ## Features
//!
//! - Audio file loading (WAV, MP3, FLAC, OGG, etc.)
//! - Resampling to target sample rate
//! - Channel conversion (stereo to mono)
//! - STFT and mel spectrogram computation
//! - Audio preprocessing (normalize, trim silence)
//!
//! ## Typical ML Pipeline
//!
//! ```text
//! let audio = audio_load("speech.wav");
//! let resampled = audio_resample(audio, 16000);
//! let mono = audio_to_mono(resampled);
//! let mel = audio_mel_spectrogram(mono, 80, 400, 160);
//! // Feed mel to Whisper encoder...
//! ```

use std::collections::HashMap;
use std::f32::consts::PI;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use rubato::{FftFixedInOut, Resampler};
use rustfft::{num_complex::Complex, FftPlanner};
use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use zrtl::zrtl_plugin;

// Import SIMD functions for optimized operations
use zrtl_simd_kernels::{vec_max_f32, vec_scale_f32, vec_dot_product_f32};

// ============================================================================
// Audio Handle
// ============================================================================

/// Internal audio storage
struct AudioData {
    /// Sample rate in Hz
    sample_rate: u32,
    /// Number of channels
    channels: u32,
    /// Interleaved sample data (all channels)
    samples: Vec<f32>,
}

impl AudioData {
    fn duration_samples(&self) -> usize {
        if self.channels == 0 {
            return 0;
        }
        self.samples.len() / self.channels as usize
    }

    fn duration_seconds(&self) -> f32 {
        if self.sample_rate == 0 {
            return 0.0;
        }
        self.duration_samples() as f32 / self.sample_rate as f32
    }
}

/// Global audio handle storage
static NEXT_HANDLE: AtomicU64 = AtomicU64::new(1);
static AUDIO_STORE: Mutex<Option<HashMap<u64, AudioData>>> = Mutex::new(None);

fn init_store() {
    let mut store = AUDIO_STORE.lock().unwrap();
    if store.is_none() {
        *store = Some(HashMap::new());
    }
}

fn insert_audio(data: AudioData) -> u64 {
    init_store();
    let handle = NEXT_HANDLE.fetch_add(1, Ordering::Relaxed);
    let mut store = AUDIO_STORE.lock().unwrap();
    if let Some(ref mut map) = *store {
        map.insert(handle, data);
    }
    handle
}

fn get_audio<F, R>(handle: u64, f: F) -> R
where
    F: FnOnce(Option<&AudioData>) -> R,
{
    init_store();
    let store = AUDIO_STORE.lock().unwrap();
    if let Some(ref map) = *store {
        f(map.get(&handle))
    } else {
        f(None)
    }
}

fn take_audio(handle: u64) -> Option<AudioData> {
    init_store();
    let mut store = AUDIO_STORE.lock().unwrap();
    if let Some(ref mut map) = *store {
        map.remove(&handle)
    } else {
        None
    }
}

pub type AudioHandle = u64;
pub const AUDIO_NULL: AudioHandle = 0;

// ============================================================================
// Loading
// ============================================================================

/// Load audio from file path
#[no_mangle]
pub extern "C" fn audio_load(path: *const u8, path_len: usize) -> AudioHandle {
    if path.is_null() || path_len == 0 {
        return AUDIO_NULL;
    }

    let path_str = unsafe {
        match std::str::from_utf8(std::slice::from_raw_parts(path, path_len)) {
            Ok(s) => s,
            Err(_) => return AUDIO_NULL,
        }
    };

    // Open the file
    let file = match std::fs::File::open(path_str) {
        Ok(f) => f,
        Err(_) => return AUDIO_NULL,
    };

    // Create media source stream
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    // Create hint from file extension
    let mut hint = Hint::new();
    if let Some(ext) = std::path::Path::new(path_str).extension() {
        if let Some(ext_str) = ext.to_str() {
            hint.with_extension(ext_str);
        }
    }

    // Probe the format
    let meta_opts = MetadataOptions::default();
    let fmt_opts = FormatOptions::default();

    let probed = match symphonia::default::get_probe().format(&hint, mss, &fmt_opts, &meta_opts) {
        Ok(p) => p,
        Err(_) => return AUDIO_NULL,
    };

    let mut format = probed.format;

    // Find first audio track
    let track = match format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
    {
        Some(t) => t,
        None => return AUDIO_NULL,
    };

    let track_id = track.id;
    let codec_params = track.codec_params.clone();

    // Create decoder
    let dec_opts = DecoderOptions::default();
    let mut decoder = match symphonia::default::get_codecs().make(&codec_params, &dec_opts) {
        Ok(d) => d,
        Err(_) => return AUDIO_NULL,
    };

    // Get sample rate and channels
    let sample_rate = codec_params.sample_rate.unwrap_or(44100);
    let channels = codec_params.channels.map(|c| c.count()).unwrap_or(2) as u32;

    // Decode all packets
    let mut all_samples: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(symphonia::core::errors::Error::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(_) => break,
        };

        if packet.track_id() != track_id {
            continue;
        }

        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(_) => continue,
        };

        // Convert to f32 interleaved
        match decoded {
            AudioBufferRef::F32(buf) => {
                let ch_count = buf.spec().channels.count();
                let frames = buf.frames();
                for frame in 0..frames {
                    for ch in 0..ch_count {
                        all_samples.push(buf.chan(ch)[frame]);
                    }
                }
            }
            AudioBufferRef::S16(buf) => {
                let ch_count = buf.spec().channels.count();
                let frames = buf.frames();
                for frame in 0..frames {
                    for ch in 0..ch_count {
                        all_samples.push(buf.chan(ch)[frame] as f32 / 32768.0);
                    }
                }
            }
            AudioBufferRef::S32(buf) => {
                let ch_count = buf.spec().channels.count();
                let frames = buf.frames();
                for frame in 0..frames {
                    for ch in 0..ch_count {
                        all_samples.push(buf.chan(ch)[frame] as f32 / 2147483648.0);
                    }
                }
            }
            AudioBufferRef::U8(buf) => {
                let ch_count = buf.spec().channels.count();
                let frames = buf.frames();
                for frame in 0..frames {
                    for ch in 0..ch_count {
                        all_samples.push((buf.chan(ch)[frame] as f32 - 128.0) / 128.0);
                    }
                }
            }
            _ => continue,
        }
    }

    let audio = AudioData {
        sample_rate,
        channels,
        samples: all_samples,
    };

    insert_audio(audio)
}

/// Free audio handle
#[no_mangle]
pub extern "C" fn audio_free(handle: AudioHandle) {
    take_audio(handle);
}

// ============================================================================
// Info
// ============================================================================

/// Get sample rate
#[no_mangle]
pub extern "C" fn audio_sample_rate(handle: AudioHandle) -> u32 {
    get_audio(handle, |audio| audio.map(|a| a.sample_rate).unwrap_or(0))
}

/// Get number of channels
#[no_mangle]
pub extern "C" fn audio_channels(handle: AudioHandle) -> u32 {
    get_audio(handle, |audio| audio.map(|a| a.channels).unwrap_or(0))
}

/// Get duration in samples (per channel)
#[no_mangle]
pub extern "C" fn audio_duration_samples(handle: AudioHandle) -> usize {
    get_audio(handle, |audio| audio.map(|a| a.duration_samples()).unwrap_or(0))
}

/// Get duration in seconds
#[no_mangle]
pub extern "C" fn audio_duration_seconds(handle: AudioHandle) -> f32 {
    get_audio(handle, |audio| audio.map(|a| a.duration_seconds()).unwrap_or(0.0))
}

/// Get raw sample data pointer and length
#[no_mangle]
pub extern "C" fn audio_samples(handle: AudioHandle, out_ptr: *mut *const f32, out_len: *mut usize) {
    if out_ptr.is_null() || out_len.is_null() {
        return;
    }

    get_audio(handle, |audio| {
        if let Some(a) = audio {
            unsafe {
                *out_ptr = a.samples.as_ptr();
                *out_len = a.samples.len();
            }
        } else {
            unsafe {
                *out_ptr = std::ptr::null();
                *out_len = 0;
            }
        }
    });
}

// ============================================================================
// Processing
// ============================================================================

/// Convert to mono by averaging channels
#[no_mangle]
pub extern "C" fn audio_to_mono(handle: AudioHandle) -> AudioHandle {
    let audio = match take_audio(handle) {
        Some(a) => a,
        None => return AUDIO_NULL,
    };

    if audio.channels == 1 {
        // Already mono, just re-insert
        return insert_audio(audio);
    }

    let frames = audio.duration_samples();
    let mut mono_samples = Vec::with_capacity(frames);

    for frame in 0..frames {
        let mut sum = 0.0f32;
        for ch in 0..audio.channels as usize {
            sum += audio.samples[frame * audio.channels as usize + ch];
        }
        mono_samples.push(sum / audio.channels as f32);
    }

    let mono = AudioData {
        sample_rate: audio.sample_rate,
        channels: 1,
        samples: mono_samples,
    };

    insert_audio(mono)
}

/// Resample audio to target sample rate
#[no_mangle]
pub extern "C" fn audio_resample(handle: AudioHandle, target_sr: u32) -> AudioHandle {
    let audio = match take_audio(handle) {
        Some(a) => a,
        None => return AUDIO_NULL,
    };

    if audio.sample_rate == target_sr {
        return insert_audio(audio);
    }

    // Use rubato for high-quality resampling
    let channels = audio.channels as usize;
    let frames = audio.duration_samples();

    // Calculate chunk sizes
    let chunk_size = 1024;

    // Create resampler
    let mut resampler = match FftFixedInOut::<f32>::new(
        audio.sample_rate as usize,
        target_sr as usize,
        chunk_size,
        channels,
    ) {
        Ok(r) => r,
        Err(_) => return AUDIO_NULL,
    };

    // Deinterleave input
    let mut channel_buffers: Vec<Vec<f32>> = vec![Vec::with_capacity(frames); channels];
    for frame in 0..frames {
        for ch in 0..channels {
            channel_buffers[ch].push(audio.samples[frame * channels + ch]);
        }
    }

    // Process in chunks
    let input_frames_per_chunk = resampler.input_frames_next();
    let output_frames_per_chunk = resampler.output_frames_next();

    let mut output_buffers: Vec<Vec<f32>> = vec![Vec::new(); channels];
    let mut pos = 0;

    while pos < frames {
        let chunk_frames = (frames - pos).min(input_frames_per_chunk);

        // Prepare input chunk
        let input_chunk: Vec<Vec<f32>> = channel_buffers
            .iter()
            .map(|ch| {
                let mut chunk = vec![0.0f32; input_frames_per_chunk];
                let copy_len = chunk_frames.min(ch.len() - pos);
                chunk[..copy_len].copy_from_slice(&ch[pos..pos + copy_len]);
                chunk
            })
            .collect();

        // Resample
        match resampler.process(&input_chunk, None) {
            Ok(output) => {
                for (ch, out_ch) in output.iter().enumerate() {
                    output_buffers[ch].extend_from_slice(out_ch);
                }
            }
            Err(_) => break,
        }

        pos += chunk_frames;
    }

    // Interleave output
    let out_frames = output_buffers.first().map(|b| b.len()).unwrap_or(0);
    let mut interleaved = Vec::with_capacity(out_frames * channels);

    for frame in 0..out_frames {
        for ch in 0..channels {
            interleaved.push(output_buffers[ch][frame]);
        }
    }

    let resampled = AudioData {
        sample_rate: target_sr,
        channels: audio.channels,
        samples: interleaved,
    };

    insert_audio(resampled)
}

/// Normalize audio to [-1, 1] range
#[no_mangle]
pub extern "C" fn audio_normalize(handle: AudioHandle) -> AudioHandle {
    let mut audio = match take_audio(handle) {
        Some(a) => a,
        None => return AUDIO_NULL,
    };

    if audio.samples.is_empty() {
        return insert_audio(audio);
    }

    // Use SIMD to find max absolute value
    // First compute max of absolute values using SIMD
    // We need to find max(|samples|), so we use vec_max_f32 on abs values
    let abs_samples: Vec<f32> = audio.samples.iter().map(|s| s.abs()).collect();
    let max_abs = vec_max_f32(abs_samples.as_ptr(), abs_samples.len() as u64);

    if max_abs > 1e-10 {
        // Use SIMD scale operation: scale by 1/max_abs
        vec_scale_f32(audio.samples.as_mut_ptr(), 1.0 / max_abs, audio.samples.len() as u64);
    }

    insert_audio(audio)
}

/// Trim silence from beginning and end
#[no_mangle]
pub extern "C" fn audio_trim_silence(handle: AudioHandle, threshold: f32) -> AudioHandle {
    let audio = match take_audio(handle) {
        Some(a) => a,
        None => return AUDIO_NULL,
    };

    let frames = audio.duration_samples();
    let channels = audio.channels as usize;

    // Find first non-silent frame
    let mut start_frame = 0;
    for frame in 0..frames {
        let mut max_in_frame = 0.0f32;
        for ch in 0..channels {
            max_in_frame = max_in_frame.max(audio.samples[frame * channels + ch].abs());
        }
        if max_in_frame > threshold {
            start_frame = frame;
            break;
        }
    }

    // Find last non-silent frame
    let mut end_frame = frames;
    for frame in (0..frames).rev() {
        let mut max_in_frame = 0.0f32;
        for ch in 0..channels {
            max_in_frame = max_in_frame.max(audio.samples[frame * channels + ch].abs());
        }
        if max_in_frame > threshold {
            end_frame = frame + 1;
            break;
        }
    }

    if start_frame >= end_frame {
        // All silence
        return insert_audio(AudioData {
            sample_rate: audio.sample_rate,
            channels: audio.channels,
            samples: Vec::new(),
        });
    }

    let new_samples = audio.samples[start_frame * channels..end_frame * channels].to_vec();

    insert_audio(AudioData {
        sample_rate: audio.sample_rate,
        channels: audio.channels,
        samples: new_samples,
    })
}

// ============================================================================
// Feature Extraction
// ============================================================================

/// Compute STFT magnitude spectrogram
/// Returns handle to tensor-like data with shape [n_frames, n_fft/2+1]
#[no_mangle]
pub extern "C" fn audio_stft(
    handle: AudioHandle,
    n_fft: usize,
    hop_length: usize,
    out_data: *mut *mut f32,
    out_frames: *mut usize,
    out_bins: *mut usize,
) -> bool {
    if out_data.is_null() || out_frames.is_null() || out_bins.is_null() {
        return false;
    }

    get_audio(handle, |audio| {
        let audio = match audio {
            Some(a) => a,
            None => return false,
        };

        if audio.channels != 1 {
            // Require mono audio
            return false;
        }

        let samples = &audio.samples;
        let n_samples = samples.len();

        if n_samples < n_fft {
            return false;
        }

        // Calculate number of frames
        let n_frames = (n_samples - n_fft) / hop_length + 1;
        let n_bins = n_fft / 2 + 1;

        // Create FFT planner
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(n_fft);

        // Create Hann window
        let window: Vec<f32> = (0..n_fft)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / n_fft as f32).cos()))
            .collect();

        // Allocate output
        let mut output = vec![0.0f32; n_frames * n_bins];
        let mut buffer: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); n_fft];

        for frame in 0..n_frames {
            let start = frame * hop_length;

            // Apply window and copy to buffer
            for i in 0..n_fft {
                buffer[i] = Complex::new(samples[start + i] * window[i], 0.0);
            }

            // Compute FFT
            fft.process(&mut buffer);

            // Store magnitude (first half + DC)
            for bin in 0..n_bins {
                output[frame * n_bins + bin] = buffer[bin].norm();
            }
        }

        // Allocate and copy output
        let data_ptr = Box::into_raw(output.into_boxed_slice()) as *mut f32;

        unsafe {
            *out_data = data_ptr;
            *out_frames = n_frames;
            *out_bins = n_bins;
        }

        true
    })
}

/// Free STFT data allocated by audio_stft
#[no_mangle]
pub extern "C" fn audio_stft_free(data: *mut f32, len: usize) {
    if !data.is_null() && len > 0 {
        unsafe {
            let _ = Box::from_raw(std::slice::from_raw_parts_mut(data, len));
        }
    }
}

/// Compute mel spectrogram
#[no_mangle]
pub extern "C" fn audio_mel_spectrogram(
    handle: AudioHandle,
    n_mels: usize,
    n_fft: usize,
    hop_length: usize,
    out_data: *mut *mut f32,
    out_frames: *mut usize,
    out_mels: *mut usize,
) -> bool {
    if out_data.is_null() || out_frames.is_null() || out_mels.is_null() {
        return false;
    }

    // First compute STFT
    let mut stft_data: *mut f32 = std::ptr::null_mut();
    let mut n_frames: usize = 0;
    let mut n_bins: usize = 0;

    if !audio_stft(handle, n_fft, hop_length, &mut stft_data, &mut n_frames, &mut n_bins) {
        return false;
    }

    if stft_data.is_null() || n_frames == 0 || n_bins == 0 {
        return false;
    }

    // Get sample rate
    let sample_rate = audio_sample_rate(handle);
    if sample_rate == 0 {
        audio_stft_free(stft_data, n_frames * n_bins);
        return false;
    }

    // Create mel filterbank
    let mel_filters = create_mel_filterbank(n_mels, n_bins, sample_rate as f32);

    // Apply mel filterbank to each frame using SIMD dot product
    let mut mel_output = vec![0.0f32; n_frames * n_mels];

    unsafe {
        for frame in 0..n_frames {
            let frame_ptr = stft_data.add(frame * n_bins);
            let mel_start = frame * n_mels;

            for mel in 0..n_mels {
                // Use SIMD dot product for mel filterbank application
                let filter_ptr = mel_filters.as_ptr().add(mel * n_bins);
                let sum = vec_dot_product_f32(frame_ptr, filter_ptr, n_bins as u64);
                // Convert to log scale (with floor to avoid log(0))
                mel_output[mel_start + mel] = (sum.max(1e-10)).ln();
            }
        }
    }

    // Free STFT data
    audio_stft_free(stft_data, n_frames * n_bins);

    // Allocate and return output
    let data_ptr = Box::into_raw(mel_output.into_boxed_slice()) as *mut f32;

    unsafe {
        *out_data = data_ptr;
        *out_frames = n_frames;
        *out_mels = n_mels;
    }

    true
}

/// Create mel filterbank
fn create_mel_filterbank(n_mels: usize, n_bins: usize, sample_rate: f32) -> Vec<f32> {
    let f_max = sample_rate / 2.0;
    let f_min = 0.0;

    // Convert Hz to mel
    fn hz_to_mel(hz: f32) -> f32 {
        2595.0 * (1.0 + hz / 700.0).log10()
    }

    // Convert mel to Hz
    fn mel_to_hz(mel: f32) -> f32 {
        700.0 * (10.0f32.powf(mel / 2595.0) - 1.0)
    }

    let mel_min = hz_to_mel(f_min);
    let mel_max = hz_to_mel(f_max);

    // Mel points
    let mel_points: Vec<f32> = (0..=n_mels + 1)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
        .collect();

    // Convert back to Hz and then to bin indices
    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();
    let bin_points: Vec<usize> = hz_points
        .iter()
        .map(|&hz| ((hz / f_max) * (n_bins - 1) as f32).round() as usize)
        .collect();

    // Create filterbank
    let mut filters = vec![0.0f32; n_mels * n_bins];

    for mel in 0..n_mels {
        let left = bin_points[mel];
        let center = bin_points[mel + 1];
        let right = bin_points[mel + 2];

        // Rising edge
        if center > left {
            for bin in left..center {
                filters[mel * n_bins + bin] = (bin - left) as f32 / (center - left) as f32;
            }
        }

        // Falling edge
        if right > center {
            for bin in center..right {
                filters[mel * n_bins + bin] = (right - bin) as f32 / (right - center) as f32;
            }
        }
    }

    filters
}

// ============================================================================
// Plugin Registration
// ============================================================================

zrtl_plugin! {
    name: "audio",
    symbols: [
        // Loading
        ("$Audio$load", audio_load),
        ("$Audio$free", audio_free),

        // Info
        ("$Audio$sample_rate", audio_sample_rate),
        ("$Audio$channels", audio_channels),
        ("$Audio$duration_samples", audio_duration_samples),
        ("$Audio$duration_seconds", audio_duration_seconds),
        ("$Audio$samples", audio_samples),

        // Processing
        ("$Audio$to_mono", audio_to_mono),
        ("$Audio$resample", audio_resample),
        ("$Audio$normalize", audio_normalize),
        ("$Audio$trim_silence", audio_trim_silence),

        // Feature extraction
        ("$Audio$stft", audio_stft),
        ("$Audio$stft_free", audio_stft_free),
        ("$Audio$mel_spectrogram", audio_mel_spectrogram),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mel_filterbank() {
        let filters = create_mel_filterbank(40, 257, 16000.0);
        assert_eq!(filters.len(), 40 * 257);

        // Just verify the filterbank is created with correct size
        // and that values are in valid range [0, 1]
        for i in 0..filters.len() {
            assert!(filters[i] >= 0.0 && filters[i] <= 1.0);
        }
    }

    #[test]
    fn test_audio_handle_lifecycle() {
        // Create a simple audio buffer
        let audio = AudioData {
            sample_rate: 16000,
            channels: 1,
            samples: vec![0.0; 16000], // 1 second of silence
        };

        let handle = insert_audio(audio);
        assert_ne!(handle, AUDIO_NULL);

        assert_eq!(audio_sample_rate(handle), 16000);
        assert_eq!(audio_channels(handle), 1);
        assert_eq!(audio_duration_samples(handle), 16000);
        assert!((audio_duration_seconds(handle) - 1.0).abs() < 0.001);

        audio_free(handle);

        // After free, should return 0
        assert_eq!(audio_sample_rate(handle), 0);
    }

    #[test]
    fn test_to_mono() {
        // Stereo audio: left=1.0, right=-1.0
        let stereo = AudioData {
            sample_rate: 16000,
            channels: 2,
            samples: vec![1.0, -1.0, 0.5, -0.5], // 2 frames
        };

        let handle = insert_audio(stereo);
        let mono_handle = audio_to_mono(handle);

        assert_ne!(mono_handle, AUDIO_NULL);
        assert_eq!(audio_channels(mono_handle), 1);
        assert_eq!(audio_duration_samples(mono_handle), 2);

        audio_free(mono_handle);
    }

    #[test]
    fn test_normalize() {
        let audio = AudioData {
            sample_rate: 16000,
            channels: 1,
            samples: vec![0.5, -0.25, 0.0],
        };

        let handle = insert_audio(audio);
        let norm_handle = audio_normalize(handle);

        assert_ne!(norm_handle, AUDIO_NULL);

        // Check max is now 1.0
        let mut ptr: *const f32 = std::ptr::null();
        let mut len: usize = 0;
        audio_samples(norm_handle, &mut ptr as *mut _, &mut len as *mut _);

        assert!(!ptr.is_null());
        assert_eq!(len, 3);

        unsafe {
            let samples = std::slice::from_raw_parts(ptr, len);
            assert!((samples[0] - 1.0).abs() < 0.001);
            assert!((samples[1] - (-0.5)).abs() < 0.001);
        }

        audio_free(norm_handle);
    }
}
