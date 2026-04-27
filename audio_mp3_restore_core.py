#!/usr/bin/env python3
"""
audio_super_resolution_core.py — DSP back-end
All audio processing functions.  No GUI, no matplotlib, no tkinter.
Import this module from audio_super_resolution_gui.py or use it headless.
"""

import numpy as np
from scipy import signal
from scipy.signal import butter, sosfiltfilt, sosfilt, resample_poly
import scipy.ndimage as ndimage
import threading
import multiprocessing as mp
import math
import concurrent.futures
from math import gcd


# ============================================================================
# Optional pyfftw acceleration
#   pip install pyfftw
# When installed, all numpy.fft and scipy.fft calls (including the STFT
# functions inside scipy.signal.stft/istft) are transparently replaced with
# FFTW3, which is 2-5x faster for the fixed sizes used here.
# Falls back to standard numpy/scipy FFT if pyfftw is not available.
# ============================================================================

def _apply_pyfftw(num_threads=None):
    """
    Monkey-patch numpy.fft and scipy.fft to use FFTW3 via pyfftw.
    Idempotent and safe to call from worker processes.

    num_threads:
        None  -> use all logical CPU cores (best for main process)
        1     -> single-threaded FFTW (best inside ProcessPoolExecutor
                 workers where parallelism is already at process level)

    Returns True if pyfftw was successfully activated, False otherwise.
    """
    try:
        import pyfftw
        import pyfftw.interfaces.numpy_fft as _pnf
        import pyfftw.interfaces.scipy_fft as _psf
        import scipy.fft as _sf

        # Cache FFT plans: a plan is computed the first time a specific
        # (size, dtype, direction) combination is seen, then reused for
        # all subsequent calls with the same parameters.
        pyfftw.interfaces.cache.enable()
        pyfftw.interfaces.cache.set_keepalive_time(60)

        if num_threads is None:
            num_threads = max(1, mp.cpu_count())
        pyfftw.config.NUM_THREADS = num_threads

        # --- numpy.fft patch -------------------------------------------
        # Covers the direct np.fft.rfft / irfft calls scattered through
        # spectral_extension, hf_rolloff, natural_hf_rolloff_filter,
        # _process_audio_chunk, etc.
        np.fft.fft    = _pnf.fft
        np.fft.ifft   = _pnf.ifft
        np.fft.rfft   = _pnf.rfft
        np.fft.irfft  = _pnf.irfft
        np.fft.fftn   = _pnf.fftn
        np.fft.ifftn  = _pnf.ifftn

        # --- scipy.fft backend patch ------------------------------------
        # scipy.signal.stft / istft resolve their FFT calls through
        # scipy.fft's backend registry.  set_backend() registers pyfftw
        # as the preferred backend without touching private internals,
        # covering spectral_copy_from_octave_below (nperseg=2048),
        # spectral_smart_eq_and_repair (nperseg=8192), and
        # _suppress_alias_spikes (nperseg=16384).
        try:
            _sf.set_backend(_psf, only=False)
        except Exception:
            # Older scipy without set_backend: fall back to direct patching
            _sf.fft    = _psf.fft
            _sf.ifft   = _psf.ifft
            _sf.rfft   = _psf.rfft
            _sf.irfft  = _psf.irfft
            _sf.fftn   = _psf.fftn
            _sf.ifftn  = _psf.ifftn
            _sf.rfftn  = _psf.rfftn
            _sf.irfftn = _psf.irfftn

        return True
    except ImportError:
        return False


# Activate in the main process immediately at import time.
# num_threads=None means use all available cores.
_PYFFTW_AVAILABLE = _apply_pyfftw(num_threads=None)


# ============================================================================
#upscalingcore
# ============================================================================

def _izotope_resample_mono(mono, sr_from, sr_to, aliasing):
    n_new = int(len(mono) * sr_to / sr_from)
    nyq = sr_from / 2.0
    cut = nyq * 1.27
    order = max(2, int(10.5 / 4))

    if cut < sr_from * 0.48:
        try:
            sos = signal.butter(order, cut, btype='low', fs=sr_from, output='sos')
            filt = signal.sosfilt(sos, mono)
        except Exception:
            filt = mono.copy()
    else:
        filt = mono.copy()

    old = np.arange(len(filt))
    new = np.linspace(0, len(filt) - 1, n_new)
    lin = np.interp(new, old, filt)
    nrst = filt[np.clip(np.round(new).astype(int), 0, len(filt) - 1)]
    return (1 - aliasing) * lin + aliasing * nrst

def izotope_resample(data, sr_from, sr_to, aliasing):
    if data.ndim == 1:
        return _izotope_resample_mono(data, sr_from, sr_to, aliasing)
    return np.column_stack([
        _izotope_resample_mono(data[:, c], sr_from, sr_to, aliasing)
        for c in range(data.shape[1])
    ])

def spectral_extension_mono(mono, sr, orig_nyq, strength=0.08):
    spec = np.fft.rfft(mono)
    freqs = np.fft.rfftfreq(len(mono), 1.0 / sr)
    ni = np.searchsorted(freqs, orig_nyq)
    for i in range(ni, len(spec)):
        f = freqs[i]
        if f < sr / 2:
            fi = int(i * 0.5)
            if fi < ni:
                spec[i] += spec[fi] * strength / ((f / orig_nyq) ** 2.0)
    return np.fft.irfft(spec, n=len(mono))

def spectral_extension(data, sr, orig_nyq, strength=0.08):
    if data.ndim == 1:
        return spectral_extension_mono(data, sr, orig_nyq, strength)
    return np.column_stack([
        spectral_extension_mono(data[:, c], sr, orig_nyq, strength)
        for c in range(data.shape[1])
    ])

def hf_rolloff_mono(mono, sr, f0, f1, db):
    spec = np.fft.rfft(mono)
    freqs = np.fft.rfftfreq(len(mono), 1.0 / sr)
    mid = (freqs >= f0) & (freqs <= f1)
    high = freqs > f1
    spec[mid] *= 10 ** (db * (freqs[mid] - f0) / (f1 - f0) / 20)
    spec[high] *= 10 ** (db / 20)
    return np.fft.irfft(spec, n=len(mono))

def hf_rolloff(data, sr, f0=24000, f1=96000, db=-18):
    if data.ndim == 1:
        return hf_rolloff_mono(data, sr, f0, f1, db)
    return np.column_stack([
        hf_rolloff_mono(data[:, c], sr, f0, f1, db)
        for c in range(data.shape[1])
    ])

def high_shelf_filter(data, sr, cutoff=20000, gain_db=-6.0, q=0.707):
    import math
    half_gain = gain_db / 2.0
    A = math.pow(10, half_gain / 40.0)
    w0 = 2 * math.pi * cutoff / sr
    alpha = math.sin(w0) / (2 * q)
    b0 = A * ((A + 1) + (A - 1) * math.cos(w0) + 2 * math.sqrt(A) * alpha)
    b1 = -2 * A * ((A - 1) + (A + 1) * math.cos(w0))
    b2 = A * ((A + 1) + (A - 1) * math.cos(w0) - 2 * math.sqrt(A) * alpha)
    a0 = (A + 1) - (A - 1) * math.cos(w0) + 2 * math.sqrt(A) * alpha
    a1 = 2 * ((A - 1) - (A + 1) * math.cos(w0))
    a2 = (A + 1) - (A - 1) * math.cos(w0) - 2 * math.sqrt(A) * alpha
    b = np.array([b0/a0, b1/a0, b2/a0])
    a = np.array([1.0, a1/a0, a2/a0])
    sos = np.concatenate([b, a]).reshape(1, 6)
    if data.ndim == 1:
        return signal.sosfiltfilt(sos, data).astype(np.float32)
    else:
        out = np.zeros_like(data)
        for ch in range(data.shape[1]):
            out[:, ch] = signal.sosfiltfilt(sos, data[:, ch])
        return out.astype(np.float32)


def natural_hf_rolloff_filter(data, sr, f_start=20000, total_db=-18.0,
                               curve_exponent=0.65, transition_hz=1500):
    squeeze = False
    if data.ndim == 1:
        data_2d = data.reshape(-1, 1)
        squeeze = True
    else:
        data_2d = data.copy()

    num_samples, num_channels = data_2d.shape
    f_nyq = sr / 2.0

    f_start       = float(np.clip(f_start, 100.0, f_nyq * 0.95))
    total_db      = float(min(total_db, 0.0))
    curve_exponent = float(max(0.05, curve_exponent))
    transition_hz = float(max(50.0, transition_hz))
    f_trans_start  = max(10.0, f_start - transition_hz)

    output = np.zeros_like(data_2d, dtype=np.float32)

    for ch in range(num_channels):
        mono = data_2d[:, ch].astype(np.float64)
        N    = len(mono)

        spec  = np.fft.rfft(mono)
        freqs = np.fft.rfftfreq(N, 1.0 / sr)

        gain_lin = np.ones(len(freqs), dtype=np.float64)


        trans_mask = (freqs >= f_trans_start) & (freqs < f_start)
        if np.any(trans_mask):
            f_t = freqs[trans_mask]
            mix = 0.5 - 0.5 * np.cos(np.pi * (f_t - f_trans_start) / transition_hz)
            norm_t   = np.clip((f_t - f_start) / max(f_nyq - f_start, 1.0), 0.0, 1.0)
            gain_lin[trans_mask] = 1.0

        rolloff_mask = freqs >= f_start
        if np.any(rolloff_mask):
            f_r    = freqs[rolloff_mask]
            norm_f = np.clip((f_r - f_start) / max(f_nyq - f_start, 1.0), 0.0, 1.0)

            gain_db_r = total_db * (norm_f ** curve_exponent)

            blend_width_norm = min(transition_hz / max(f_nyq - f_start, 1.0), 0.15)
            blend_ramp = np.where(
                norm_f < blend_width_norm,
                0.5 - 0.5 * np.cos(np.pi * norm_f / blend_width_norm),
                1.0
            )
            gain_db_r *= blend_ramp

            gain_lin[rolloff_mask] = np.power(10.0, gain_db_r / 20.0)

        spec_out = spec * gain_lin

        y_out = np.fft.irfft(spec_out, n=N)
        if len(y_out) > num_samples:
            y_out = y_out[:num_samples]
        elif len(y_out) < num_samples:
            y_out = np.pad(y_out, (0, num_samples - len(y_out)))

        output[:, ch] = y_out.astype(np.float32)

    return (output.squeeze() if squeeze else output).astype(np.float32)

def spectral_copy_from_octave_below(data, sr, split_freq=50000):
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    num_samples, num_channels = data.shape
    output = np.zeros_like(data, dtype=np.float32)
    
    nperseg = 2048 
    noverlap = 1536
    
    for ch in range(num_channels):
        f, t, Zxx = signal.stft(data[:, ch], fs=sr, nperseg=nperseg, noverlap=noverlap)
        
        split_bin = np.searchsorted(f, split_freq)

        # Vectorized: compute all source bins at once with integer division,
        # then copy with a single numpy fancy-index assignment (no Python loop).
        dst_bins = np.arange(split_bin, len(f))
        src_bins = dst_bins >> 1          # floor(i / 2) via bit-shift
        Zxx[dst_bins, :] = Zxx[src_bins, :]
            
        _, y_out = signal.istft(Zxx, fs=sr, nperseg=nperseg, noverlap=noverlap)
        
        if len(y_out) > num_samples:
            y_out = y_out[:num_samples]
        elif len(y_out) < num_samples:
            y_out = np.pad(y_out, (0, num_samples - len(y_out)))
            
        output[:, ch] = y_out
        
    return output.squeeze()

# ============================================================================
#normalupsample
# ============================================================================

def high_quality_upsample(data, sr_from, sr_to):
    if sr_from == sr_to:
        return data.astype(np.float32)
    g = gcd(int(sr_from), int(sr_to))
    up   = int(sr_to)   // g
    down = int(sr_from) // g
    kaiser_beta = 14.0 
    if data.ndim == 1:
        return resample_poly(data.astype(np.float64), up, down, window=('kaiser', kaiser_beta)).astype(np.float32)
    channels = []
    for ch in range(data.shape[1]):
        channels.append(resample_poly(data[:, ch].astype(np.float64), up, down, window=('kaiser', kaiser_beta)))
    return np.column_stack(channels).astype(np.float32)

# ============================================================================
#spikessuppressing
# ============================================================================

def _suppress_alias_spikes(data, sr,
                           f_start=18000, f_end=24000,
                           kernel_hz=450,
                           max_cut_db=10.0, max_boost_db=1.5):
    squeeze = False
    if data.ndim == 1:
        data = data.reshape(-1, 1)
        squeeze = True

    num_samples, num_channels = data.shape
    output = np.zeros_like(data, dtype=np.float32)

    nperseg  = 16384
    noverlap = 12288
    bin_res  = sr / nperseg           # Hz per bin

    min_mult = 10 ** (-abs(max_cut_db)   / 20.0)
    max_mult = 10 ** ( abs(max_boost_db) / 20.0)

    kernel_bins = max(3, int(kernel_hz / bin_res))
    if kernel_bins % 2 == 0:
        kernel_bins += 1

    fade_hz   = 500.0
    n_bins_total = nperseg // 2 + 1
    freqs     = np.fft.rfftfreq(nperseg, 1.0 / sr)

    mix_curve = np.zeros(n_bins_total, dtype=np.float64)
    f_in_end  = f_start + fade_hz
    f_out_st  = f_end   - fade_hz

    for i, f in enumerate(freqs):
        if f < f_start or f > f_end:
            mix_curve[i] = 0.0
        elif f <= f_in_end:
            mix_curve[i] = 0.5 - 0.5 * np.cos(np.pi * (f - f_start) / fade_hz)
        elif f >= f_out_st:
            mix_curve[i] = 0.5 - 0.5 * np.cos(np.pi * (f_end - f) / fade_hz)
        else:
            mix_curve[i] = 1.0

    for ch in range(num_channels):
        mono = data[:, ch].astype(np.float64)
        f_ax, t_ax, Zxx = signal.stft(mono, fs=sr,
                                       nperseg=nperseg, noverlap=noverlap)
        mag   = np.abs(Zxx)
        phase = np.exp(1j * np.angle(Zxx))

        avg_mag  = np.mean(mag, axis=1)
        log_mag  = np.log10(avg_mag + 1e-12)

        target_log = ndimage.median_filter(log_mag, size=kernel_bins)
        target_log = ndimage.gaussian_filter(target_log, sigma=kernel_bins / 6.0)

        raw_eq = np.power(10.0, target_log - log_mag)
        eq_lin = np.clip(raw_eq, min_mult, max_mult)

        final_eq = eq_lin * mix_curve + 1.0 * (1.0 - mix_curve)

        mag_out = mag * final_eq[:, np.newaxis]

        Zxx_new = mag_out * phase
        _, y_out = signal.istft(Zxx_new, fs=sr,
                                nperseg=nperseg, noverlap=noverlap)
        if len(y_out) > num_samples:
            y_out = y_out[:num_samples]
        elif len(y_out) < num_samples:
            y_out = np.pad(y_out, (0, num_samples - len(y_out)))

        output[:, ch] = y_out.astype(np.float32)

    return (output.squeeze() if squeeze else output).astype(np.float32)

def upsample_441_to_48k(data, progress_callback=None):
    SR_FROM  = 44100
    SR_TO    = 48000
    ORIG_NYQ = SR_FROM / 2.0

    if progress_callback:
        progress_callback(1, 4, 'High-quality Kaiser resampling 44.1->48kHz ...')
    clean = high_quality_upsample(data, SR_FROM, SR_TO)

    if progress_callback:
        progress_callback(2, 4, 'Aliasing resample: filling 22050~24000Hz ...')

    aliased = izotope_resample(data, SR_FROM, SR_TO, aliasing=0.82)

    aliased = spectral_extension(aliased, SR_TO, ORIG_NYQ, strength=0.05)

    if progress_callback:
        progress_callback(3, 4, 'Smooth splice at 22kHz crossover ...')

    result = spectral_splice_20k(
        clean,
        aliased,
        SR_TO,
        crossover=int(ORIG_NYQ),   # 22050 Hz
        fade_hz=1000
    )

    result = spectral_smart_eq_and_repair(
        result, SR_TO,
        enable_v_repair=True,
        eq_low=20000,
        eq_high=25000,
        max_boost_db=10.0,
        enable_auto_flatten=False
    )

    #
    if progress_callback:
        progress_callback(4, 5, 'Crossover spectral EQ: flatten splice step at 22kHz ...')

    result = spectral_smart_eq_and_repair(
        result, SR_TO,
        enable_v_repair=False,
        enable_auto_flatten=True,
        flat_start_freq=18500,
        flat_width_hz=7000,
        flat_max_boost=10.0,
        flat_max_cut=10.0,
    )

    #
    #
    if progress_callback:
        progress_callback(5, 5, 'Narrow-kernel spike suppression: remove alias harmonics ...')

    result = _suppress_alias_spikes(result, SR_TO)

    return result.astype(np.float32)

def spectral_splice_20k(hq_audio, super_audio, sr, crossover=20000, fade_hz=500):
    min_len = min(len(hq_audio), len(super_audio))
    hq_audio    = hq_audio[:min_len]
    super_audio = super_audio[:min_len]
    squeeze = False
    if hq_audio.ndim == 1:
        hq_audio    = hq_audio.reshape(-1, 1)
        super_audio = super_audio.reshape(-1, 1)
        squeeze = True
    num_channels = hq_audio.shape[1]
    output = np.zeros_like(hq_audio, dtype=np.float32)

    half_sr   = sr / 2.0
    lp_cutoff = np.clip((crossover - fade_hz / 2.0) / half_sr, 0.0005, 0.9995)
    hp_cutoff = np.clip((crossover + fade_hz / 2.0) / half_sr, 0.0005, 0.9995)
    sos_lp = butter(12, lp_cutoff, btype='low',  output='sos')
    sos_hp = butter(12, hp_cutoff, btype='high', output='sos')

    for ch in range(num_channels):
        lp_part = sosfiltfilt(sos_lp, hq_audio[:, ch].astype(np.float64))
        hp_part = sosfiltfilt(sos_hp, super_audio[:, ch].astype(np.float64))
        output[:, ch] = (lp_part + hp_part).astype(np.float32)
    return output.squeeze() if squeeze else output

def compute_band_envelope(audio_mono, sr, f_low, f_high, frame_ms=1.0, smooth_ms=10.0):
    audio_mono = audio_mono.astype(np.float64)
    nyq = sr / 2.0
    f_low_safe  = np.clip(f_low,  1.0, nyq * 0.99)
    f_high_safe = np.clip(f_high, 1.0, nyq * 0.99)
    if f_high_safe <= f_low_safe + 10:
        frame_samples = max(1, int(sr * frame_ms / 1000))
        n_frames = max(1, len(audio_mono) // frame_samples)
        return np.ones(n_frames), frame_samples
    f_low_n  = f_low_safe  / nyq
    f_high_n = f_high_safe / nyq
    try:
        sos_bp = butter(8, [f_low_n, f_high_n], btype='bandpass', output='sos')
        filtered = sosfilt(sos_bp, audio_mono)
    except Exception:
        filtered = audio_mono.copy()
    
    frame_samples = max(1, int(sr * frame_ms / 1000))
    n_frames = max(1, len(filtered) // frame_samples)
    # Vectorized: reshape into (n_frames, frame_samples) and reduce axis=1.
    frames_2d = filtered[:n_frames * frame_samples].reshape(n_frames, frame_samples)
    rms = np.sqrt(np.mean(frames_2d ** 2, axis=1) + 1e-20)
    
    smooth_frames = max(1, int(round(smooth_ms / frame_ms)))
    if smooth_frames > 1:
        kernel = np.ones(smooth_frames) / smooth_frames
        rms = np.convolve(rms, kernel, mode='same')
    return np.maximum(rms, 0.0), frame_samples

def spectral_attenuate_hf(audio, sr, orig_audio, orig_sr,
                           ref_low=19000, ref_high=20000, target_low=20000,
                           frame_ms=1.0, smooth_ms=10.0, gain_floor=0.1):
    if orig_audio.ndim > 1:
        ref_mono = np.mean(orig_audio, axis=1).astype(np.float64)
    else:
        ref_mono = orig_audio.astype(np.float64)

    orig_nyq = orig_sr / 2.0
    actual_ref_high = min(ref_high, orig_nyq * 0.95)
    actual_ref_low  = min(ref_low,  orig_nyq * 0.90)
    if actual_ref_low >= actual_ref_high - 10:
        actual_ref_low  = max(100.0, orig_nyq * 0.80)
        actual_ref_high = orig_nyq * 0.95

    env, _ = compute_band_envelope(ref_mono, orig_sr, actual_ref_low, actual_ref_high, frame_ms, smooth_ms)
    env_99 = np.percentile(env, 99)
    if env_99 > 1e-12:
        env_norm = env / env_99
    else:
        env_norm = np.ones_like(env)
    env_norm = np.clip(env_norm, 0.0, 1.0)

    gain_curve = (gain_floor + (1.0 - gain_floor) * env_norm).astype(np.float32)
    frame_center_ms = np.arange(len(gain_curve), dtype=np.float64) * frame_ms + frame_ms * 0.5
    output_time_ms  = np.arange(len(audio), dtype=np.float64) / (sr / 1000.0)

    gain_samples = np.interp(
        output_time_ms, frame_center_ms, gain_curve,
        left=float(gain_curve[0]), right=float(gain_curve[-1])
    ).astype(np.float32)

    squeeze = False
    if audio.ndim == 1:
        audio = audio.reshape(-1, 1)
        squeeze = True

    num_samples, num_channels = audio.shape
    gain_samples = gain_samples[:num_samples]
    output = np.zeros_like(audio, dtype=np.float32)

    hp_cutoff = np.clip(target_low / (sr / 2.0), 0.0005, 0.9995)
    sos_hp = butter(12, hp_cutoff, btype='high', output='sos')

    for ch in range(num_channels):
        ch_data = audio[:, ch].astype(np.float64)
        hf = sosfiltfilt(sos_hp, ch_data)
        lf = ch_data - hf
        hf_mod = hf * gain_samples.astype(np.float64)
        output[:, ch] = (lf + hf_mod).astype(np.float32)

    return output.squeeze() if squeeze else output


# ============================================================================
#spectralrepair
# ============================================================================

def spectral_smart_eq_and_repair(audio, sr, 
                                 enable_v_repair=True, eq_low=19000, eq_high=25000, max_boost_db=15.0,
                                 enable_auto_flatten=True, flat_start_freq=15000, flat_width_hz=3000,
                                 flat_max_boost=6.0, flat_max_cut=12.0,
                                 notch_center=50000, notch_width=3000):
    squeeze = False
    if audio.ndim == 1:
        audio = audio.reshape(-1, 1)
        squeeze = True

    num_samples, num_channels = audio.shape
    output = np.zeros_like(audio, dtype=np.float32)

    nperseg = 8192
    noverlap = 6144
    bin_res = sr / nperseg

    for ch in range(num_channels):
        f, t, Zxx = signal.stft(audio[:, ch], fs=sr, nperseg=nperseg, noverlap=noverlap)
        mag = np.abs(Zxx)
        phase = np.exp(1j * np.angle(Zxx))

        # ---------------------------------------------------------
        # ---------------------------------------------------------
        if enable_v_repair:
            idx_eq_start = np.where((f >= eq_low - 1000) & (f < eq_low))[0]
            idx_eq_end = np.where((f >= eq_high) & (f < eq_high + 1000))[0]
            idx_eq_target = np.where((f >= eq_low) & (f < eq_high))[0]

            if len(idx_eq_start) > 0 and len(idx_eq_end) > 0 and len(idx_eq_target) > 0:
                mag_start = np.mean(np.log10(mag[idx_eq_start, :] + 1e-10), axis=0)
                mag_end = np.mean(np.log10(mag[idx_eq_end, :] + 1e-10), axis=0)
                
                max_gain_lin = 10 ** (max_boost_db / 20.0)
                gain_matrix = np.ones((len(idx_eq_target), mag.shape[1]))

                for i, bin_idx in enumerate(idx_eq_target):
                    freq = f[bin_idx]
                    alpha = (freq - eq_low) / (eq_high - eq_low)
                    alpha_smooth = 0.5 - 0.5 * np.cos(np.pi * alpha) 
                    ideal_mag_log = mag_start * (1 - alpha_smooth) + mag_end * alpha_smooth
                    ideal_mag = 10 ** ideal_mag_log

                    raw_gain = ideal_mag / (mag[bin_idx, :] + 1e-10)
                    gain_matrix[i, :] = np.clip(raw_gain, 0.1, max_gain_lin)

                gain_matrix = ndimage.gaussian_filter(gain_matrix, sigma=(1.5, 2.0))
                for i, bin_idx in enumerate(idx_eq_target):
                    mag[bin_idx, :] *= gain_matrix[i, :]

            notch_low = notch_center - (notch_width / 2)
            notch_high = notch_center + (notch_width / 2)
            idx_n_start = np.where((f >= notch_low - 1000) & (f < notch_low))[0]
            idx_n_end = np.where((f >= notch_high) & (f < notch_high + 1000))[0]
            idx_n_target = np.where((f >= notch_low) & (f < notch_high))[0]

            if len(idx_n_start) > 0 and len(idx_n_end) > 0 and len(idx_n_target) > 0:
                mag_n_start = np.mean(np.log10(mag[idx_n_start, :] + 1e-10), axis=0)
                mag_n_end = np.mean(np.log10(mag[idx_n_end, :] + 1e-10), axis=0)
                gain_matrix_n = np.ones((len(idx_n_target), mag.shape[1]))
                for i, bin_idx in enumerate(idx_n_target):
                    freq = f[bin_idx]
                    alpha = (freq - notch_low) / (notch_high - notch_low)
                    ideal_mag = 10 ** (mag_n_start * (1 - alpha) + mag_n_end * alpha)
                    raw_gain = ideal_mag / (mag[bin_idx, :] + 1e-10)
                    gain_matrix_n[i, :] = np.maximum(1.0, np.clip(raw_gain, 1.0, 10.0))
                gain_matrix_n = ndimage.gaussian_filter(gain_matrix_n, sigma=(1.0, 1.5))
                for i, bin_idx in enumerate(idx_n_target):
                    mag[bin_idx, :] *= gain_matrix_n[i, :]

        # ---------------------------------------------------------
        # ---------------------------------------------------------
        if enable_auto_flatten:
            avg_mag = np.mean(mag, axis=1)
            log_mag = np.log10(avg_mag + 1e-12)

            smooth_bins = max(5, int(flat_width_hz / bin_res))
            if smooth_bins % 2 == 0: smooth_bins += 1

            target_log_mag = ndimage.median_filter(log_mag, size=smooth_bins)
            target_log_mag = ndimage.gaussian_filter(target_log_mag, sigma=smooth_bins / 4.0)

            raw_eq_curve = 10 ** (target_log_mag - log_mag)

            min_mult = 10 ** (-abs(flat_max_cut) / 20.0)
            max_mult = 10 ** (abs(flat_max_boost) / 20.0)
            eq_curve = np.clip(raw_eq_curve, min_mult, max_mult)

            mix_curve = np.zeros_like(eq_curve)
            start_bin = int(flat_start_freq / bin_res)
            fade_bins = int(1000 / bin_res)

            if start_bin < len(mix_curve):
                mix_curve[start_bin + fade_bins:] = 1.0
                fade_idx = np.arange(fade_bins)
                if start_bin + fade_bins <= len(mix_curve):
                    mix_curve[start_bin : start_bin + fade_bins] = 0.5 - 0.5 * np.cos(np.pi * fade_idx / fade_bins)

            final_eq = eq_curve * mix_curve + 1.0 * (1.0 - mix_curve)
            mag *= final_eq[:, np.newaxis]

        Zxx_new = mag * phase
        _, y_out = signal.istft(Zxx_new, fs=sr, nperseg=nperseg, noverlap=noverlap)
        
        if len(y_out) > num_samples:
            y_out = y_out[:num_samples]
        elif len(y_out) < num_samples:
            y_out = np.pad(y_out, (0, num_samples - len(y_out)))
            
        output[:, ch] = y_out

    return output.squeeze() if squeeze else output


# ============================================================================
#hfspectralsmoothing
# ============================================================================

def super_resolve(data, sr_in, sr_out, ali_start=0.8, ali_end=0.5,
                  rolloff_db=-18, enable_octave_copy=True,
                  split_freq=50000, progress_callback=None):
    cur = data.copy()
    csr = sr_in
    orig_nyq = sr_in / 2.0
    steps = [50000] + list(range(60000, 191000, 10000)) + [192000]
    steps = [s for s in steps if sr_in < s <= sr_out]
    total_steps = len(steps) + 2

    for i, tgt in enumerate(steps, 1):
        a = ali_start - (ali_start - ali_end) * i / len(steps)
        if progress_callback: progress_callback(i, total_steps, f"Resample {csr}→{tgt}Hz")
        cur = izotope_resample(cur, csr, tgt, a)
        if i % 4 == 0: cur = spectral_extension(cur, tgt, orig_nyq)
        csr = tgt

    if progress_callback: progress_callback(len(steps) + 1, total_steps, "Apply HF rolloff")
    cur = hf_rolloff(cur, csr, f0=orig_nyq, f1=sr_out / 2, db=rolloff_db)

    if enable_octave_copy:
        if progress_callback: progress_callback(len(steps) + 2, total_steps, f"Octave copy smoothing >{split_freq / 1000:.0f}kHz")
        cur = spectral_copy_from_octave_below(cur, csr, split_freq=split_freq)
    return cur

def _process_audio_chunk(args):
    # Re-apply pyfftw inside each worker process.
    # On Windows/macOS (spawn start method) workers begin fresh and
    # don't inherit the main-process monkey-patch.  num_threads=1:
    # FFTW uses one thread per worker; the process pool already
    # provides coarse-grained parallelism across chunks.
    _apply_pyfftw(num_threads=1)
    chunk_data, sr, fft_size, hop_size, cutoff_freq, smoothing_mode = args
    freqs = np.fft.rfftfreq(fft_size, 1.0 / sr)
    cutoff_bin = int(np.searchsorted(freqs, cutoff_freq))
    window = np.hanning(fft_size).astype(np.float64)
    w2 = window * window
    num_samples = len(chunk_data)
    chunk_data = np.asarray(chunk_data, dtype=np.float64)
    num_frames = max(0, (num_samples - fft_size) // hop_size + 1)
    if num_frames == 0: return chunk_data.copy()

    starts = np.arange(num_frames, dtype=np.int64) * hop_size
    row_idx = np.clip(starts[:, np.newaxis] + np.arange(fft_size, dtype=np.int64)[np.newaxis, :], 0, num_samples - 1)
    frames = chunk_data[row_idx]                          
    windowed  = frames * window[np.newaxis, :]
    spectra   = np.fft.rfft(windowed, axis=1)             
    magnitudes = np.abs(spectra)                           
    phases     = np.angle(spectra)                         
    n_bins     = magnitudes.shape[1]

    ref_w   = max(3, cutoff_bin // 20)
    ref_s   = max(0, cutoff_bin - ref_w)
    ref_e   = min(n_bins, cutoff_bin + ref_w)
    ref_seg = magnitudes[:, ref_s:ref_e]                  
    ref_level = (np.median(ref_seg, axis=1) + np.mean(ref_seg, axis=1)) * 0.5        

    target_mag = magnitudes.copy()
    if cutoff_bin < n_bins:
        if smoothing_mode == 'perfect_flat':
            target_mag[:, cutoff_bin:] = ref_level[:, np.newaxis]
        elif smoothing_mode == 'gentle_slope':
            safe_ratio = np.maximum(freqs[cutoff_bin:] / max(cutoff_freq, 1.0), 1e-9)
            lin_slope  = 10.0 ** (-3.0 * np.log2(safe_ratio) / 20.0)
            target_mag[:, cutoff_bin:] = ref_level[:, np.newaxis] * lin_slope[np.newaxis, :]

    eps      = 1e-10
    gain_min = 10.0 ** (-24.0 / 20.0)
    gain_max = 10.0 ** (24.0 / 20.0)
    gains = np.clip(target_mag[:, cutoff_bin:] / (magnitudes[:, cutoff_bin:] + eps), gain_min, gain_max)                                                     
    proc_mag = magnitudes.copy()
    proc_mag[:, cutoff_bin:] *= gains

    proc_spectra = proc_mag * np.exp(1j * phases)
    proc_frames  = np.fft.irfft(proc_spectra, n=fft_size, axis=1)  
    proc_frames *= window[np.newaxis, :]

    # Vectorized overlap-add (OLA).
    # hop_size = fft_size // 4  =>  n_overlap = 4 frames touch each sample.
    # Instead of num_frames Python iterations we do n_overlap (=4) numpy
    # row-slice additions entirely in C, regardless of song length.
    #
    # n_segs must satisfy TWO constraints:
    #   1. n_segs >= num_frames + n_overlap - 1  (all frame rows fit)
    #   2. n_segs * hop_size >= num_samples      (ravel covers full output)
    #      e.g. num_samples=8000, hop=128 → need ceil(8000/128)=63 segs,
    #           but num_frames+3=62 — so take the max of both.
    if fft_size % hop_size == 0:
        n_overlap   = fft_size // hop_size
        n_segs_rows = num_frames + n_overlap - 1
        n_segs_len  = -(-num_samples // hop_size)  # ceil division
        n_segs      = max(n_segs_rows, n_segs_len)
        pf_r        = proc_frames.reshape(num_frames, n_overlap, hop_size)
        w2_r        = w2.reshape(n_overlap, hop_size)
        out_segs    = np.zeros((n_segs, hop_size), dtype=np.float64)
        wsum_segs   = np.zeros((n_segs, hop_size), dtype=np.float64)
        for k in range(n_overlap):          # exactly 4 iterations
            out_segs [k : k + num_frames] += pf_r[:, k, :]
            wsum_segs[k : k + num_frames] += w2_r[k]   # broadcast
        output  = out_segs .ravel()[:num_samples].copy()
        win_sum = wsum_segs.ravel()[:num_samples].copy()
    else:
        output  = np.zeros(num_samples, dtype=np.float64)
        win_sum = np.zeros(num_samples, dtype=np.float64)
        for i in range(num_frames):
            s = int(starts[i])
            output[s:s + fft_size]  += proc_frames[i]
            win_sum[s:s + fft_size] += w2

    mask = win_sum > 1e-10
    output[mask] /= win_sum[mask]
    output[~mask] = chunk_data[~mask]
    return output

def smooth_hf_multithread(data, sr, fft_size=512, cutoff_freq=22000,
                          smoothing_mode='perfect_flat', num_workers=None, progress_callback=None):
    if data.ndim == 1: data = data.reshape(-1, 1)
    num_samples, num_channels = data.shape
    output = np.zeros_like(data, dtype=np.float64)
    if num_workers is None: num_workers = max(1, mp.cpu_count() - 1)

    hop_size = fft_size // 4
    target_chunk_samples = max(fft_size * 200, int(sr * 0.5))

    for ch in range(num_channels):
        if progress_callback: progress_callback(ch, num_channels, f"Smoothing ch {ch + 1}/{num_channels}")
        channel_data = data[:, ch].astype(np.float64)
        overlap_size = fft_size * 8
        chunks, chunk_starts, chunk_info = [], [], []
        pos = 0
        while pos < num_samples:
            c_start = max(0, pos - overlap_size)
            c_end   = min(num_samples, pos + target_chunk_samples + overlap_size)
            chunks.append(channel_data[c_start:c_end])
            chunk_starts.append(c_start)
            chunk_info.append((pos - c_start, min(pos + target_chunk_samples, num_samples) - c_start))
            pos += target_chunk_samples
            if c_end >= num_samples: break

        args_list = [(chunk, sr, fft_size, hop_size, cutoff_freq, smoothing_mode) for chunk in chunks]
        if num_workers > 1 and len(chunks) > 1:
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as pool:
                results = list(pool.map(_process_audio_chunk, args_list))
        else:
            results = [_process_audio_chunk(a) for a in args_list]

        fade_length = min(fft_size * 4, overlap_size // 2)
        channel_output = np.zeros(num_samples, dtype=np.float64)

        for idx, (c_start, result, (vs, ve)) in enumerate(zip(chunk_starts, results, chunk_info)):
            actual_s = c_start + vs
            actual_e = c_start + ve
            seg = result[vs:ve]
            if actual_e > num_samples:
                actual_e = num_samples
                seg = seg[:actual_e - actual_s]
            if idx == 0:
                channel_output[actual_s:actual_e] = seg
            else:
                ov_s = actual_s
                ov_e = min(actual_s + fade_length, actual_e)
                if ov_e > ov_s:
                    fl = ov_e - ov_s
                    t  = np.linspace(0.0, np.pi / 2.0, fl)
                    channel_output[ov_s:ov_e] = channel_output[ov_s:ov_e] * np.cos(t) + seg[:fl] * np.sin(t)
                    if ov_e < actual_e: channel_output[ov_e:actual_e] = seg[fl:]
                else:
                    channel_output[actual_s:actual_e] = seg
        output[:, ch] = channel_output
    return output

_CHUNK_OVERLAP_SECONDS = 0.15
def compute_smart_chunks(total_samples, sr_in, max_seconds=5.0):
    total_duration = total_samples / sr_in
    if total_duration <= max_seconds: return [(0, total_samples)]
    n_chunks = math.ceil(total_duration / max_seconds)
    chunk_size = total_samples // n_chunks
    boundaries = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < n_chunks - 1 else total_samples
        boundaries.append((start, end))
    return boundaries

def _process_chunk_task(args):
    (chunk_idx, chunk_data, sr_in, target_sr, ali_start, ali_end, rolloff_db, enable_octave_copy, split_freq,
     fft_size, cutoff_freq, smoothing_mode, _) = args
    try:
        upsampled = super_resolve(chunk_data, sr_in, target_sr, ali_start=ali_start, ali_end=ali_end,
                                  rolloff_db=rolloff_db, enable_octave_copy=enable_octave_copy, split_freq=split_freq)
        smoothed = smooth_hf_multithread(upsampled, target_sr, fft_size=fft_size, cutoff_freq=cutoff_freq,
                                         smoothing_mode=smoothing_mode, num_workers=1)
        return chunk_idx, smoothed, None
    except Exception as e:
        import traceback
        return chunk_idx, None, traceback.format_exc()

def stitch_chunks_seamless(processed_chunks, pre_ovs_out, post_ovs_out):
    n = len(processed_chunks)
    if n == 1:
        d = processed_chunks[0]
        p, q = pre_ovs_out[0], post_ovs_out[0]
        return d[p:(len(d) - q if q > 0 else len(d))]
    ndim = processed_chunks[0].ndim
    junction_cfs = [2 * min(post_ovs_out[i], pre_ovs_out[i + 1]) for i in range(n - 1)]
    cores = []
    for i in range(n):
        d = processed_chunks[i]
        cf_head = junction_cfs[i - 1] if i > 0 else 0
        cf_tail = junction_cfs[i] if i < n - 1 else 0
        end_idx = len(d) - cf_tail if cf_tail > 0 else len(d)
        cores.append(d[cf_head:end_idx])
    total = sum(len(c) for c in cores) + sum(junction_cfs)
    output = np.zeros(total) if ndim == 1 else np.zeros((total, processed_chunks[0].shape[1]))
    pos = 0
    for i in range(n):
        core = cores[i]
        output[pos:pos + len(core)] = core
        pos += len(core)
        if i < n - 1:
            cf = junction_cfs[i]
            d_cur = processed_chunks[i]
            d_nxt = processed_chunks[i + 1]
            tail = d_cur[len(d_cur) - cf:]
            head = d_nxt[:cf]
            t = np.linspace(0.0, np.pi / 2.0, cf)
            fade_out = np.cos(t)[:, np.newaxis] if ndim > 1 else np.cos(t)
            fade_in  = np.sin(t)[:, np.newaxis] if ndim > 1 else np.sin(t)
            output[pos:pos + cf] = tail * fade_out + head * fade_in
            pos += cf
    return output

def process_audio_chunked(data, sr_in, target_sr, ali_start=0.8, ali_end=0.5, rolloff_db=-18,
                           enable_octave_copy=True, split_freq=50000, fft_size=512, cutoff_freq=22000,
                           smoothing_mode='perfect_flat', max_chunk_seconds=5.0, num_chunk_workers=4,
                           num_smooth_workers_per_chunk=1, progress_callback=None):
    total_samples = len(data)
    boundaries = compute_smart_chunks(total_samples, sr_in, max_chunk_seconds)
    n_chunks = len(boundaries)
    ov_in  = int(_CHUNK_OVERLAP_SECONDS * sr_in)
    scale  = target_sr / sr_in
    padded_chunks, pre_ov_in_list, post_ov_in_list = [], [], []

    for i, (start, end) in enumerate(boundaries):
        pre_ov  = 0 if i == 0 else ov_in
        post_ov = 0 if i == n_chunks - 1 else ov_in
        pad_start = max(0, start - pre_ov)
        pad_end   = min(total_samples, end + post_ov)
        padded_chunks.append(data[pad_start:pad_end])
        pre_ov_in_list.append(start - pad_start)
        post_ov_in_list.append(pad_end - end)

    args_list = [(i, padded_chunks[i], sr_in, target_sr, ali_start, ali_end, rolloff_db, enable_octave_copy, split_freq,
                  fft_size, cutoff_freq, smoothing_mode, num_smooth_workers_per_chunk) for i in range(n_chunks)]

    results = {}
    progress_lock = threading.Lock()
    done_count = [0]

    def _on_chunk_done(future_obj):
        chunk_i, result, error = future_obj.result()
        if error: raise RuntimeError(f"Chunk {chunk_i + 1} failed:\n{error}")
        with progress_lock:
            results[chunk_i] = result
            done_count[0] += 1
            if progress_callback: progress_callback(done_count[0], n_chunks, f"Chunks: {done_count[0]}/{n_chunks} done")

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_chunk_workers) as executor:
        futures = {executor.submit(_process_chunk_task, args): i for i, args in enumerate(args_list)}
        for f in concurrent.futures.as_completed(futures): _on_chunk_done(f)

    processed_list = [results[i] for i in range(n_chunks)]
    pre_ovs_out, post_ovs_out = [], []
    for i in range(n_chunks):
        actual_out_len = len(processed_list[i])
        pre_ovs_out.append(min(int(pre_ov_in_list[i] * scale), actual_out_len // 4))
        post_ovs_out.append(min(int(post_ov_in_list[i] * scale), actual_out_len // 4))

    return stitch_chunks_seamless(processed_list, pre_ovs_out, post_ovs_out)


# ============================================================================
#finallimiter
# ============================================================================

def true_peak_limiter(data, sr, ceiling_db=-0.1, lookahead_ms=2.0, release_ms=150.0):
    from scipy.ndimage import minimum_filter1d

    squeeze = data.ndim == 1
    if squeeze:
        data = data.reshape(-1, 1)
    data = data.astype(np.float64)
    num_samples, num_channels = data.shape

    ceiling_lin   = 10.0 ** (ceiling_db / 20.0)
    la_samples    = max(1, int(sr * lookahead_ms / 1000.0))
    release_coeff = np.exp(-1.0 / (sr * release_ms / 1000.0))

    peak = np.max(np.abs(data), axis=1)

    with np.errstate(divide='ignore', invalid='ignore'):
        gain_required = np.where(
            peak > ceiling_lin,
            ceiling_lin / np.maximum(peak, 1e-30),
            1.0
        )

    # gain_ahead[i] = min(gain_required[i : i + la_samples])
    padded_gain = np.concatenate([gain_required, np.ones(la_samples + 1)])
    win_size    = la_samples + 1
    origin_val  = -(la_samples // 2)
    gain_ahead_padded = minimum_filter1d(
        padded_gain, size=win_size,
        mode='constant', cval=1.0,
        origin=origin_val
    )
    gain_ahead = gain_ahead_padded[:num_samples]

    gain_smooth = np.empty(num_samples, dtype=np.float64)
    g = 1.0
    for i in range(num_samples):
        t = gain_ahead[i]
        if t < g:
            g = t
        else:
            g = release_coeff * g + (1.0 - release_coeff) * t
        gain_smooth[i] = g

    if la_samples > 0:
        zero_pad     = np.zeros((la_samples, num_channels), dtype=np.float64)
        data_aligned = np.concatenate([zero_pad, data], axis=0)[:num_samples]
    else:
        data_aligned = data

    output = data_aligned * gain_smooth[:, np.newaxis]

    output = np.clip(output, -ceiling_lin, ceiling_lin)

    return (output.squeeze() if squeeze else output).astype(np.float32)


# ============================================================================
# ============================================================================


# ============================================================================
# Channel pipeline runner  (used by _process_one for L/R parallel execution)
# ============================================================================

def _run_channel_pipeline(mono, orig_mono, orig_sr, target_sr, params,
                           progress_callback=None):
    def _p(s, t, m):
        if progress_callback:
            progress_callback(s, t, m)

    duration = len(mono) / orig_sr

    # Step 1: super-resolution + HF smoothing
    if params['use_chunk'] and duration > params['max_chunk_sec']:
        def _cp(d, t, m): _p(10 + int(d / max(t, 1) * 50), 100, f'[Chunk] {m}')
        sr_result = process_audio_chunked(
            mono, orig_sr, target_sr,
            enable_octave_copy=params['enable_octave_copy'],
            split_freq=params['split_freq'],
            fft_size=params['fft_size'],
            cutoff_freq=params['cutoff_freq'],
            smoothing_mode=params['smooth_mode'],
            max_chunk_seconds=params['max_chunk_sec'],
            num_chunk_workers=params['chunk_workers'],
            num_smooth_workers_per_chunk=1,
            progress_callback=_cp)
    else:
        def _sp(s, t, m): _p(10 + int(s / t * 30), 100, f'[SuperRes] {m}')
        upsampled = super_resolve(
            mono, orig_sr, target_sr,
            enable_octave_copy=params['enable_octave_copy'],
            split_freq=params['split_freq'],
            progress_callback=_sp)
        def _sm(s, t, m): _p(40 + int(s / t * 20), 100, f'[Smooth] {m}')
        sr_result = smooth_hf_multithread(
            upsampled, target_sr,
            fft_size=params['fft_size'],
            cutoff_freq=params['cutoff_freq'],
            smoothing_mode=params['smooth_mode'],
            num_workers=params['smooth_workers'],
            progress_callback=_sm)

    result = sr_result

    # Step 2: faithful splice
    if params['enable_splice']:
        _p(65, 100, f"Faithful splice ({params['calc_crossover']} Hz)...")
        hq = high_quality_upsample(mono, orig_sr, target_sr)
        result = spectral_splice_20k(hq, result, target_sr,
                                     crossover=params['calc_crossover'],
                                     fade_hz=params['splice_fade_hz'])

    # Step 3: dynamic HF attenuation
    if params['enable_attenuate']:
        _p(78, 100, "Envelope-follower dynamic HF attenuation...")
        gf = 10 ** (params['att_gain_floor_db'] / 20.0)
        result = spectral_attenuate_hf(
            result, target_sr,
            orig_audio=orig_mono, orig_sr=orig_sr,
            ref_low=params['calc_att_low'], ref_high=params['calc_att_high'],
            target_low=params['calc_crossover'],
            smooth_ms=params['att_smooth_ms'],
            gain_floor=gf)

    # Step 4: smart EQ + auto flatten
    if params['enable_smart_eq'] or params['enable_auto_flatten']:
        _p(88, 100, "STFT smart spectrum repair...")
        result = spectral_smart_eq_and_repair(
            result, target_sr,
            enable_v_repair=params['enable_smart_eq'],
            eq_low=params['calc_eq_low'], eq_high=params['calc_eq_high'],
            max_boost_db=params['smart_eq_boost'],
            enable_auto_flatten=params['enable_auto_flatten'],
            flat_start_freq=params['flat_start_freq'],
            flat_max_boost=params['flat_max_boost'],
            flat_max_cut=params['flat_max_cut'])

    # Step 5: natural HF rolloff
    if params['enable_final_shelf']:
        _p(94, 100, "Ultra-HF power-law rolloff...")
        result = natural_hf_rolloff_filter(
            result, target_sr,
            f_start=params['final_shelf_freq'],
            total_db=params['final_shelf_gain'],
            curve_exponent=params['final_shelf_curve'],
            transition_hz=1500)

    return np.asarray(result, dtype=np.float32).ravel()

# ============================================================================
# Public API — everything above is importable by the GUI layer.
# Worker sub-processes (ProcessPoolExecutor) re-import this module;
# the guard below is also the entry point for the standalone CLI.
# ============================================================================

# ── Default parameters (identical to GUI defaults) ────────────────────────
_CLI_DEFAULTS = dict(
    target_sr           = 192000,
    # Step 1 — super-resolution
    use_chunk           = True,
    max_chunk_sec       = 5.0,
    chunk_workers       = max(1, min(4, mp.cpu_count() // 2)),
    smooth_workers      = max(1, mp.cpu_count() - 1),
    enable_octave_copy  = True,
    split_freq          = 50000,
    fft_size            = 512,
    cutoff_freq         = 22000,
    smooth_mode         = 'perfect_flat',
    # Step 2 — faithful splice
    enable_splice       = True,
    splice_crossover    = 20000,   # overridden by auto_adapt_sr
    splice_fade_hz      = 500,
    # Step 3 — dynamic HF attenuation
    enable_attenuate    = True,
    att_ref_low         = 19000,
    att_ref_high        = 20000,
    att_smooth_ms       = 10.0,
    att_gain_floor_db   = -20.0,
    # Step 4 — smart EQ + auto-flatten
    enable_smart_eq     = True,
    smart_eq_boost      = 15.0,
    enable_auto_flatten = True,
    flat_start_freq     = 15000,
    flat_max_boost      = 6.0,
    flat_max_cut        = 12.0,
    # Step 5 — power-law HF rolloff
    enable_final_shelf  = True,
    final_shelf_freq    = 20000,
    final_shelf_gain    = -18.0,
    final_shelf_curve   = 0.65,
    # Step 6 — limiter
    enable_limiter      = True,
    limiter_ceiling_db  = -0.1,
    limiter_lookahead   = 2.0,
    limiter_release     = 150.0,
    # Adaptive SR (auto-calc crossover / EQ range from input SR)
    auto_adapt_sr       = True,
)


def _cli_run(src_path: str, dst_path: str, quiet: bool = False) -> None:
    """
    Process one audio file through the full super-resolution pipeline
    using the same default parameters as the GUI.
    """
    import soundfile as sf
    from pathlib import Path

    src = Path(src_path)
    dst = Path(dst_path)

    def _log(msg: str) -> None:
        if not quiet:
            print(msg, flush=True)

    def _prog(step: int, total: int, msg: str) -> None:
        if not quiet:
            pct = int(step / max(total, 1) * 100)
            bar = '█' * (pct // 5) + '░' * (20 - pct // 5)
            print(f"\r  [{bar}] {pct:3d}%  {msg:<55}", end='', flush=True)

    _log(f"\n{'─'*60}")
    _log(f"  Input  : {src}")
    _log(f"  Output : {dst}")
    _log(f"  Target : {_CLI_DEFAULTS['target_sr'] // 1000} kHz  |  WAV 32-bit float")
    _log(f"{'─'*60}")

    # ── Read ──────────────────────────────────────────────────────────────
    _prog(0, 100, "Reading audio…")
    orig_data, orig_sr = sf.read(str(src))
    orig_data = orig_data.astype(np.float32)
    target_sr = _CLI_DEFAULTS['target_sr']
    p         = dict(_CLI_DEFAULTS)     # local mutable copy

    # ── Adaptive SR parameter calculation (mirrors GUI logic) ─────────────
    def _calc_adapt(sr):
        roll = (sr / 2.0) - 2200
        return dict(
            calc_crossover = int(roll),
            calc_eq_low    = int(roll - 1000),
            calc_eq_high   = int(sr - int(roll - 1000)),
            calc_att_low   = int(roll - 2000),
            calc_att_high  = int(roll - 1000),
        )

    if p['auto_adapt_sr']:
        p.update(_calc_adapt(orig_sr))
    else:
        p['calc_crossover'] = p['splice_crossover']
        p['calc_eq_low']    = 19000
        p['calc_eq_high']   = 23000
        p['calc_att_low']   = p['att_ref_low']
        p['calc_att_high']  = p['att_ref_high']

    # ── 44.1 kHz pre-processing ───────────────────────────────────────────
    if orig_sr == 44100:
        _log('')
        _prog(3, 100, 'Pre-upsampling 44.1→48 kHz…')
        def _p0(s, t, m): _prog(3 + int(s / max(t, 1) * 6), 100, f'[44.1→48k] {m}')
        orig_data = upsample_441_to_48k(orig_data, progress_callback=_p0)
        orig_sr   = 48000
        if p['auto_adapt_sr']:
            p.update(_calc_adapt(orig_sr))

    # ── Build run_params dict expected by _run_channel_pipeline ───────────
    run_p = dict(
        use_chunk           = p['use_chunk'],
        max_chunk_sec       = p['max_chunk_sec'],
        chunk_workers       = p['chunk_workers'],
        smooth_workers      = p['smooth_workers'],
        enable_octave_copy  = p['enable_octave_copy'],
        split_freq          = p['split_freq'],
        fft_size            = p['fft_size'],
        cutoff_freq         = p['cutoff_freq'],
        smooth_mode         = p['smooth_mode'],
        enable_splice       = p['enable_splice'],
        calc_crossover      = p['calc_crossover'],
        splice_fade_hz      = p['splice_fade_hz'],
        enable_attenuate    = p['enable_attenuate'],
        calc_att_low        = p['calc_att_low'],
        calc_att_high       = p['calc_att_high'],
        att_smooth_ms       = p['att_smooth_ms'],
        att_gain_floor_db   = p['att_gain_floor_db'],
        enable_smart_eq     = p['enable_smart_eq'],
        enable_auto_flatten = p['enable_auto_flatten'],
        calc_eq_low         = p['calc_eq_low'],
        calc_eq_high        = p['calc_eq_high'],
        smart_eq_boost      = p['smart_eq_boost'],
        flat_start_freq     = p['flat_start_freq'],
        flat_max_boost      = p['flat_max_boost'],
        flat_max_cut        = p['flat_max_cut'],
        enable_final_shelf  = p['enable_final_shelf'],
        final_shelf_freq    = p['final_shelf_freq'],
        final_shelf_gain    = p['final_shelf_gain'],
        final_shelf_curve   = p['final_shelf_curve'],
    )

    # ── Stereo or mono channel pipeline ───────────────────────────────────
    is_stereo = orig_data.ndim == 2 and orig_data.shape[1] >= 2
    _log('')

    if is_stereo:
        ch_smooth = max(1, p['smooth_workers'] // 2)
        ch_chunk  = max(1, p['chunk_workers']  // 2)
        run_p['smooth_workers'] = ch_smooth
        run_p['chunk_workers']  = ch_chunk

        ch_results = [None, None]
        ch_errors  = [None, None]

        def _run_ch(ci):
            try:
                mono = orig_data[:, ci]
                cb   = (lambda s, t, m: _prog(s, t, m)) if ci == 0 else None
                ch_results[ci] = _run_channel_pipeline(
                    mono, mono, orig_sr, target_sr, run_p, cb)
            except Exception:
                import traceback
                ch_errors[ci] = traceback.format_exc()

        import threading
        tl = threading.Thread(target=_run_ch, args=(0,), daemon=True)
        tr = threading.Thread(target=_run_ch, args=(1,), daemon=True)
        tl.start(); tr.start(); tl.join(); tr.join()

        if ch_errors[0] or ch_errors[1]:
            raise RuntimeError(
                f"Channel pipeline failed:\nL: {ch_errors[0]}\nR: {ch_errors[1]}")
        result = np.column_stack([ch_results[0], ch_results[1]])
    else:
        mono   = orig_data.ravel()
        result = _run_channel_pipeline(
            mono, mono, orig_sr, target_sr, run_p,
            lambda s, t, m: _prog(s, t, m))

    # ── True-peak limiter ─────────────────────────────────────────────────
    if p['enable_limiter']:
        _prog(97, 100, f"Limiter ({p['limiter_ceiling_db']:+.2f} dBFS)…")
        result = true_peak_limiter(result, target_sr,
                                   ceiling_db=p['limiter_ceiling_db'],
                                   lookahead_ms=p['limiter_lookahead'],
                                   release_ms=p['limiter_release'])

    # ── Write ─────────────────────────────────────────────────────────────
    _prog(99, 100, "Writing output…")
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), result, target_sr, subtype='FLOAT')
    _prog(100, 100, "Done.")
    _log(f"\n\n  ✓ Saved → {dst}\n")


if __name__ == '__main__':
    import argparse
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(
        prog='audio_super_resolution_core',
        description=(
            'Audio Super-Resolution CLI  —  '
            'upsample WAV/FLAC to 192 kHz using default GUI settings.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Examples:\n'
            '  python audio_super_resolution_core.py -i song.flac -o song_192k.wav\n'
            '  python audio_super_resolution_core.py -i ./music/ -o ./output/\n'
            '  python audio_super_resolution_core.py -i song.flac -o song_192k.wav -q\n'
        ),
    )
    parser.add_argument('-i', '--input',  required=True,
                        help='Input WAV/FLAC file, or a folder for batch processing.')
    parser.add_argument('-o', '--output', required=True,
                        help='Output WAV file, or output folder (for batch mode).')
    parser.add_argument('-q', '--quiet',  action='store_true',
                        help='Suppress progress output (errors still printed).')

    args = parser.parse_args()

    src = Path(args.input)
    dst = Path(args.output)

    # ── Collect files ─────────────────────────────────────────────────────
    if src.is_dir():
        files = sorted(f for f in src.iterdir()
                       if f.suffix.lower() in ('.wav', '.flac'))
        if not files:
            print(f"Error: no WAV/FLAC files found in '{src}'", file=sys.stderr)
            sys.exit(1)
        # In batch mode -o must be a directory (created if needed)
        out_dir = dst
        pairs   = [(f, out_dir / (f.stem + '_192k.wav')) for f in files]
    elif src.is_file():
        if src.suffix.lower() not in ('.wav', '.flac'):
            print(f"Error: '{src}' is not a WAV or FLAC file.", file=sys.stderr)
            sys.exit(1)
        # If -o looks like a directory (no suffix), treat as output folder
        if dst.suffix == '':
            pairs = [(src, dst / (src.stem + '_192k.wav'))]
        else:
            pairs = [(src, dst)]
    else:
        print(f"Error: input '{src}' not found.", file=sys.stderr)
        sys.exit(1)

    # ── Process ───────────────────────────────────────────────────────────
    errors = []
    for idx, (in_f, out_f) in enumerate(pairs):
        if len(pairs) > 1 and not args.quiet:
            print(f"\n[{idx+1}/{len(pairs)}] {in_f.name}", flush=True)
        try:
            _cli_run(str(in_f), str(out_f), quiet=args.quiet)
        except Exception as exc:
            import traceback
            msg = f"{in_f.name}: {exc}"
            errors.append(msg)
            print(f"\n  ✗ {msg}", file=sys.stderr)
            if not args.quiet:
                traceback.print_exc()

    if errors:
        print(f"\n{len(errors)} file(s) failed.", file=sys.stderr)
        sys.exit(1)
    elif not args.quiet and len(pairs) > 1:
        print(f"\n✓ All {len(pairs)} file(s) processed successfully.", flush=True)
