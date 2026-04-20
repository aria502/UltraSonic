#!/usr/bin/env python3
"""
audio_mp3_restore_core.py  —  HF restoration back-end for low-quality audio.

Uses the same aliasing-based spectral synthesis principle as
audio_super_resolution_core.py, but works entirely within the original
sample rate, reconstructing missing high-frequency content from a detected
(or manually specified) cutoff frequency up to the file's own Nyquist.

Requires: audio_super_resolution_core.py in the same directory.

Typical use-cases
-----------------
  128 kbps MP3  →  content cut at ~16 kHz  →  restore 16–22 kHz
   96 kbps MP3  →  content cut at ~14 kHz  →  restore 14–22 kHz
   64 kbps MP3  →  content cut at ~11 kHz  →  restore 11–22 kHz

The output is written at the same sample rate as the input.  No SR
conversion is performed.
"""

import numpy as np
from scipy import signal
import multiprocessing as mp
import threading

# ── shared DSP functions from the super-resolution core ──────────────────
from audio_super_resolution_core import (
    _apply_pyfftw,
    izotope_resample,
    spectral_extension,
    high_quality_upsample,
    spectral_splice_20k,
    spectral_attenuate_hf,
    spectral_smart_eq_and_repair,
    true_peak_limiter,
)

# Activate pyfftw in the main process (same as super_resolution_core).
_PYFFTW_AVAILABLE = _apply_pyfftw(num_threads=None)


# ============================================================================
# Auto-detection of the MP3 brick-wall cutoff
# ============================================================================

def detect_mp3_cutoff(data: np.ndarray, sr: int,
                      search_low: int = 8000,
                      search_high: int = None,
                      threshold_db: float = 30.0) -> int:
    """
    Auto-detect the HF brick-wall cutoff in an MP3-decoded audio file.

    MP3 encoders apply a sharp low-pass filter before encoding; the
    resulting spectrum shows a distinctive cliff-edge that this function
    locates by scanning downward from the top of the search window until
    it finds a bin whose average level is within ``threshold_db`` of the
    2–8 kHz reference band.

    Parameters
    ----------
    data            : audio array (mono or stereo, any float dtype)
    sr              : sample rate of *data*
    search_low      : lowest frequency (Hz) the cutoff can be reported as
    search_high     : highest frequency (Hz) to scan; defaults to 97 % of Nyquist
    threshold_db    : a bin is considered "active" if it is less than this many dB
                      below the 2–8 kHz reference median (default 30 dB)

    Returns
    -------
    Detected cutoff in Hz, rounded to the nearest 500 Hz and clamped to
    [search_low, search_high].
    """
    if search_high is None:
        search_high = int(sr / 2 * 0.97)

    mono = (data.mean(axis=1) if data.ndim == 2 else data.ravel()).astype(np.float64)
    max_samples = int(60 * sr)
    if len(mono) > max_samples:
        mono = mono[:max_samples]

    n_fft   = 8192
    noverlap = n_fft * 3 // 4
    freqs, _, Zxx = signal.stft(mono, fs=sr, nperseg=n_fft,
                                  noverlap=noverlap, window='hann')

    avg_mag = np.mean(np.abs(Zxx), axis=1)
    avg_db  = 20.0 * np.log10(avg_mag + 1e-12)

    # Reference level: 2–8 kHz median (never cut by any MP3 encoder)
    ref_lo = np.searchsorted(freqs, 2000)
    ref_hi = np.searchsorted(freqs, 8000)
    ref_db = float(np.median(avg_db[ref_lo:ref_hi]))

    floor_db = ref_db - threshold_db
    lo_bin   = np.searchsorted(freqs, search_low)
    hi_bin   = min(np.searchsorted(freqs, search_high), len(freqs) - 1)

    # Scan downward; first bin that clears the floor is the cutoff
    cutoff_bin = lo_bin
    for i in range(hi_bin, lo_bin, -1):
        if avg_db[i] >= floor_db:
            cutoff_bin = i
            break

    raw     = float(freqs[cutoff_bin])
    rounded = round(raw / 500.0) * 500.0
    return int(np.clip(rounded, search_low, search_high))


# ============================================================================
# Core HF synthesis  (the new counterpart of super_resolve)
# ============================================================================

def restore_hf_band(data: np.ndarray, sr: int, cutoff_freq: int,
                    aliasing: float = 0.72,
                    spectral_strength: float = 0.10) -> np.ndarray:
    """
    Synthesise missing HF content above *cutoff_freq* up to sr/2.

    Mechanism (same family as izotope_resample in super_resolution_core)
    ---------------------------------------------------------------------
    1. **Downsample** to virtual_sr = 2 × cutoff_freq using a high-quality
       Kaiser-windowed polyphase filter.  This produces a clean band-limited
       version of the signal at a sample rate whose Nyquist equals the cutoff.

    2. **Spectral extension** at virtual_sr seeds harmonic energy just below
       the virtual Nyquist, giving the subsequent aliaser richer material to
       work with.

    3. **Aliasing upsample** back to the original sr.  The nearest-neighbour
       component of izotope_resample creates spectral images at multiples of
       virtual_sr ± f; the first image fold lands in [cutoff_freq, sr/2],
       which is exactly the band we want to fill.  The *aliasing* parameter
       controls how strong this synthesis is (0 = no synthesis, 1 = maximum).

    The returned array is at the original sr and contains synthesised
    full-band content; the caller's splice step discards it below cutoff_freq.

    Parameters
    ----------
    data             : 1-D float32 mono array
    sr               : sample rate of *data*
    cutoff_freq      : Hz — synthesis starts here and fills up to sr/2
    aliasing         : nearest-neighbour blend ratio for izotope_resample
    spectral_strength: strength passed to spectral_extension at virtual_sr
    """
    virtual_sr = int(cutoff_freq * 2)

    if virtual_sr >= sr:
        # Cutoff is already near Nyquist — no room to synthesise
        return data.astype(np.float32)

    # Step 1: high-quality downsample  (sr → virtual_sr)
    low_bw = high_quality_upsample(data, sr, virtual_sr)

    # Step 2: spectral extension at virtual_sr
    # orig_nyq slightly below cutoff_freq so the extension seeds bins in the
    # upper part of the virtual band, which the aliaser will later fold upward.
    orig_nyq_virtual = float(cutoff_freq) * 0.82
    low_bw = spectral_extension(low_bw, virtual_sr,
                                 orig_nyq_virtual, strength=spectral_strength)

    # Step 3: aliasing upsample  (virtual_sr → sr)
    # Nearest-neighbour images at (virtual_sr - f) for f in [0, cutoff_freq]
    # land at [cutoff_freq, sr/2], filling the target band.
    synthesized = izotope_resample(low_bw, virtual_sr, sr, aliasing)

    return synthesized.astype(np.float32)


# ============================================================================
# Full per-channel restoration pipeline  (4 stages, no power-law rolloff)
# ============================================================================

def _run_restore_pipeline(mono: np.ndarray, orig_sr: int,
                           cutoff_freq: int, params: dict,
                           progress_callback=None) -> np.ndarray:
    """
    Run the complete HF restoration pipeline on one mono channel.

    Stages
    ------
    1. HF synthesis      — restore_hf_band (aliasing resample)
    2. Faithful splice   — spectral_splice_20k at cutoff_freq
    3. Dynamic attenuation — spectral_attenuate_hf (envelope follower)
    4. STFT repair       — V-notch fill + auto-flatten around the splice point
    (Stage 5 of the super-resolution pipeline — power-law rolloff — is
     intentionally omitted for restoration; we want to restore, not attenuate.)

    Parameters
    ----------
    mono        : 1-D float32 mono channel
    orig_sr     : sample rate (Hz)
    cutoff_freq : detected or user-specified cutoff (Hz)
    params      : parameter dict (keys match _CLI_DEFAULTS)
    """
    def _p(s, t, m):
        if progress_callback:
            progress_callback(s, t, m)

    nyq = orig_sr / 2.0

    # Derive frequency ranges for each stage from the cutoff
    calc_crossover = cutoff_freq
    calc_eq_low    = max(1000,        cutoff_freq - 1500)
    calc_eq_high   = min(int(nyq) - 500, cutoff_freq + 4000)
    calc_att_low   = max(500,         cutoff_freq - 3000)
    calc_att_high  = cutoff_freq
    flat_start     = max(1000,        cutoff_freq - 2500)

    # ── Stage 1: HF synthesis ─────────────────────────────────────────
    _p(5, 100, f"Synthesising HF content above {cutoff_freq} Hz…")
    synthesized = restore_hf_band(
        mono, orig_sr, cutoff_freq,
        aliasing          = params.get('synthesis_aliasing',   0.72),
        spectral_strength = params.get('spectral_strength',    0.10))
    _p(38, 100, "HF synthesis complete")

    # ── Stage 2: Faithful splice ──────────────────────────────────────
    # Below cutoff_freq: original signal (bit-accurate original content).
    # Above cutoff_freq: synthesised signal.
    if params.get('enable_splice', True):
        _p(42, 100, f"Faithful splice at {calc_crossover} Hz…")
        result = spectral_splice_20k(
            mono.astype(np.float32),   # original clean signal
            synthesized,               # synthesised HF extension
            orig_sr,
            crossover = calc_crossover,
            fade_hz   = params.get('splice_fade_hz', 500))
    else:
        result = synthesized

    # ── Stage 3: Envelope-follower dynamic HF attenuation ────────────
    # Derives gain envelope from the band just below the cutoff so that
    # synthesised content follows the natural dynamic of the original.
    if params.get('enable_attenuate', True):
        _p(58, 100, "Envelope-follower dynamic HF attenuation…")
        gf = 10.0 ** (params.get('att_gain_floor_db', -20.0) / 20.0)
        result = spectral_attenuate_hf(
            result, orig_sr,
            orig_audio = mono.astype(np.float32),
            orig_sr    = orig_sr,
            ref_low    = calc_att_low,
            ref_high   = calc_att_high,
            target_low = calc_crossover,
            smooth_ms  = params.get('att_smooth_ms', 10.0),
            gain_floor = gf)

    # ── Stage 4: STFT spectral repair ────────────────────────────────
    # V-notch: fills the level dip at the splice point.
    # Auto-flatten: evens out peaks/troughs in the restored band.
    if params.get('enable_smart_eq', True) or params.get('enable_auto_flatten', True):
        _p(75, 100, "STFT spectral repair (V-notch + auto-flatten)…")
        result = spectral_smart_eq_and_repair(
            result, orig_sr,
            enable_v_repair    = params.get('enable_smart_eq',     True),
            eq_low             = calc_eq_low,
            eq_high            = calc_eq_high,
            max_boost_db       = params.get('smart_eq_boost',      12.0),
            enable_auto_flatten= params.get('enable_auto_flatten', True),
            flat_start_freq    = flat_start,
            flat_max_boost     = params.get('flat_max_boost',       6.0),
            flat_max_cut       = params.get('flat_max_cut',        12.0),
            notch_center       = cutoff_freq,
            notch_width        = 2000)

    _p(95, 100, "Pipeline complete")
    return np.asarray(result, dtype=np.float32).ravel()


# ============================================================================
# CLI defaults  (identical to GUI defaults)
# ============================================================================

_CLI_DEFAULTS = dict(
    # Cutoff detection
    auto_detect_cutoff  = True,
    manual_cutoff       = 16000,    # used when auto_detect_cutoff is False
    # Stage 1 — synthesis
    synthesis_aliasing  = 0.72,     # 0 = no aliasing, 1 = full nearest-neighbour
    spectral_strength   = 0.10,     # spectral extension seed strength
    # Stage 2 — faithful splice
    enable_splice       = True,
    splice_fade_hz      = 500,
    # Stage 3 — dynamic attenuation
    enable_attenuate    = True,
    att_smooth_ms       = 10.0,
    att_gain_floor_db   = -20.0,
    # Stage 4 — STFT repair
    enable_smart_eq     = True,
    smart_eq_boost      = 12.0,
    enable_auto_flatten = True,
    flat_max_boost      =  6.0,
    flat_max_cut        = 12.0,
    # Stage 6 — limiter (applied joint-stereo after channel merge)
    enable_limiter      = True,
    limiter_ceiling_db  = -0.1,
    limiter_lookahead   =  2.0,
    limiter_release     = 150.0,
)


# ============================================================================
# CLI runner  (used by both the __main__ block and importers)
# ============================================================================

def _cli_run(src_path: str, dst_path: str,
             cutoff_freq: int = None,
             quiet: bool = False) -> int:
    """
    Process one audio file through the full restoration pipeline.

    Parameters
    ----------
    src_path    : input WAV/FLAC path
    dst_path    : output WAV path (written at the same SR as input)
    cutoff_freq : Hz — if None, auto-detect from the file
    quiet       : suppress progress output

    Returns
    -------
    The cutoff frequency (Hz) that was used.
    """
    import soundfile as sf
    from pathlib import Path

    src = Path(src_path)
    dst = Path(dst_path)

    def _log(msg):
        if not quiet:
            print(msg, flush=True)

    def _prog(s, t, msg):
        if not quiet:
            pct = int(s / max(t, 1) * 100)
            bar = '█' * (pct // 5) + '░' * (20 - pct // 5)
            print(f"\r  [{bar}] {pct:3d}%  {msg:<55}", end='', flush=True)

    _log(f"\n{'─'*60}")
    _log(f"  Input  : {src}")
    _log(f"  Output : {dst}")

    # Read
    _prog(0, 100, "Reading audio…")
    orig_data, orig_sr = sf.read(str(src))
    orig_data = orig_data.astype(np.float32)

    # Auto-detect or validate cutoff
    if cutoff_freq is None:
        _prog(1, 100, "Auto-detecting HF cutoff…")
        cutoff_freq = detect_mp3_cutoff(orig_data, orig_sr)
        _log(f"\n  Detected cutoff : {cutoff_freq} Hz")
    else:
        nyq = orig_sr // 2
        cutoff_freq = int(np.clip(cutoff_freq, 8000, nyq - 1000))
        _log(f"\n  Cutoff (manual) : {cutoff_freq} Hz")

    _log(f"  Sample rate     : {orig_sr} Hz  (output unchanged)")
    _log(f"{'─'*60}")

    p = dict(_CLI_DEFAULTS)
    is_stereo = orig_data.ndim == 2 and orig_data.shape[1] >= 2

    if is_stereo:
        ch_results = [None, None]
        ch_errors  = [None, None]

        def _run_ch(ci):
            try:
                mono = orig_data[:, ci]
                cb   = (lambda s, t, m: _prog(s, t, m)) if ci == 0 else None
                ch_results[ci] = _run_restore_pipeline(
                    mono, orig_sr, cutoff_freq, p, cb)
            except Exception:
                import traceback
                ch_errors[ci] = traceback.format_exc()

        _log('')
        tl = threading.Thread(target=_run_ch, args=(0,), daemon=True)
        tr = threading.Thread(target=_run_ch, args=(1,), daemon=True)
        tl.start(); tr.start(); tl.join(); tr.join()

        if ch_errors[0] or ch_errors[1]:
            raise RuntimeError(
                f"Channel pipeline failed:\nL: {ch_errors[0]}\nR: {ch_errors[1]}")
        result = np.column_stack([ch_results[0], ch_results[1]])
    else:
        _log('')
        mono   = orig_data.ravel()
        result = _run_restore_pipeline(
            mono, orig_sr, cutoff_freq, p,
            lambda s, t, m: _prog(s, t, m))

    # True-peak limiter (joint stereo, same as super_resolution_core)
    if p['enable_limiter']:
        _prog(97, 100, f"Limiter ({p['limiter_ceiling_db']:+.2f} dBFS)…")
        result = true_peak_limiter(result, orig_sr,
                                   ceiling_db   = p['limiter_ceiling_db'],
                                   lookahead_ms = p['limiter_lookahead'],
                                   release_ms   = p['limiter_release'])

    _prog(99, 100, "Writing output…")
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), result, orig_sr, subtype='FLOAT')
    _prog(100, 100, "Done.")
    _log(f"\n\n  ✓ Saved → {dst}\n")

    return cutoff_freq


# ============================================================================
# Entry point  (CLI)
# ============================================================================

if __name__ == '__main__':
    import argparse
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(
        prog='audio_mp3_restore_core',
        description=(
            'Audio HF Restore CLI  —  reconstruct missing high-frequency '
            'content in low-quality MP3-decoded WAV/FLAC files.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Examples:\n'
            '  # Auto-detect cutoff and restore:\n'
            '  python audio_mp3_restore_core.py -i song.wav -o song_restored.wav\n'
            '\n'
            '  # Manually specify a 14 kHz cutoff (128 kbps MP3):\n'
            '  python audio_mp3_restore_core.py -i song.wav -o song_restored.wav -c 14000\n'
            '\n'
            '  # Only detect and print the cutoff, do not process:\n'
            '  python audio_mp3_restore_core.py -i song.wav --detect-only\n'
            '\n'
            '  # Batch-process a folder, suppress progress:\n'
            '  python audio_mp3_restore_core.py -i ./mp3s/ -o ./restored/ -q\n'
        ),
    )
    parser.add_argument('-i', '--input',  required=True,
                        help='Input WAV/FLAC file, or a folder for batch mode.')
    parser.add_argument('-o', '--output', default=None,
                        help='Output WAV file or folder.  Required unless --detect-only.')
    parser.add_argument('-c', '--cutoff', type=int, default=None,
                        help='Manually specify the HF cutoff in Hz (8000–22050). '
                             'Skips auto-detection.')
    parser.add_argument('--detect-only', action='store_true',
                        help='Only detect and print the cutoff frequency; do not process.')
    parser.add_argument('-q', '--quiet',  action='store_true',
                        help='Suppress progress output (errors still go to stderr).')

    args = parser.parse_args()

    src = Path(args.input)

    # ── Collect files ─────────────────────────────────────────────────
    if src.is_dir():
        files = sorted(f for f in src.iterdir()
                       if f.suffix.lower() in ('.wav', '.flac'))
        if not files:
            print(f"Error: no WAV/FLAC files found in '{src}'", file=sys.stderr)
            sys.exit(1)
        out_dir = Path(args.output) if args.output else src.parent / (src.name + '_restored')
        pairs   = [(f, out_dir / (f.stem + '_restored.wav')) for f in files]
    elif src.is_file():
        if src.suffix.lower() not in ('.wav', '.flac'):
            print(f"Error: '{src}' is not a WAV or FLAC file.", file=sys.stderr)
            sys.exit(1)
        if args.output:
            dst = Path(args.output)
            if dst.suffix == '':
                dst = dst / (src.stem + '_restored.wav')
        else:
            dst = src.with_name(src.stem + '_restored.wav')
        pairs = [(src, dst)]
    else:
        print(f"Error: input '{src}' not found.", file=sys.stderr)
        sys.exit(1)

    # ── detect-only mode ──────────────────────────────────────────────
    if args.detect_only:
        import soundfile as sf
        for in_f, _ in pairs:
            try:
                data, sr = sf.read(str(in_f))
                c = detect_mp3_cutoff(data.astype(np.float32), sr)
                print(f"{in_f.name}: detected cutoff = {c} Hz")
            except Exception as exc:
                print(f"{in_f.name}: ERROR — {exc}", file=sys.stderr)
        sys.exit(0)

    # ── require -o in normal mode ─────────────────────────────────────
    if args.output is None and not src.is_dir():
        # We already set dst to a default above, so this branch won't trigger.
        pass

    # ── Process ───────────────────────────────────────────────────────
    errors = []
    for idx, (in_f, out_f) in enumerate(pairs):
        if len(pairs) > 1 and not args.quiet:
            print(f"\n[{idx+1}/{len(pairs)}] {in_f.name}", flush=True)
        try:
            _cli_run(str(in_f), str(out_f),
                     cutoff_freq = args.cutoff,
                     quiet       = args.quiet)
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
