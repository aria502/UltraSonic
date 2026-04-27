# Audio Super-Resolution

A DSP-based audio super-resolution tool that upsamples standard audio files
(WAV / FLAC) to high sample rates up to 192 kHz, reconstructing plausible
high-frequency content above the original Nyquist frequency through a
multi-stage aliasing, spectral copying, and adaptive EQ approach — entirely
in Python with no neural-network inference required.

---
# Lossy codec audio repair(New)
Before(MP3)
<img width="789" height="860" alt="屏幕截图 2026-04-27 212014" src="https://github.com/user-attachments/assets/db09f02d-e8d4-4d0a-ac2c-05b79770c29e" />

After(Repaired with audio_mp3_restore)
<img width="785" height="860" alt="屏幕截图 2026-04-27 212024" src="https://github.com/user-attachments/assets/4af80fbf-bda0-453c-bd19-9efa2ec9af0b" />

# Sample Audio Spectrogram (Upsampled to 192khz)

<img width="1599" height="871" alt="SonicVsualiser_192khz" src="https://github.com/user-attachments/assets/1dec71de-9aa1-4e98-9526-4278d7f92472" />

# Spectrogram Compare(Before:original 44.1khz sample rate)
<img width="1511" height="767" alt="before" src="https://github.com/user-attachments/assets/dd2d16af-c0a4-4db9-9920-78e32fb77230" />

# Spectrogram Compare(After:upsampled to 192khz)
<img width="1505" height="772" alt="after" src="https://github.com/user-attachments/assets/50235eef-2927-4099-a966-0c353b07ae46" />

## Features

- **Upsampling up to 192 kHz** — iterative aliasing-based resampling followed
  by multi-pass spectral extension fills content from the original Nyquist
  frequency all the way to 96 kHz
- **44.1 kHz auto pre-processing** — files at 44.1 kHz are automatically
  converted to 48 kHz first, filling the 22–24 kHz gap before the main
  pipeline runs
- **6-step DSP pipeline** — each stage is independently toggleable via the GUI
  (see [Pipeline](#pipeline) below)
- **True stereo** — left and right channels run through the full pipeline in
  parallel threads, then are recombined for the final limiter pass
- **Smart chunked processing** — long files are split into overlapping chunks
  processed by a `ProcessPoolExecutor`, keeping memory usage bounded
- **PyQt6 GUI** — scrollable, HiDPI-aware interface with collapsible advanced
  settings and an embedded before/after spectrogram preview (0–96 kHz)
- **Command-line interface** — headless batch processing with a single command,
  no GUI required
- **Optional FFTW3 acceleration** — if `pyfftw` is installed, all FFT
  operations are transparently replaced with FFTW3, giving a 2–5× speed-up

---

## Requirements

- Python >= 3.10
- See [`requirements.txt`](requirements.txt) for the full dependency list

Quick install:

```bash
pip install -r requirements.txt
```

With the optional FFTW3 accelerator:

```bash
pip install -r requirements.txt pyfftw>=0.13.1
```

> **pyfftw on Linux / macOS** requires the FFTW3 C library to be installed
> separately before `pip install pyfftw`:
> ```bash
> # Debian / Ubuntu
> sudo apt install libfftw3-dev
> # macOS (Homebrew)
> brew install fftw
> ```
> On Windows, the `pyfftw` wheel bundles the library automatically.

---

## Project Structure

```
audio_super_resolution_core.py   # DSP back-end — all signal processing logic
audio_super_resolution_gui.py    # PyQt6 front-end — imports from core
requirements.txt
README.md
```

The back-end and front-end are intentionally separated.  `core.py` has no GUI
dependencies and can be imported by other scripts or run directly from the
command line.  Worker sub-processes spawned by `ProcessPoolExecutor` re-import
`core.py` safely because no side-effectful code runs at module level.

---

## Usage

### GUI

```bash
python audio_super_resolution_gui.py
```

Select an input file or folder, choose an output folder, then click
**Start Processing**.  The advanced settings panel exposes every pipeline
parameter.  Enable **Preview Spectrogram** to display a before/after
frequency plot after processing a single file.

### Command Line

Process a single file:

```bash
python audio_super_resolution_core.py -i input.flac -o output_192k.wav
```

Batch-process an entire folder:

```bash
python audio_super_resolution_core.py -i ./music/ -o ./output/
```

Suppress progress output (errors are still printed to stderr):

```bash
python audio_super_resolution_core.py -i input.flac -o output_192k.wav -q
```

All CLI runs use the same default parameters as the GUI.  Output is always
32-bit float WAV at 192 kHz.

#### CLI arguments

| Argument | Description |
|---|---|
| `-i`, `--input` | Input WAV/FLAC file, or a folder for batch mode |
| `-o`, `--output` | Output WAV file, or output folder for batch mode |
| `-q`, `--quiet` | Suppress progress bar (errors still go to stderr) |

---

## Pipeline

Each file passes through six sequential stages.  All stages are enabled by
default and can be toggled individually in the GUI.

### Stage 1 — Aliasing-based super-resolution

The core upsampling uses a deliberately **imperfect** resampler (`izotope_resample`)
that blends linear interpolation with nearest-neighbour interpolation.  The
nearest-neighbour component introduces controlled aliasing that folds
existing harmonic content into the newly created frequency band above the
original Nyquist, mimicking the spectral character of natively high-resolution
recordings.  The aliasing amount decreases linearly across the upsampling
steps (from 0.8 at the start to 0.5 at the end) so that the lowest frequency
extensions are the most aggressive and the highest are the most conservative.

Upsampling is performed in multiple steps (48k → 50k → 60k → … → 192k),
with periodic spectral extension passes that copy energy from the
octave-below band into freshly created bins every four steps.

After upsampling, a multi-threaded overlap-add STFT smoother flattens any
sharp discontinuities in the synthesised HF region above the chosen cutoff
frequency (default 22 kHz).

### Stage 2 — Faithful splice

A high-quality Kaiser-windowed resampler (`resample_poly`) produces a
phase-accurate reference copy of the original signal at the target sample
rate.  This clean copy is crossover-filtered and spliced with the
aliased super-resolved signal at the original Nyquist frequency, ensuring
that all content below the original cutoff is bit-identical to what a
lossless resampler would produce.

### Stage 3 — Envelope-follower dynamic HF attenuation

An envelope follower tracks the short-time RMS energy of the original
signal in a narrow reference band just below the Nyquist frequency.
This envelope is used as a gain-control signal applied to the synthesised
content above the crossover — so silent passages receive little or no
artificial HF content, while loud passages receive proportionally more,
maintaining a natural dynamic relationship across the spectrum.

### Stage 4 — STFT spectral repair

A two-part STFT-domain correction:

1. **V-notch repair** — detects the characteristic dip that forms at the
   original Nyquist frequency after splicing and fills it by interpolating
   the average spectral level from the bands just below and just above the
   notch, with the gain smoothed across time by a 2-D Gaussian kernel.
2. **Auto-flatten** — a median/Gaussian-filtered version of the per-bin
   average magnitude serves as a target envelope; bins that deviate from
   this envelope are gently pushed towards it within configurable boost/cut
   limits, suppressing narrow tonal spikes and dark troughs that can arise
   from the aliasing resampler.

### Stage 5 — Ultra-HF power-law rolloff

A frequency-domain filter applies a smooth power-law attenuation curve
above a configurable start frequency (default 20 kHz).  The rolloff shape
follows `gain_dB = total_dB × ((f − f_start) / (f_nyq − f_start)) ^ exponent`,
with a cosine-tapered transition zone at the boundaries to avoid ringing.
This replicates the natural high-frequency roll-off found in real
microphone chains and analogue-tape masters.

### Stage 6 — Look-ahead true-peak limiter

A look-ahead (default 2 ms) true-peak limiter operates on the
stereo-recombined output, catching inter-sample overs introduced by the
upsampling process.  Gain reduction is applied with a fast attack
(instantaneous) and a slow exponential release (default 150 ms), keeping
peaks below the configurable ceiling (default −0.1 dBFS) with minimal
audible distortion.

---

## Advanced Options (GUI)

| Tab | Option | Default | Description |
|---|---|---|---|
| Spectrum Repair | Auto-flatten | On | STFT median EQ to remove narrow peaks/dips |
| Spectrum Repair | V-notch repair | On | Fill the Nyquist splice dip |
| Spectrum Repair | Auto-detect SR | On | Derive crossover and EQ range from input SR automatically |
| Spectrum Repair | Faithful splice | On | Lossless crossover below original Nyquist |
| Spectrum Repair | Dynamic attenuation | On | Envelope-follower HF gain modulation |
| Spectrum Repair | Power-law rolloff | On | Natural arc HF attenuation above 20 kHz |
| Spectrum Repair | True-peak limiter | On | Look-ahead inter-sample peak control |
| Sample & Threads | Target SR | 192000 | Output sample rate (48k / 96k / 192k) |
| Sample & Threads | FFT size | 512 | STFT window for HF smoothing |
| Sample & Threads | Smooth start | 22000 Hz | Frequency above which smoothing is applied |
| Sample & Threads | Threads | CPU count − 1 | Worker threads for HF smoothing |
| Octave Copy | Octave copy | On | Copy spectral bins from the octave below into the newly created band |
| Octave Copy | Smart chunking | On | Split long files into parallel chunks for lower memory usage |

---

## Output Format

All output files are written as **32-bit float WAV** at the target sample
rate (default 192 kHz).  The floating-point subtype preserves full dynamic
range without any quantisation noise; most DAWs and audio tools read it
natively.  Batch mode appends `_192k` to the original filename.

---

## License

This project is released under the [MIT License](LICENSE).
