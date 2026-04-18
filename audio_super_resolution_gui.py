#!/usr/bin/env python3
"""
audio_super_resolution_gui.py  —  PyQt6 front-end
Requires : audio_super_resolution_core.py in the same directory.
Install  : pip install PyQt6 matplotlib soundfile scipy numpy

HiDPI notes
-----------
PyQt6 handles per-monitor DPI scaling natively on Windows, macOS, and Linux.
No manual DPI detection or geometry scaling is needed.  The matplotlib
FigureCanvasQTAgg backend renders through Qt's device-pixel-ratio mechanism,
so spectrograms are always sharp and correctly sized at any scale factor.
"""

import sys
import threading
import multiprocessing as mp
from pathlib import Path

import numpy as np
from scipy import signal
import soundfile as sf

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QScrollArea,
    QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QPushButton, QCheckBox, QComboBox,
    QSpinBox, QDoubleSpinBox, QProgressBar, QFileDialog,
    QMessageBox, QListWidget, QGroupBox, QFrame, QTabWidget,
    QRadioButton, QSizePolicy,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont

# ── matplotlib (optional, for spectrogram) ───────────────────────────────
_MATPLOTLIB_AVAILABLE = False
_MATPLOTLIB_ERROR = ""
try:
    import matplotlib
    matplotlib.use('QtAgg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    _MATPLOTLIB_AVAILABLE = True
except Exception as _mpl_err:
    _MATPLOTLIB_ERROR = str(_mpl_err)

# ── DSP back-end ─────────────────────────────────────────────────────────
from audio_super_resolution_core import (
    upsample_441_to_48k,
    _run_channel_pipeline,
    true_peak_limiter,
)


# ============================================================================
# Spectrogram computation  (GUI layer; identical logic to the original)
# ============================================================================

def _compute_spectrogram(data: np.ndarray, sr: int,
                          target_max_khz: float = 96.0,
                          n_time_cols: int = 1000):
    mono = (data.mean(axis=1) if data.ndim == 2 else data.ravel()).astype(np.float32)
    max_samples = int(120 * sr)
    if len(mono) > max_samples:
        mono = mono[:max_samples]
    n_fft = 4096 if sr >= 96000 else 2048
    hop   = max(n_fft // 8, len(mono) // n_time_cols)
    freqs, times, Zxx = signal.stft(mono, fs=sr,
                                     nperseg=n_fft, noverlap=n_fft - hop,
                                     window='hann')
    mag_db = 20.0 * np.log10(np.abs(Zxx) + 1e-9)
    f_khz  = freqs / 1000.0
    mask   = f_khz <= target_max_khz
    f_khz, mag_db = f_khz[mask], mag_db[mask, :]
    if mag_db.shape[1] > n_time_cols:
        idx    = np.linspace(0, mag_db.shape[1] - 1, n_time_cols, dtype=int)
        mag_db = mag_db[:, idx]
        times  = times[np.minimum(idx, len(times) - 1)]
    return mag_db, times, f_khz


# ============================================================================
# Worker thread  (QThread + Qt signals for thread-safe UI updates)
# ============================================================================

class WorkerThread(QThread):
    sig_step        = pyqtSignal(int, str)   # (pct 0-100, message)
    sig_batch       = pyqtSignal(int, int)   # (done, total)
    sig_done        = pyqtSignal(int, list)  # (n_processed, error_list)
    sig_spec_before = pyqtSignal(object)     # (mag_db, times, f_khz)
    sig_spec_after  = pyqtSignal(object)

    def __init__(self, files, out_dir, params, target_sr):
        super().__init__()
        self.files     = files
        self.out_dir   = out_dir
        self.params    = params
        self.target_sr = target_sr
        self._cancel   = False

    def cancel(self):
        self._cancel = True

    # ── main thread body ─────────────────────────────────────────────────
    def run(self):
        errors   = []
        n        = len(self.files)
        out_path = Path(self.out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        for idx, src in enumerate(self.files):
            if self._cancel:
                break
            self.sig_batch.emit(idx, n)
            out_file = out_path / (src.stem + "_192k.wav")
            try:
                self._process_one(src, out_file, idx, n)
            except Exception:
                import traceback
                errors.append(f"{src.name}:\n{traceback.format_exc()}")

        self.sig_batch.emit(n, n)
        self.sig_done.emit(n, errors)

    # ── progress helper ───────────────────────────────────────────────────
    def _prog(self, s, t, msg):
        if self._cancel:
            raise InterruptedError("User cancelled")
        self.sig_step.emit(int(s / max(t, 1) * 100), msg)

    # ── per-file pipeline (mirrors original _process_one exactly) ─────────
    def _process_one(self, src_path: Path, out_file: Path,
                     batch_idx: int, batch_total: int):
        def _p(s, t, m):
            self._prog(s, t, f"[{batch_idx+1}/{batch_total}] {src_path.name} — {m}")

        _p(0, 100, "Reading audio...")
        orig_data, orig_sr = sf.read(str(src_path))
        orig_data = orig_data.astype(np.float32)
        target_sr = self.target_sr
        p         = self.params

        _is_single   = (batch_total == 1)
        _do_spectrum = _is_single and _MATPLOTLIB_AVAILABLE

        if _do_spectrum:
            _p(1, 100, "Computing input spectrum…")
            self.sig_spec_before.emit(_compute_spectrogram(orig_data, orig_sr))

        # ── adaptive SR parameters ────────────────────────────────────
        if p['auto_adapt_sr']:
            roll           = (orig_sr / 2.0) - 2200
            calc_crossover = int(roll)
            calc_eq_low    = int(roll - 1000)
            calc_eq_high   = int(orig_sr - calc_eq_low)
            calc_att_low   = int(calc_eq_low - 1000)
            calc_att_high  = int(calc_eq_low)
        else:
            calc_crossover = p['splice_crossover']
            calc_eq_low    = 19000
            calc_eq_high   = 23000
            calc_att_low   = p['att_ref_low']
            calc_att_high  = p['att_ref_high']

        # ── 44.1 kHz pre-processing ───────────────────────────────────
        if orig_sr == 44100:
            _p(3, 100, 'Detected 44.1 kHz — pre-upsampling to 48 kHz...')
            def _p0(s, t, m):
                self._prog(3 + int(s / max(t, 1) * 6), 100, f'[44.1→48k] {m}')
            orig_data = upsample_441_to_48k(orig_data, progress_callback=_p0)
            orig_sr   = 48000
            if p['auto_adapt_sr']:
                roll           = (orig_sr / 2.0) - 2200
                calc_crossover = int(roll)
                calc_eq_low    = int(roll - 1000)
                calc_eq_high   = int(orig_sr - calc_eq_low)
                calc_att_low   = int(calc_eq_low - 1000)
                calc_att_high  = int(calc_eq_low)

        is_stereo = orig_data.ndim == 2 and orig_data.shape[1] >= 2
        if is_stereo:
            ch_smooth = max(1, p['num_threads'] // 2)
            ch_chunk  = max(1, p['num_chunk_workers'] // 2)
        else:
            ch_smooth = p['num_threads']
            ch_chunk  = p['num_chunk_workers']

        run_p = dict(
            use_chunk           = p['use_chunk'],
            max_chunk_sec       = p['max_chunk_sec'],
            chunk_workers       = ch_chunk,
            smooth_workers      = ch_smooth,
            enable_octave_copy  = p['enable_octave_copy'],
            split_freq          = p['split_freq'],
            fft_size            = p['fft_size'],
            cutoff_freq         = p['cutoff_freq'],
            smooth_mode         = p['smooth_mode'],
            enable_splice       = p['enable_splice'],
            calc_crossover      = calc_crossover,
            splice_fade_hz      = p['splice_fade_hz'],
            enable_attenuate    = p['enable_attenuate'],
            calc_att_low        = calc_att_low,
            calc_att_high       = calc_att_high,
            att_smooth_ms       = p['att_smooth_ms'],
            att_gain_floor_db   = p['att_gain_floor_db'],
            enable_smart_eq     = p['enable_smart_eq'],
            enable_auto_flatten = p['enable_auto_flatten'],
            calc_eq_low         = calc_eq_low,
            calc_eq_high        = calc_eq_high,
            smart_eq_boost      = p['smart_eq_boost'],
            flat_start_freq     = p['flat_start_freq'],
            flat_max_boost      = p['flat_max_boost'],
            flat_max_cut        = p['flat_max_cut'],
            enable_final_shelf  = p['enable_final_shelf'],
            final_shelf_freq    = p['final_shelf_freq'],
            final_shelf_gain    = p['final_shelf_gain'],
            final_shelf_curve   = p['final_shelf_curve'],
        )

        if is_stereo:
            ch_results = [None, None]
            ch_errors  = [None, None]

            def _run_ch(ci):
                try:
                    mono = orig_data[:, ci]
                    cb   = (lambda s, t, m: _p(s, t, m)) if ci == 0 else None
                    ch_results[ci] = _run_channel_pipeline(
                        mono, mono, orig_sr, target_sr, run_p, cb)
                except Exception:
                    import traceback
                    ch_errors[ci] = traceback.format_exc()

            _p(9, 100, "Processing L+R channels in parallel...")
            tl = threading.Thread(target=_run_ch, args=(0,), daemon=True)
            tr = threading.Thread(target=_run_ch, args=(1,), daemon=True)
            tl.start(); tr.start(); tl.join(); tr.join()
            if ch_errors[0] or ch_errors[1]:
                raise RuntimeError(f"L: {ch_errors[0]}\nR: {ch_errors[1]}")
            result = np.column_stack([ch_results[0], ch_results[1]])
        else:
            mono   = orig_data.ravel()
            result = _run_channel_pipeline(
                mono, mono, orig_sr, target_sr, run_p,
                lambda s, t, m: _p(s, t, m))

        if p['enable_limiter']:
            _p(97, 100, f"Limiter ({p['limiter_ceiling_db']:+.2f} dBFS)...")
            result = true_peak_limiter(result, target_sr,
                                       ceiling_db=p['limiter_ceiling_db'],
                                       lookahead_ms=p['limiter_lookahead'],
                                       release_ms=p['limiter_release'])

        _p(99, 100, "Saving...")
        sf.write(str(out_file), result, target_sr, subtype='FLOAT')
        _p(100, 100, f"✓ Done → {out_file.name}")

        if _do_spectrum:
            self.sig_spec_after.emit(
                _compute_spectrogram(result.astype(np.float32), target_sr))


# ============================================================================
# Spectrogram widget
# ============================================================================

class SpectrogramWidget(QFrame):
    """
    Self-contained matplotlib spectrogram panel.
    FigureCanvasQTAgg handles HiDPI via Qt device-pixel-ratio automatically —
    no manual DPI scaling or figsize arithmetic required.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self._spec_before = None
        self._spec_after  = None
        self._view        = 'before'
        self._canvas      = None
        self._fig         = None
        self._ax          = None
        self._cmap        = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 8)
        layout.setSpacing(4)

        # title
        ttl = QLabel("Spectrogram Preview  (0 – 96 kHz)")
        ttl.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(ttl)

        # control row
        ctrl = QHBoxLayout()
        self._rb_before = QRadioButton("Before (original)")
        self._rb_after  = QRadioButton("After (super-res)")
        self._rb_before.setChecked(True)
        self._rb_before.toggled.connect(self._on_toggle)
        ctrl.addWidget(self._rb_before)
        ctrl.addWidget(self._rb_after)
        ctrl.addStretch()
        self._lbl_status = QLabel("Awaiting processing…")
        self._lbl_status.setStyleSheet("color: #888; font-size: 11px;")
        ctrl.addWidget(self._lbl_status)
        layout.addLayout(ctrl)

        # canvas or error label
        if _MATPLOTLIB_AVAILABLE:
            self._build_canvas(layout)
        else:
            err = QLabel(
                f"Spectrogram preview requires matplotlib.\n"
                f"Install: pip install matplotlib\n\n{_MATPLOTLIB_ERROR}")
            err.setAlignment(Qt.AlignmentFlag.AlignCenter)
            err.setStyleSheet("color: #888; font-size: 11px; padding: 20px;")
            layout.addWidget(err)

    def _build_canvas(self, layout):
        from matplotlib.colors import LinearSegmentedColormap
        self._cmap = LinearSegmentedColormap.from_list(
            'sv', ['#000000', '#001800', '#004400',
                   '#00aa00', '#88dd00', '#ffff00', '#ffffff'])

        # figsize in logical inches; Qt renders at the correct physical
        # resolution via its internal device-pixel-ratio scaling.
        self._fig, self._ax = plt.subplots(
            figsize=(8, 3.6), dpi=96, layout='constrained')
        self._fig.patch.set_facecolor('#1e1e1e')
        self._ax.set_facecolor('#000000')
        self._apply_empty_style()

        self._canvas = FigureCanvas(self._fig)
        self._canvas.setMinimumHeight(240)
        self._canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding)
        layout.addWidget(self._canvas)

    def _apply_empty_style(self):
        ax = self._ax
        ax.set_title("Spectrogram preview — awaiting processing",
                     fontsize=9, color='#888')
        ax.set_xlabel("Time (s)",        fontsize=9, color='#aaa')
        ax.set_ylabel("Frequency (kHz)", fontsize=9, color='#aaa')
        ax.tick_params(labelsize=8, colors='#aaa')
        for sp in ax.spines.values():
            sp.set_edgecolor('#444')

    def _on_toggle(self):
        self._view = 'before' if self._rb_before.isChecked() else 'after'
        self.redraw()

    # ── public API ────────────────────────────────────────────────────────
    def reset(self):
        self._spec_before = self._spec_after = None
        self._lbl_status.setText("Awaiting processing…")
        self._rb_before.setChecked(True)
        self.redraw()

    def set_before(self, data):
        self._spec_before = data
        self._lbl_status.setText("Before ready — processing…")
        self._rb_before.setChecked(True)
        self.redraw()

    def set_after(self, data):
        self._spec_after = data
        self._lbl_status.setText("Before / After ready  ✓")
        self._rb_after.setChecked(True)
        self.redraw()

    def redraw(self):
        if not _MATPLOTLIB_AVAILABLE or self._ax is None:
            return
        ax = self._ax
        ax.cla()
        ax.set_facecolor('#000000')

        data = self._spec_before if self._view == 'before' else self._spec_after
        if data is None:
            self._apply_empty_style()
        else:
            mag_db, t_axis, f_khz = data
            DB_MIN, DB_MAX = -110, 0
            mag_db = np.clip(mag_db, DB_MIN, DB_MAX)
            extent = [float(t_axis[0]), float(t_axis[-1]),
                      float(f_khz[0]),  float(f_khz[-1])]
            ax.imshow(mag_db, aspect='auto', origin='lower',
                      extent=extent, vmin=DB_MIN, vmax=DB_MAX,
                      cmap=self._cmap, interpolation='nearest')
            ax.set_ylim(0, 96)
            ax.yaxis.set_major_locator(ticker.MultipleLocator(16))
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(8))
            for khz, col, alp in [(22.05, '#ff8a65', 0.7),
                                   (24,    '#ff8a65', 0.4),
                                   (48,    '#ce93d8', 0.4)]:
                if khz <= f_khz[-1]:
                    ax.axhline(khz, color=col, linewidth=0.6,
                               linestyle='--', alpha=alp)
            t_col = '#4fc3f7' if self._view == 'before' else '#a5d6a7'
            t_txt = ("Before — original audio spectrogram"
                     if self._view == 'before' else
                     "After — super-resolved audio spectrogram")
            ax.set_title(t_txt,          fontsize=9, color=t_col)
            ax.set_xlabel("Time (s)",        fontsize=9, color='#aaa')
            ax.set_ylabel("Frequency (kHz)", fontsize=9, color='#aaa')
            ax.tick_params(labelsize=8, colors='#aaa')
            for sp in ax.spines.values():
                sp.set_edgecolor('#444')

        self._canvas.draw()


# ============================================================================
# Main window
# ============================================================================

class AudioProcessorGUI(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Super-Resolution v2.3")
        self.setMinimumSize(600, 440)
        self._worker       = None
        self._adv_expanded = False
        self._spec_visible = False
        self._setup_ui()
        # Defer initial sizing until after the window is shown so that
        # self.screen() returns a valid QScreen object.
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(0, self._initial_resize)

    def _initial_resize(self):
        """Set the initial window size, capped to the available screen area."""
        screen = self.screen()
        avail_h = screen.availableGeometry().height() if screen else 900
        h = min(520, avail_h - 40)
        self.resize(760, h)

    # ── top-level layout ──────────────────────────────────────────────────
    def _setup_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        main = QVBoxLayout(root)
        main.setContentsMargins(0, 0, 0, 0)
        main.setSpacing(0)

        # header
        hdr = QWidget()
        hdr.setContentsMargins(16, 8, 16, 4)
        hdr_row = QHBoxLayout(hdr)
        hdr_row.setContentsMargins(0, 0, 0, 0)
        lbl_icon = QLabel("🎵")
        lbl_icon.setFont(QFont("Arial", 18))
        lbl_title = QLabel("Audio Super-Resolution")
        lbl_title.setFont(QFont("Arial", 15, QFont.Weight.Bold))
        lbl_ver = QLabel("v2.3")
        lbl_ver.setFont(QFont("Arial", 9))
        lbl_ver.setStyleSheet("color: #888;")
        hdr_row.addWidget(lbl_icon)
        hdr_row.addWidget(lbl_title)
        hdr_row.addWidget(lbl_ver)
        hdr_row.addStretch()
        main.addWidget(hdr)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #ccc;")
        main.addWidget(sep)

        # scrollable content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget()
        self._inner = QVBoxLayout(inner)
        self._inner.setContentsMargins(12, 8, 12, 8)
        self._inner.setSpacing(6)
        scroll.setWidget(inner)
        main.addWidget(scroll, stretch=1)

        self._build_file_group()
        self._build_files_list()
        self._build_option_row()
        self._build_advanced()
        self._build_spectrogram_section()
        self._inner.addStretch()

        # bottom bar (outside scroll — always visible)
        bot_sep = QFrame()
        bot_sep.setFrameShape(QFrame.Shape.HLine)
        bot_sep.setStyleSheet("color: #ccc;")
        main.addWidget(bot_sep)
        self._build_bottom(main)

    # ── File / Folder group ───────────────────────────────────────────────
    def _build_file_group(self):
        grp = QGroupBox("File / Folder")
        gl  = QFormLayout(grp)
        gl.setHorizontalSpacing(8)
        gl.setVerticalSpacing(6)

        self._edit_input = QLineEdit()
        self._edit_input.textChanged.connect(self._refresh_file_list)
        row_in = QHBoxLayout()
        row_in.addWidget(self._edit_input)
        btn_file = QPushButton("File…");     btn_file.setFixedWidth(68)
        btn_dir  = QPushButton("Folder…");   btn_dir.setFixedWidth(68)
        btn_file.clicked.connect(self._browse_file)
        btn_dir.clicked.connect(self._browse_input_folder)
        row_in.addWidget(btn_file); row_in.addWidget(btn_dir)
        gl.addRow("Input:", row_in)

        self._edit_output = QLineEdit()
        row_out = QHBoxLayout()
        row_out.addWidget(self._edit_output)
        btn_out = QPushButton("Folder…"); btn_out.setFixedWidth(68)
        btn_out.clicked.connect(self._browse_output_folder)
        row_out.addWidget(btn_out)
        gl.addRow("Output Folder:", row_out)

        hint = QLabel("Supports WAV / FLAC.  Select a single file or an entire folder (batch)."
                      "  Output is always WAV 32-bit float.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #666; font-size: 11px;")
        gl.addRow("", hint)

        self._inner.addWidget(grp)

    # ── Files to Process list ─────────────────────────────────────────────
    def _build_files_list(self):
        grp = QGroupBox("Files to Process")
        gl  = QVBoxLayout(grp)
        gl.setContentsMargins(8, 6, 8, 6)

        self._list_files = QListWidget()
        self._list_files.setFixedHeight(72)
        self._list_files.setFont(QFont("Consolas", 9))
        self._list_files.setStyleSheet(
            "background: #f7f7f7; border: 1px solid #ddd;")
        gl.addWidget(self._list_files)

        self._lbl_count = QLabel("No file selected")
        self._lbl_count.setStyleSheet("color: #888; font-size: 11px;")
        gl.addWidget(self._lbl_count)

        self._inner.addWidget(grp)

    # ── Advanced toggle + Spectrum checkbox row ───────────────────────────
    def _build_option_row(self):
        row = QHBoxLayout()

        self._btn_adv = QPushButton("▶  Advanced Options (Expand)")
        self._btn_adv.clicked.connect(self._toggle_advanced)
        row.addWidget(self._btn_adv)

        vsep = QFrame()
        vsep.setFrameShape(QFrame.Shape.VLine)
        vsep.setStyleSheet("color: #ccc;")
        row.addWidget(vsep)

        self._chk_spectrum = QCheckBox("☰  Preview Spectrogram (single file only)")
        if not _MATPLOTLIB_AVAILABLE:
            self._chk_spectrum.setEnabled(False)
            self._chk_spectrum.setToolTip(
                f"matplotlib not available: {_MATPLOTLIB_ERROR or 'pip install matplotlib'}")
        self._chk_spectrum.stateChanged.connect(self._toggle_spectrum)
        row.addWidget(self._chk_spectrum)
        row.addStretch()

        self._inner.addLayout(row)

    # ── Advanced options (collapsible) ────────────────────────────────────
    def _build_advanced(self):
        self._adv_frame = QGroupBox("Advanced Options")
        self._adv_frame.setVisible(False)

        tabs = QTabWidget()
        tabs.addTab(self._tab_spectrum_repair(), "Spectrum Repair")
        tabs.addTab(self._tab_sample_threads(),  "Sample & Threads")
        tabs.addTab(self._tab_chunking(),        "Octave Copy & Chunking")

        lay = QVBoxLayout(self._adv_frame)
        lay.addWidget(tabs)
        self._inner.addWidget(self._adv_frame)

    # ── Tab 1: Spectrum Repair ────────────────────────────────────────────
    def _tab_spectrum_repair(self):
        w = QWidget()
        f = QVBoxLayout(w)
        f.setContentsMargins(10, 8, 10, 8)
        f.setSpacing(4)
        f.setAlignment(Qt.AlignmentFlag.AlignTop)

        def _sec(text, color='#222'):
            lbl = QLabel(f"▎ {text}")
            lbl.setStyleSheet(
                f"font-weight: bold; font-size: 12px; color: {color}; padding-top: 6px;")
            return lbl

        def _hsep():
            s = QFrame(); s.setFrameShape(QFrame.Shape.HLine)
            s.setStyleSheet("color: #ddd; margin: 4px 0;")
            return s

        # ── Smart Spectral Iron ───────────────────────────────────────
        f.addWidget(_sec("Smart Spectral Iron (auto-flatten narrow bands)"))
        self._chk_auto_flatten = QCheckBox("Enable global HF envelope smart matching")
        self._chk_auto_flatten.setChecked(True); f.addWidget(self._chk_auto_flatten)
        r1 = QHBoxLayout(); r1.setContentsMargins(18, 0, 0, 0)
        r1.addWidget(QLabel("Start Freq:"))
        self._spin_flat_start = QSpinBox()
        self._spin_flat_start.setRange(1000, 48000); self._spin_flat_start.setValue(15000)
        self._spin_flat_start.setSuffix(" Hz"); r1.addWidget(self._spin_flat_start)
        r1.addWidget(QLabel("Max Boost:"))
        self._spin_flat_boost = QDoubleSpinBox()
        self._spin_flat_boost.setRange(0, 30); self._spin_flat_boost.setValue(6.0)
        self._spin_flat_boost.setSuffix(" dB"); r1.addWidget(self._spin_flat_boost)
        r1.addWidget(QLabel("Max Cut:"))
        self._spin_flat_cut = QDoubleSpinBox()
        self._spin_flat_cut.setRange(0, 30); self._spin_flat_cut.setValue(12.0)
        self._spin_flat_cut.setSuffix(" dB"); r1.addWidget(self._spin_flat_cut)
        r1.addStretch(); f.addLayout(r1)

        f.addWidget(_hsep())

        # ── Nyquist V-Notch Repair ────────────────────────────────────
        f.addWidget(_sec("Nyquist V-Notch Repair"))
        self._chk_auto_adapt = QCheckBox("Auto-detect input SR and repair transition band")
        self._chk_auto_adapt.setChecked(True); f.addWidget(self._chk_auto_adapt)
        self._chk_smart_eq = QCheckBox("Run V-notch STFT dedicated repair")
        self._chk_smart_eq.setChecked(True); f.addWidget(self._chk_smart_eq)
        r2 = QHBoxLayout(); r2.setContentsMargins(18, 0, 0, 0)
        r2.addWidget(QLabel("Max Boost:"))
        self._spin_eq_boost = QDoubleSpinBox()
        self._spin_eq_boost.setRange(0, 30); self._spin_eq_boost.setValue(15.0)
        self._spin_eq_boost.setSuffix(" dB"); r2.addWidget(self._spin_eq_boost)
        r2.addStretch(); f.addLayout(r2)

        f.addWidget(_hsep())

        # ── Faithful Splice ───────────────────────────────────────────
        f.addWidget(_sec("Faithful Splice & Dynamic Attenuation"))
        self._chk_splice = QCheckBox("Enable lossless faithful splice")
        self._chk_splice.setChecked(True); f.addWidget(self._chk_splice)
        self._chk_attenuate = QCheckBox("Enable envelope-follower dynamic HF modulation")
        self._chk_attenuate.setChecked(True); f.addWidget(self._chk_attenuate)
        r3 = QHBoxLayout(); r3.setContentsMargins(18, 0, 0, 0)
        r3.addWidget(QLabel("Splice Fade:"))
        self._spin_splice_fade = QSpinBox()
        self._spin_splice_fade.setRange(50, 5000); self._spin_splice_fade.setValue(500)
        self._spin_splice_fade.setSuffix(" Hz"); r3.addWidget(self._spin_splice_fade)
        r3.addWidget(QLabel("Att. Smooth:"))
        self._spin_att_smooth = QDoubleSpinBox()
        self._spin_att_smooth.setRange(0.5, 100); self._spin_att_smooth.setValue(10.0)
        self._spin_att_smooth.setSuffix(" ms"); r3.addWidget(self._spin_att_smooth)
        r3.addWidget(QLabel("Gain Floor:"))
        self._spin_att_floor = QDoubleSpinBox()
        self._spin_att_floor.setRange(-40, 0); self._spin_att_floor.setValue(-20.0)
        self._spin_att_floor.setSuffix(" dB"); r3.addWidget(self._spin_att_floor)
        r3.addStretch(); f.addLayout(r3)

        f.addWidget(_hsep())

        # ── Ultra-HF Power-Law Rolloff ────────────────────────────────
        f.addWidget(_sec("Ultra-HF Power-Law Rolloff", '#2E7D32'))
        self._chk_final_shelf = QCheckBox("Enable natural arc rolloff")
        self._chk_final_shelf.setChecked(True); f.addWidget(self._chk_final_shelf)
        r4 = QHBoxLayout(); r4.setContentsMargins(18, 0, 0, 0)
        r4.addWidget(QLabel("Start Freq:"))
        self._spin_shelf_freq = QSpinBox()
        self._spin_shelf_freq.setRange(1000, 96000); self._spin_shelf_freq.setValue(20000)
        self._spin_shelf_freq.setSuffix(" Hz"); r4.addWidget(self._spin_shelf_freq)
        r4.addWidget(QLabel("Total Atten:"))
        self._spin_shelf_gain = QDoubleSpinBox()
        self._spin_shelf_gain.setRange(-60, 0); self._spin_shelf_gain.setValue(-18.0)
        self._spin_shelf_gain.setSuffix(" dB"); r4.addWidget(self._spin_shelf_gain)
        r4.addWidget(QLabel("Curve Exp:"))
        self._spin_shelf_curve = QDoubleSpinBox()
        self._spin_shelf_curve.setRange(0.05, 3.0); self._spin_shelf_curve.setValue(0.65)
        self._spin_shelf_curve.setSingleStep(0.05); r4.addWidget(self._spin_shelf_curve)
        r4.addStretch(); f.addLayout(r4)

        f.addWidget(_hsep())

        # ── True-Peak Limiter ─────────────────────────────────────────
        f.addWidget(_sec("Look-ahead Transparent True-Peak Limiter", '#6A1B9A'))
        self._chk_limiter = QCheckBox(
            "Enable transparent limiter (push peaks to ceiling with minimal distortion)")
        self._chk_limiter.setChecked(True); f.addWidget(self._chk_limiter)
        r5 = QHBoxLayout(); r5.setContentsMargins(18, 0, 0, 0)
        r5.addWidget(QLabel("Peak Ceiling:"))
        self._spin_lim_ceil = QDoubleSpinBox()
        self._spin_lim_ceil.setRange(-6, 0); self._spin_lim_ceil.setValue(-0.1)
        self._spin_lim_ceil.setSingleStep(0.05); self._spin_lim_ceil.setSuffix(" dBFS")
        r5.addWidget(self._spin_lim_ceil)
        r5.addWidget(QLabel("Look-ahead:"))
        self._spin_lim_la = QDoubleSpinBox()
        self._spin_lim_la.setRange(0.5, 20); self._spin_lim_la.setValue(2.0)
        self._spin_lim_la.setSuffix(" ms"); r5.addWidget(self._spin_lim_la)
        r5.addWidget(QLabel("Release:"))
        self._spin_lim_rel = QDoubleSpinBox()
        self._spin_lim_rel.setRange(10, 2000); self._spin_lim_rel.setValue(150.0)
        self._spin_lim_rel.setSuffix(" ms"); r5.addWidget(self._spin_lim_rel)
        r5.addStretch(); f.addLayout(r5)

        return w

    # ── Tab 2: Sample & Threads ───────────────────────────────────────────
    def _tab_sample_threads(self):
        w = QWidget()
        f = QFormLayout(w)
        f.setContentsMargins(10, 8, 10, 8)
        f.setHorizontalSpacing(10)
        f.setVerticalSpacing(8)

        self._combo_sr = QComboBox()
        self._combo_sr.addItems(["48000", "96000", "192000"])
        self._combo_sr.setCurrentText("192000")
        f.addRow("Target Sample Rate:", self._combo_sr)

        self._combo_fft = QComboBox()
        self._combo_fft.addItems(["256", "512", "1024", "2048"])
        self._combo_fft.setCurrentText("512")
        f.addRow("FFT Size:", self._combo_fft)

        self._spin_cutoff = QSpinBox()
        self._spin_cutoff.setRange(1000, 96000)
        self._spin_cutoff.setValue(22000)
        self._spin_cutoff.setSuffix(" Hz")
        f.addRow("Smooth Start:", self._spin_cutoff)

        self._combo_smooth = QComboBox()
        self._combo_smooth.addItems(["perfect_flat", "gentle_slope"])
        f.addRow("Smooth Mode:", self._combo_smooth)

        tr = QHBoxLayout()
        self._chk_multithread = QCheckBox("Multithreaded")
        self._chk_multithread.setChecked(True)
        self._spin_threads = QSpinBox()
        self._spin_threads.setRange(1, mp.cpu_count())
        self._spin_threads.setValue(max(1, mp.cpu_count() - 1))
        tr.addWidget(self._chk_multithread)
        tr.addWidget(self._spin_threads)
        tr.addWidget(QLabel("threads"))
        tr.addStretch()
        f.addRow("Threads:", tr)

        return w

    # ── Tab 3: Octave Copy & Chunking ─────────────────────────────────────
    def _tab_chunking(self):
        w = QWidget()
        f = QVBoxLayout(w)
        f.setContentsMargins(10, 8, 10, 8)
        f.setSpacing(4)
        f.setAlignment(Qt.AlignmentFlag.AlignTop)

        def _sec(text):
            lbl = QLabel(f"▎ {text}")
            lbl.setStyleSheet(
                "font-weight: bold; font-size: 12px; padding-top: 6px;")
            return lbl

        def _hsep():
            s = QFrame(); s.setFrameShape(QFrame.Shape.HLine)
            s.setStyleSheet("color: #ddd; margin: 4px 0;")
            return s

        f.addWidget(_sec("Octave Copy Smoothing"))
        self._chk_octave = QCheckBox("Enable octave copy smoothing")
        self._chk_octave.setChecked(True); f.addWidget(self._chk_octave)
        r1 = QHBoxLayout(); r1.setContentsMargins(18, 0, 0, 0)
        r1.addWidget(QLabel("Split Freq:"))
        self._spin_split = QSpinBox()
        self._spin_split.setRange(1000, 96000); self._spin_split.setValue(50000)
        self._spin_split.setSuffix(" Hz"); r1.addWidget(self._spin_split)
        r1.addStretch(); f.addLayout(r1)

        f.addWidget(_hsep())

        f.addWidget(_sec("Smart Chunked Parallel Processing"))
        self._chk_chunk = QCheckBox("Enable smart chunked parallel processing")
        self._chk_chunk.setChecked(True); f.addWidget(self._chk_chunk)
        r2 = QHBoxLayout(); r2.setContentsMargins(18, 0, 0, 0)
        r2.addWidget(QLabel("Chunk Length:"))
        self._spin_chunk_sec = QDoubleSpinBox()
        self._spin_chunk_sec.setRange(1, 60); self._spin_chunk_sec.setValue(5.0)
        self._spin_chunk_sec.setSuffix(" s"); r2.addWidget(self._spin_chunk_sec)
        r2.addWidget(QLabel("Parallel Chunks:"))
        self._spin_chunk_workers = QSpinBox()
        self._spin_chunk_workers.setRange(1, mp.cpu_count())
        self._spin_chunk_workers.setValue(max(1, min(4, mp.cpu_count() // 2)))
        r2.addWidget(self._spin_chunk_workers)
        r2.addStretch(); f.addLayout(r2)

        return w

    # ── Spectrogram panel ─────────────────────────────────────────────────
    def _build_spectrogram_section(self):
        self._spec_widget = SpectrogramWidget()
        self._spec_widget.setVisible(False)
        self._inner.addWidget(self._spec_widget)

    # ── Bottom bar ────────────────────────────────────────────────────────
    def _build_bottom(self, parent_layout):
        bot = QWidget()
        bl  = QVBoxLayout(bot)
        bl.setContentsMargins(14, 6, 14, 8)
        bl.setSpacing(4)

        def _bar_row(label_text):
            row = QHBoxLayout()
            lbl = QLabel(label_text)
            lbl.setFixedWidth(38)
            lbl.setStyleSheet("color: #555; font-size: 11px;")
            bar = QProgressBar()
            bar.setRange(0, 100); bar.setValue(0)
            bar.setTextVisible(False); bar.setFixedHeight(14)
            extra = QLabel()
            extra.setStyleSheet("color: #555; font-size: 11px;")
            extra.setFixedWidth(80)
            row.addWidget(lbl); row.addWidget(bar); row.addWidget(extra)
            return row, bar, extra

        batch_row, self._bar_batch, self._lbl_batch = _bar_row("Total:")
        bl.addLayout(batch_row)
        self._lbl_batch.setText("Ready")

        step_row, self._bar_step, _ = _bar_row("Step:")
        bl.addLayout(step_row)

        self._lbl_status = QLabel("")
        self._lbl_status.setStyleSheet("color: #444; font-size: 11px;")
        bl.addWidget(self._lbl_status)

        btn_row = QHBoxLayout()
        self._btn_start  = QPushButton("▶   Start Processing")
        self._btn_cancel = QPushButton("⏹  Cancel")
        self._btn_cancel.setEnabled(False)
        self._btn_start.setFixedHeight(30)
        self._btn_cancel.setFixedHeight(30)
        self._btn_start.clicked.connect(self._on_start)
        self._btn_cancel.clicked.connect(self._on_cancel)
        btn_row.addWidget(self._btn_start)
        btn_row.addWidget(self._btn_cancel)
        btn_row.addStretch()
        bl.addLayout(btn_row)

        parent_layout.addWidget(bot)

    # ── Toggle helpers ────────────────────────────────────────────────────
    def _toggle_advanced(self):
        self._adv_expanded = not self._adv_expanded
        self._adv_frame.setVisible(self._adv_expanded)
        self._btn_adv.setText(
            "▼  Advanced Options (Collapse)" if self._adv_expanded
            else "▶  Advanced Options (Expand)")
        self._resize_window()

    def _toggle_spectrum(self, state):
        self._spec_visible = bool(state)
        self._spec_widget.setVisible(self._spec_visible)
        self._resize_window()

    def _resize_window(self):
        # Desired height based on visible panels.
        # QScrollArea handles any overflow — we never need to exceed the
        # screen's available height.  Requesting a size larger than the
        # available geometry causes Qt to emit QWindowsWindow::setGeometry
        # warnings and silently clamp the window, so we cap it here first.
        h = 520
        if self._adv_expanded: h += 260
        if self._spec_visible:  h += 320

        # Clamp to available screen height (minus a small margin for taskbar).
        screen = self.screen()
        if screen is not None:
            avail_h = screen.availableGeometry().height()
            h = min(h, avail_h - 40)   # 40 px margin keeps titlebar reachable

        self.resize(self.width(), h)

    # ── File browsing ─────────────────────────────────────────────────────
    def _browse_file(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Select Audio File", "",
            "Audio Files (*.wav *.flac);;All Files (*.*)")
        if f:
            self._edit_input.setText(f)
            if not self._edit_output.text():
                self._edit_output.setText(str(Path(f).parent))

    def _browse_input_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if d:
            self._edit_input.setText(d)
            if not self._edit_output.text():
                self._edit_output.setText(d)

    def _browse_output_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if d:
            self._edit_output.setText(d)

    # ── File list refresh ─────────────────────────────────────────────────
    def _collect_files(self):
        p = self._edit_input.text().strip()
        if not p:
            return []
        path = Path(p)
        if path.is_file():
            return [path] if path.suffix.lower() in ('.wav', '.flac') else []
        if path.is_dir():
            return sorted(f for f in path.iterdir()
                          if f.suffix.lower() in ('.wav', '.flac'))
        return []

    def _refresh_file_list(self):
        files = self._collect_files()
        self._list_files.clear()
        for f in files:
            self._list_files.addItem(f.name)
        if files:
            self._lbl_count.setText(
                f"{len(files)} file(s) found  (WAV / FLAC supported)")
            self._lbl_count.setStyleSheet("color: #2E7D32; font-size: 11px;")
        else:
            self._lbl_count.setText("No file selected")
            self._lbl_count.setStyleSheet("color: #888; font-size: 11px;")

    # ── Collect all parameters ────────────────────────────────────────────
    def _get_params(self):
        return dict(
            auto_adapt_sr       = self._chk_auto_adapt.isChecked(),
            enable_auto_flatten = self._chk_auto_flatten.isChecked(),
            flat_start_freq     = self._spin_flat_start.value(),
            flat_max_boost      = self._spin_flat_boost.value(),
            flat_max_cut        = self._spin_flat_cut.value(),
            enable_smart_eq     = self._chk_smart_eq.isChecked(),
            smart_eq_boost      = self._spin_eq_boost.value(),
            enable_splice       = self._chk_splice.isChecked(),
            splice_crossover    = 20000,
            splice_fade_hz      = self._spin_splice_fade.value(),
            att_ref_low         = 19000,
            att_ref_high        = 20000,
            enable_attenuate    = self._chk_attenuate.isChecked(),
            att_smooth_ms       = self._spin_att_smooth.value(),
            att_gain_floor_db   = self._spin_att_floor.value(),
            enable_final_shelf  = self._chk_final_shelf.isChecked(),
            final_shelf_freq    = self._spin_shelf_freq.value(),
            final_shelf_gain    = self._spin_shelf_gain.value(),
            final_shelf_curve   = self._spin_shelf_curve.value(),
            enable_limiter      = self._chk_limiter.isChecked(),
            limiter_ceiling_db  = self._spin_lim_ceil.value(),
            limiter_lookahead   = self._spin_lim_la.value(),
            limiter_release     = self._spin_lim_rel.value(),
            fft_size            = int(self._combo_fft.currentText()),
            cutoff_freq         = self._spin_cutoff.value(),
            smooth_mode         = self._combo_smooth.currentText(),
            num_threads         = self._spin_threads.value(),
            use_chunk           = self._chk_chunk.isChecked(),
            max_chunk_sec       = self._spin_chunk_sec.value(),
            num_chunk_workers   = self._spin_chunk_workers.value(),
            enable_octave_copy  = self._chk_octave.isChecked(),
            split_freq          = self._spin_split.value(),
        )

    # ── Processing ────────────────────────────────────────────────────────
    def _on_start(self):
        files = self._collect_files()
        if not files:
            QMessageBox.warning(self, "Notice",
                                "Please select an input file or folder first.")
            return
        out_dir = self._edit_output.text().strip()
        if not out_dir:
            QMessageBox.warning(self, "Notice",
                                "Please select an output folder first.")
            return

        if self._spec_visible and _MATPLOTLIB_AVAILABLE:
            self._spec_widget.reset()

        target_sr = int(self._combo_sr.currentText())
        self._worker = WorkerThread(files, out_dir, self._get_params(), target_sr)
        self._worker.sig_step.connect(self._on_step)
        self._worker.sig_batch.connect(self._on_batch)
        self._worker.sig_done.connect(self._on_done)
        self._worker.sig_spec_before.connect(self._spec_widget.set_before)
        self._worker.sig_spec_after.connect(self._spec_widget.set_after)
        self._worker.start()

        self._btn_start.setEnabled(False)
        self._btn_cancel.setEnabled(True)
        self._chk_spectrum.setEnabled(False)
        self._bar_batch.setValue(0)
        self._bar_step.setValue(0)

    def _on_cancel(self):
        if self._worker and QMessageBox.question(
                self, "Confirm", "Cancel processing?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        ) == QMessageBox.StandardButton.Yes:
            self._worker.cancel()
            self._lbl_status.setText("Cancelling, please wait…")

    def _on_step(self, pct: int, msg: str):
        self._bar_step.setValue(pct)
        self._lbl_status.setText(msg)

    def _on_batch(self, done: int, total: int):
        self._bar_batch.setValue(int(done / max(total, 1) * 100))
        self._lbl_batch.setText(f"{done}/{total} tracks")

    def _on_done(self, n: int, errors: list):
        self._btn_start.setEnabled(True)
        self._btn_cancel.setEnabled(False)
        self._chk_spectrum.setEnabled(_MATPLOTLIB_AVAILABLE)
        if errors:
            self._lbl_status.setText(f"Done ({len(errors)} file(s) with errors)")
            QMessageBox.critical(
                self, "Some Files Failed",
                "\n\n".join(errors[:3]) + ("\n\n…" if len(errors) > 3 else ""))
        else:
            self._lbl_status.setText(f"✓ All done! Processed {n} track(s)")
            QMessageBox.information(
                self, "Done",
                f"All {n} track(s) processed successfully!\n"
                f"Output: {self._edit_output.text()}")


# ============================================================================
# Entry point
# ============================================================================

if __name__ == '__main__':
    mp.freeze_support()

    # Qt handles per-monitor HiDPI natively on all platforms.
    # PassThrough lets Qt use fractional scale factors (e.g. 1.5×) exactly,
    # rather than rounding to 1× or 2×, giving correct sizing on 150% screens.
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)

    app = QApplication(sys.argv)
    app.setStyle('Fusion')   # consistent look on Windows / macOS / Linux

    win = AudioProcessorGUI()
    win.show()
    sys.exit(app.exec())
