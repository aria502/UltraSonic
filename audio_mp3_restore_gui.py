#!/usr/bin/env python3
"""
audio_mp3_restore_gui.py  —  PyQt6 front-end for the HF restoration tool.
Requires: audio_mp3_restore_core.py and audio_super_resolution_core.py
          in the same directory.
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
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
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

# ── DSP back-end ──────────────────────────────────────────────────────────
from audio_mp3_restore_core import (
    detect_mp3_cutoff,
    _run_restore_pipeline,
    true_peak_limiter,
    _CLI_DEFAULTS,
)


# ============================================================================
# Spectrogram computation
# ============================================================================

def _compute_spectrogram(data: np.ndarray, sr: int,
                          target_max_khz: float = 24.0,
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
# Auto-detect worker thread  (keep UI responsive during file analysis)
# ============================================================================

class DetectThread(QThread):
    sig_result = pyqtSignal(int)    # detected cutoff Hz
    sig_error  = pyqtSignal(str)

    def __init__(self, path: str):
        super().__init__()
        self._path = path

    def run(self):
        try:
            data, sr = sf.read(self._path)
            cutoff = detect_mp3_cutoff(data.astype(np.float32), sr)
            self.sig_result.emit(cutoff)
        except Exception as exc:
            self.sig_error.emit(str(exc))


# ============================================================================
# Processing worker thread
# ============================================================================

class WorkerThread(QThread):
    sig_step        = pyqtSignal(int, str)
    sig_batch       = pyqtSignal(int, int)
    sig_done        = pyqtSignal(int, list)
    sig_spec_before = pyqtSignal(object)
    sig_spec_after  = pyqtSignal(object)

    def __init__(self, files, out_dir, params, cutoff_freq):
        super().__init__()
        self.files       = files
        self.out_dir     = out_dir
        self.params      = params
        self.cutoff_freq = cutoff_freq   # None = auto-detect per file
        self._cancel     = False

    def cancel(self):
        self._cancel = True

    def run(self):
        errors   = []
        n        = len(self.files)
        out_path = Path(self.out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        for idx, src in enumerate(self.files):
            if self._cancel:
                break
            self.sig_batch.emit(idx, n)
            out_file = out_path / (src.stem + '_restored.wav')
            try:
                self._process_one(src, out_file, idx, n)
            except Exception:
                import traceback
                errors.append(f"{src.name}:\n{traceback.format_exc()}")

        self.sig_batch.emit(n, n)
        self.sig_done.emit(n, errors)

    def _prog(self, s, t, msg):
        if self._cancel:
            raise InterruptedError("User cancelled")
        self.sig_step.emit(int(s / max(t, 1) * 100), msg)

    def _process_one(self, src_path: Path, out_file: Path,
                     batch_idx: int, batch_total: int):
        def _p(s, t, m):
            self._prog(s, t, f"[{batch_idx+1}/{batch_total}] {src_path.name} — {m}")

        _p(0, 100, "Reading audio…")
        orig_data, orig_sr = sf.read(str(src_path))
        orig_data = orig_data.astype(np.float32)
        p         = self.params

        _is_single   = (batch_total == 1)
        _do_spectrum = _is_single and _MATPLOTLIB_AVAILABLE

        # Auto-detect cutoff if not locked
        if self.cutoff_freq is None:
            _p(1, 100, "Auto-detecting HF cutoff…")
            cutoff = detect_mp3_cutoff(orig_data, orig_sr)
            self.sig_step.emit(2, f"Detected cutoff: {cutoff} Hz")
        else:
            cutoff = int(np.clip(self.cutoff_freq,
                                  8000, int(orig_sr / 2) - 1000))

        if _do_spectrum:
            _p(2, 100, "Computing input spectrum…")
            nyq_khz = orig_sr / 2000.0
            self.sig_spec_before.emit(
                _compute_spectrogram(orig_data, orig_sr, target_max_khz=nyq_khz))

        is_stereo = orig_data.ndim == 2 and orig_data.shape[1] >= 2

        if is_stereo:
            ch_results = [None, None]
            ch_errors  = [None, None]

            def _run_ch(ci):
                try:
                    mono = orig_data[:, ci]
                    cb   = (lambda s, t, m: _p(s, t, m)) if ci == 0 else None
                    ch_results[ci] = _run_restore_pipeline(
                        mono, orig_sr, cutoff, p, cb)
                except Exception:
                    import traceback
                    ch_errors[ci] = traceback.format_exc()

            _p(5, 100, "Processing L+R channels in parallel…")
            tl = threading.Thread(target=_run_ch, args=(0,), daemon=True)
            tr = threading.Thread(target=_run_ch, args=(1,), daemon=True)
            tl.start(); tr.start(); tl.join(); tr.join()
            if ch_errors[0] or ch_errors[1]:
                raise RuntimeError(
                    f"Channel pipeline failed:\nL: {ch_errors[0]}\nR: {ch_errors[1]}")
            result = np.column_stack([ch_results[0], ch_results[1]])
        else:
            mono   = orig_data.ravel()
            result = _run_restore_pipeline(
                mono, orig_sr, cutoff, p,
                lambda s, t, m: _p(s, t, m))

        if p.get('enable_limiter', True):
            _p(97, 100, f"Limiter ({p['limiter_ceiling_db']:+.2f} dBFS)…")
            result = true_peak_limiter(result, orig_sr,
                                       ceiling_db   = p['limiter_ceiling_db'],
                                       lookahead_ms = p['limiter_lookahead'],
                                       release_ms   = p['limiter_release'])

        _p(99, 100, "Saving…")
        sf.write(str(out_file), result, orig_sr, subtype='FLOAT')
        _p(100, 100, f"✓ Done → {out_file.name}")

        if _do_spectrum:
            nyq_khz = orig_sr / 2000.0
            self.sig_spec_after.emit(
                _compute_spectrogram(result.astype(np.float32), orig_sr,
                                      target_max_khz=nyq_khz))


# ============================================================================
# Spectrogram widget  (identical to audio_super_resolution_gui.py)
# ============================================================================

class SpectrogramWidget(QFrame):
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

        ttl = QLabel("Spectrogram Preview  (0 – Nyquist)")
        ttl.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(ttl)

        ctrl = QHBoxLayout()
        self._rb_before = QRadioButton("Before (original)")
        self._rb_after  = QRadioButton("After (restored)")
        self._rb_before.setChecked(True)
        self._rb_before.toggled.connect(self._on_toggle)
        ctrl.addWidget(self._rb_before)
        ctrl.addWidget(self._rb_after)
        ctrl.addStretch()
        self._lbl_status = QLabel("Awaiting processing…")
        self._lbl_status.setStyleSheet("color: #888; font-size: 11px;")
        ctrl.addWidget(self._lbl_status)
        layout.addLayout(ctrl)

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
        self._fig, self._ax = plt.subplots(
            figsize=(8, 3.6), dpi=96, layout='constrained')
        self._fig.patch.set_facecolor('#1e1e1e')
        self._ax.set_facecolor('#000000')
        self._apply_empty_style()
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setMinimumHeight(240)
        self._canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
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
            ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
            t_col = '#4fc3f7' if self._view == 'before' else '#a5d6a7'
            t_txt = ("Before — original audio spectrogram"
                     if self._view == 'before' else
                     "After — HF-restored audio spectrogram")
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

class AudioRestoreGUI(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio HF Restore  v1.0  (MP3 Repair)")
        self.setMinimumSize(620, 460)
        self._worker        = None
        self._detect_thread = None
        self._adv_expanded  = False
        self._spec_visible  = False
        self._setup_ui()
        QTimer.singleShot(0, self._initial_resize)

    def _initial_resize(self):
        screen  = self.screen()
        avail_h = screen.availableGeometry().height() if screen else 900
        self.resize(760, min(540, avail_h - 40))

    # ── top-level layout ──────────────────────────────────────────────────
    def _setup_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        main = QVBoxLayout(root)
        main.setContentsMargins(0, 0, 0, 0)
        main.setSpacing(0)

        # header
        hdr     = QWidget()
        hdr.setContentsMargins(16, 8, 16, 4)
        hdr_row = QHBoxLayout(hdr)
        hdr_row.setContentsMargins(0, 0, 0, 0)
        lbl_icon  = QLabel("🎛️"); lbl_icon.setFont(QFont("Arial", 18))
        lbl_title = QLabel("Audio HF Restore")
        lbl_title.setFont(QFont("Arial", 15, QFont.Weight.Bold))
        lbl_sub   = QLabel("MP3 Repair")
        lbl_sub.setFont(QFont("Arial", 10))
        lbl_sub.setStyleSheet("color: #2E7D32; font-weight: bold;")
        lbl_ver   = QLabel("v1.0"); lbl_ver.setFont(QFont("Arial", 9))
        lbl_ver.setStyleSheet("color: #888;")
        for w in (lbl_icon, lbl_title, lbl_sub, lbl_ver):
            hdr_row.addWidget(w)
        hdr_row.addStretch()
        main.addWidget(hdr)

        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #ccc;"); main.addWidget(sep)

        # scrollable content
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        inner  = QWidget()
        self._inner = QVBoxLayout(inner)
        self._inner.setContentsMargins(12, 8, 12, 8)
        self._inner.setSpacing(6)
        scroll.setWidget(inner)
        main.addWidget(scroll, stretch=1)

        self._build_file_group()
        self._build_files_list()
        self._build_cutoff_group()
        self._build_option_row()
        self._build_advanced()
        self._build_spectrogram_section()
        self._inner.addStretch()

        bot_sep = QFrame(); bot_sep.setFrameShape(QFrame.Shape.HLine)
        bot_sep.setStyleSheet("color: #ccc;"); main.addWidget(bot_sep)
        self._build_bottom(main)

    # ── File / Folder group ───────────────────────────────────────────────
    def _build_file_group(self):
        grp = QGroupBox("File / Folder")
        gl  = QFormLayout(grp)
        gl.setHorizontalSpacing(8); gl.setVerticalSpacing(6)

        self._edit_input = QLineEdit()
        self._edit_input.textChanged.connect(self._refresh_file_list)
        row_in = QHBoxLayout(); row_in.addWidget(self._edit_input)
        for txt, cb in [("File…", self._browse_file),
                        ("Folder…", self._browse_input_folder)]:
            b = QPushButton(txt); b.setFixedWidth(68); b.clicked.connect(cb)
            row_in.addWidget(b)
        gl.addRow("Input:", row_in)

        self._edit_output = QLineEdit()
        row_out = QHBoxLayout(); row_out.addWidget(self._edit_output)
        b_out = QPushButton("Folder…"); b_out.setFixedWidth(68)
        b_out.clicked.connect(self._browse_output_folder)
        row_out.addWidget(b_out)
        gl.addRow("Output Folder:", row_out)

        hint = QLabel("Supports WAV / FLAC (decode your MP3 first with Audacity or ffmpeg). "
                      "Output is written at the original sample rate as WAV 32-bit float.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #666; font-size: 11px;")
        gl.addRow("", hint)
        self._inner.addWidget(grp)

    # ── Files list ────────────────────────────────────────────────────────
    def _build_files_list(self):
        grp = QGroupBox("Files to Process")
        gl  = QVBoxLayout(grp)
        gl.setContentsMargins(8, 6, 8, 6)
        self._list_files = QListWidget()
        self._list_files.setFixedHeight(72)
        self._list_files.setFont(QFont("Consolas", 9))
        self._list_files.setStyleSheet("background: #f7f7f7; border: 1px solid #ddd;")
        gl.addWidget(self._list_files)
        self._lbl_count = QLabel("No file selected")
        self._lbl_count.setStyleSheet("color: #888; font-size: 11px;")
        gl.addWidget(self._lbl_count)
        self._inner.addWidget(grp)

    # ── Cutoff frequency group (the key new control) ──────────────────────
    def _build_cutoff_group(self):
        grp = QGroupBox("HF Restoration Cutoff")
        gl  = QVBoxLayout(grp)
        gl.setContentsMargins(10, 8, 10, 8)
        gl.setSpacing(6)

        # Mode toggle row
        mode_row = QHBoxLayout()
        self._chk_auto_detect = QCheckBox("Auto-detect from file")
        self._chk_auto_detect.setChecked(True)
        self._chk_auto_detect.stateChanged.connect(self._on_detect_mode_changed)
        mode_row.addWidget(self._chk_auto_detect)

        self._btn_detect = QPushButton("▶ Detect Now")
        self._btn_detect.setFixedWidth(110)
        self._btn_detect.clicked.connect(self._run_detect)
        mode_row.addWidget(self._btn_detect)
        mode_row.addStretch()
        gl.addLayout(mode_row)

        # Manual cutoff row
        manual_row = QHBoxLayout()
        lbl = QLabel("Manual Cutoff:")
        lbl.setStyleSheet("color: #555;")
        manual_row.addWidget(lbl)

        self._spin_cutoff = QSpinBox()
        self._spin_cutoff.setRange(8000, 22050)
        self._spin_cutoff.setValue(16000)
        self._spin_cutoff.setSingleStep(500)
        self._spin_cutoff.setSuffix(" Hz")
        self._spin_cutoff.setFixedWidth(110)
        self._spin_cutoff.setEnabled(False)   # disabled when auto-detect is on
        manual_row.addWidget(self._spin_cutoff)

        self._lbl_detected = QLabel("")
        self._lbl_detected.setStyleSheet("color: #2E7D32; font-size: 11px;")
        manual_row.addWidget(self._lbl_detected)
        manual_row.addStretch()
        gl.addLayout(manual_row)

        # Explanation
        tip = QLabel(
            "Auto-detect scans the spectrum for the characteristic MP3 brick-wall roll-off "
            "and derives the cutoff automatically.  Use Manual if the result looks wrong.")
        tip.setWordWrap(True)
        tip.setStyleSheet("color: #666; font-size: 11px;")
        gl.addWidget(tip)

        self._inner.addWidget(grp)

    # ── Advanced toggle + Spectrum checkbox ───────────────────────────────
    def _build_option_row(self):
        row = QHBoxLayout()
        self._btn_adv = QPushButton("▶  Advanced Options (Expand)")
        self._btn_adv.clicked.connect(self._toggle_advanced)
        row.addWidget(self._btn_adv)

        vsep = QFrame(); vsep.setFrameShape(QFrame.Shape.VLine)
        vsep.setStyleSheet("color: #ccc;"); row.addWidget(vsep)

        self._chk_spectrum = QCheckBox("☰  Preview Spectrogram (single file only)")
        if not _MATPLOTLIB_AVAILABLE:
            self._chk_spectrum.setEnabled(False)
            self._chk_spectrum.setToolTip(
                f"matplotlib not available: {_MATPLOTLIB_ERROR or 'pip install matplotlib'}")
        self._chk_spectrum.stateChanged.connect(self._toggle_spectrum)
        row.addWidget(self._chk_spectrum)
        row.addStretch()
        self._inner.addLayout(row)

    # ── Advanced options ──────────────────────────────────────────────────
    def _build_advanced(self):
        self._adv_frame = QGroupBox("Advanced Options")
        self._adv_frame.setVisible(False)
        tabs = QTabWidget()
        tabs.addTab(self._tab_restore_params(), "Restore Parameters")
        tabs.addTab(self._tab_stages(),         "Pipeline Stages")
        tabs.addTab(self._tab_threads(),        "Threads")
        lay = QVBoxLayout(self._adv_frame)
        lay.addWidget(tabs)
        self._inner.addWidget(self._adv_frame)

    def _tab_restore_params(self):
        w = QWidget(); f = QFormLayout(w)
        f.setContentsMargins(10, 8, 10, 8)
        f.setHorizontalSpacing(10); f.setVerticalSpacing(8)

        self._spin_aliasing = QDoubleSpinBox()
        self._spin_aliasing.setRange(0.0, 1.0); self._spin_aliasing.setValue(0.72)
        self._spin_aliasing.setSingleStep(0.02)
        tip_a = QLabel("0 = no synthesis · 1 = maximum aliasing")
        tip_a.setStyleSheet("color: #666; font-size: 11px;")
        row_a = QHBoxLayout(); row_a.addWidget(self._spin_aliasing); row_a.addWidget(tip_a)
        f.addRow("Synthesis Aliasing:", row_a)

        self._spin_strength = QDoubleSpinBox()
        self._spin_strength.setRange(0.01, 0.5); self._spin_strength.setValue(0.10)
        self._spin_strength.setSingleStep(0.01)
        tip_s = QLabel("spectral extension seed strength")
        tip_s.setStyleSheet("color: #666; font-size: 11px;")
        row_s = QHBoxLayout(); row_s.addWidget(self._spin_strength); row_s.addWidget(tip_s)
        f.addRow("Spectral Strength:", row_s)

        self._spin_splice_fade = QSpinBox()
        self._spin_splice_fade.setRange(50, 5000); self._spin_splice_fade.setValue(500)
        self._spin_splice_fade.setSuffix(" Hz")
        f.addRow("Splice Fade Width:", self._spin_splice_fade)

        self._spin_att_smooth = QDoubleSpinBox()
        self._spin_att_smooth.setRange(0.5, 100); self._spin_att_smooth.setValue(10.0)
        self._spin_att_smooth.setSuffix(" ms")
        f.addRow("Attenuation Smooth:", self._spin_att_smooth)

        self._spin_att_floor = QDoubleSpinBox()
        self._spin_att_floor.setRange(-40, 0); self._spin_att_floor.setValue(-20.0)
        self._spin_att_floor.setSuffix(" dB")
        f.addRow("Attenuation Floor:", self._spin_att_floor)

        self._spin_eq_boost = QDoubleSpinBox()
        self._spin_eq_boost.setRange(0, 30); self._spin_eq_boost.setValue(12.0)
        self._spin_eq_boost.setSuffix(" dB")
        f.addRow("V-Notch Max Boost:", self._spin_eq_boost)

        self._spin_flat_boost = QDoubleSpinBox()
        self._spin_flat_boost.setRange(0, 30); self._spin_flat_boost.setValue(6.0)
        self._spin_flat_boost.setSuffix(" dB")
        f.addRow("Flatten Max Boost:", self._spin_flat_boost)

        self._spin_flat_cut = QDoubleSpinBox()
        self._spin_flat_cut.setRange(0, 30); self._spin_flat_cut.setValue(12.0)
        self._spin_flat_cut.setSuffix(" dB")
        f.addRow("Flatten Max Cut:", self._spin_flat_cut)

        return w

    def _tab_stages(self):
        w = QWidget(); f = QVBoxLayout(w)
        f.setContentsMargins(10, 8, 10, 8); f.setSpacing(6)
        f.setAlignment(Qt.AlignmentFlag.AlignTop)

        def _sec(text, color='#222'):
            lbl = QLabel(f"▎ {text}")
            lbl.setStyleSheet(
                f"font-weight: bold; font-size: 12px; color: {color}; padding-top: 4px;")
            return lbl

        f.addWidget(_sec("Stage 2 — Faithful Splice"))
        self._chk_splice = QCheckBox(
            "Enable lossless faithful splice (original below cutoff, synthesised above)")
        self._chk_splice.setChecked(True); f.addWidget(self._chk_splice)

        f.addWidget(QFrame(frameShape=QFrame.Shape.HLine))
        f.addWidget(_sec("Stage 3 — Dynamic HF Attenuation"))
        self._chk_attenuate = QCheckBox(
            "Enable envelope-follower dynamic modulation "
            "(ties synthesised HF energy to original dynamics)")
        self._chk_attenuate.setChecked(True); f.addWidget(self._chk_attenuate)

        f.addWidget(QFrame(frameShape=QFrame.Shape.HLine))
        f.addWidget(_sec("Stage 4 — STFT Spectral Repair"))
        self._chk_smart_eq = QCheckBox(
            "V-notch repair (fill level dip at the splice point)")
        self._chk_smart_eq.setChecked(True); f.addWidget(self._chk_smart_eq)
        self._chk_auto_flatten = QCheckBox(
            "Auto-flatten (smooth narrow peaks / troughs in the restored band)")
        self._chk_auto_flatten.setChecked(True); f.addWidget(self._chk_auto_flatten)

        f.addWidget(QFrame(frameShape=QFrame.Shape.HLine))
        # Note: Stage 5 (power-law rolloff) intentionally not present
        note = QLabel(
            "ℹ  Stage 5 (Ultra-HF power-law rolloff) is omitted in the restore variant — "
            "we are adding content, not attenuating it.")
        note.setWordWrap(True)
        note.setStyleSheet("color: #888; font-size: 11px; padding: 4px 0;")
        f.addWidget(note)

        f.addWidget(QFrame(frameShape=QFrame.Shape.HLine))
        f.addWidget(_sec("Stage 6 — True-Peak Limiter", '#6A1B9A'))
        self._chk_limiter = QCheckBox("Enable transparent look-ahead true-peak limiter")
        self._chk_limiter.setChecked(True); f.addWidget(self._chk_limiter)
        r5 = QHBoxLayout(); r5.setContentsMargins(18, 0, 0, 0)
        r5.addWidget(QLabel("Ceiling:"))
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

    def _tab_threads(self):
        w = QWidget(); f = QFormLayout(w)
        f.setContentsMargins(10, 8, 10, 8)
        f.setHorizontalSpacing(10); f.setVerticalSpacing(8)

        tr = QHBoxLayout()
        self._chk_multithread = QCheckBox("Multithreaded  (L/R channels in parallel)")
        self._chk_multithread.setChecked(True)
        self._spin_threads = QSpinBox()
        self._spin_threads.setRange(1, mp.cpu_count())
        self._spin_threads.setValue(max(1, mp.cpu_count() - 1))
        tr.addWidget(self._chk_multithread); tr.addWidget(self._spin_threads)
        tr.addWidget(QLabel("threads")); tr.addStretch()
        f.addRow("Threads:", tr)

        note = QLabel(
            "Unlike super-resolution, the restoration pipeline does not chunk-process "
            "files — it operates at the original sample rate so memory usage stays low "
            "regardless of file length.")
        note.setWordWrap(True)
        note.setStyleSheet("color: #666; font-size: 11px;")
        f.addRow("", note)
        return w

    # ── Spectrogram ───────────────────────────────────────────────────────
    def _build_spectrogram_section(self):
        self._spec_widget = SpectrogramWidget()
        self._spec_widget.setVisible(False)
        self._inner.addWidget(self._spec_widget)

    # ── Bottom bar ────────────────────────────────────────────────────────
    def _build_bottom(self, parent_layout):
        bot = QWidget(); bl = QVBoxLayout(bot)
        bl.setContentsMargins(14, 6, 14, 8); bl.setSpacing(4)

        def _bar_row(label_text):
            row = QHBoxLayout()
            lbl = QLabel(label_text); lbl.setFixedWidth(38)
            lbl.setStyleSheet("color: #555; font-size: 11px;")
            bar = QProgressBar(); bar.setRange(0, 100); bar.setValue(0)
            bar.setTextVisible(False); bar.setFixedHeight(14)
            extra = QLabel(); extra.setStyleSheet("color: #555; font-size: 11px;")
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
        self._btn_start.setFixedHeight(30); self._btn_cancel.setFixedHeight(30)
        self._btn_start.clicked.connect(self._on_start)
        self._btn_cancel.clicked.connect(self._on_cancel)
        btn_row.addWidget(self._btn_start); btn_row.addWidget(self._btn_cancel)
        btn_row.addStretch()
        bl.addLayout(btn_row)
        parent_layout.addWidget(bot)

    # ── Toggle / resize ───────────────────────────────────────────────────
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
        h = 580
        if self._adv_expanded: h += 260
        if self._spec_visible:  h += 320
        screen  = self.screen()
        avail_h = screen.availableGeometry().height() if screen else 1080
        self.resize(self.width(), min(h, avail_h - 40))

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

    # ── File list ─────────────────────────────────────────────────────────
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
                f"{len(files)} file(s) found  (WAV / FLAC)")
            self._lbl_count.setStyleSheet("color: #2E7D32; font-size: 11px;")
        else:
            self._lbl_count.setText("No file selected")
            self._lbl_count.setStyleSheet("color: #888; font-size: 11px;")

    # ── Cutoff detection helpers ──────────────────────────────────────────
    def _on_detect_mode_changed(self, state):
        manual = not bool(state)
        self._spin_cutoff.setEnabled(manual)
        self._btn_detect.setEnabled(bool(state))
        if manual:
            self._lbl_detected.setText("")

    def _run_detect(self):
        files = self._collect_files()
        if not files:
            QMessageBox.warning(self, "No file",
                                "Please select an input file first.")
            return
        self._btn_detect.setEnabled(False)
        self._lbl_detected.setText("Detecting…")
        self._detect_thread = DetectThread(str(files[0]))
        self._detect_thread.sig_result.connect(self._on_detected)
        self._detect_thread.sig_error.connect(self._on_detect_error)
        self._detect_thread.start()

    def _on_detected(self, cutoff: int):
        self._lbl_detected.setText(f"Detected: {cutoff} Hz")
        self._spin_cutoff.setValue(cutoff)
        self._btn_detect.setEnabled(True)

    def _on_detect_error(self, msg: str):
        self._lbl_detected.setText("Detection failed")
        QMessageBox.warning(self, "Detection failed", msg)
        self._btn_detect.setEnabled(True)

    # ── Collect parameters ────────────────────────────────────────────────
    def _get_params(self):
        return dict(
            synthesis_aliasing  = self._spin_aliasing.value(),
            spectral_strength   = self._spin_strength.value(),
            enable_splice       = self._chk_splice.isChecked(),
            splice_fade_hz      = self._spin_splice_fade.value(),
            enable_attenuate    = self._chk_attenuate.isChecked(),
            att_smooth_ms       = self._spin_att_smooth.value(),
            att_gain_floor_db   = self._spin_att_floor.value(),
            enable_smart_eq     = self._chk_smart_eq.isChecked(),
            smart_eq_boost      = self._spin_eq_boost.value(),
            enable_auto_flatten = self._chk_auto_flatten.isChecked(),
            flat_max_boost      = self._spin_flat_boost.value(),
            flat_max_cut        = self._spin_flat_cut.value(),
            enable_limiter      = self._chk_limiter.isChecked(),
            limiter_ceiling_db  = self._spin_lim_ceil.value(),
            limiter_lookahead   = self._spin_lim_la.value(),
            limiter_release     = self._spin_lim_rel.value(),
        )

    def _get_cutoff(self):
        """Return None for auto-detect, or the manual value."""
        if self._chk_auto_detect.isChecked():
            return None
        return self._spin_cutoff.value()

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

        self._worker = WorkerThread(files, out_dir,
                                    self._get_params(),
                                    self._get_cutoff())
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
        self._lbl_batch.setText(f"{done}/{total} files")

    def _on_done(self, n: int, errors: list):
        self._btn_start.setEnabled(True)
        self._btn_cancel.setEnabled(False)
        self._chk_spectrum.setEnabled(_MATPLOTLIB_AVAILABLE)
        if errors:
            self._lbl_status.setText(f"Done ({len(errors)} file(s) with errors)")
            QMessageBox.critical(self, "Some Files Failed",
                                 "\n\n".join(errors[:3]) +
                                 ("\n\n…" if len(errors) > 3 else ""))
        else:
            self._lbl_status.setText(f"✓ All done! Processed {n} file(s)")
            QMessageBox.information(
                self, "Done",
                f"All {n} file(s) processed successfully!\n"
                f"Output: {self._edit_output.text()}")


# ============================================================================
# Entry point
# ============================================================================

if __name__ == '__main__':
    mp.freeze_support()

    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)

    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    win = AudioRestoreGUI()
    win.show()
    sys.exit(app.exec())
