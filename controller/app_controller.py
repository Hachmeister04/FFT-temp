import getpass
import os
import time
import logging
import numpy as np
from PyQt5.QtWidgets import QApplication, QFileDialog
from PyQt5.QtCore import QObject, QTimer

from constants import (
    BANDS, SIDES, MIN_NPERSEG, MAX_NFFT,
    DECIMALS_EXCLUSIONS, FREQUENCY_EXCLUSIONS_MAX_N, SPECTROGRAM_MASKS_MAX_N,
    DEFAULT_PREFIX_HDF5, DEFAULT_POSTFIX_HDF5, DEFAULT_FOLDER_HDF5,
    DEFAULT_PREFIX_CONFIG, DEFAULT_POSTFIX_CONFIG, DEFAULT_FOLDER_CONFIG
)
from model.shot_model import ShotModel
from model.state import ReconstructionInput, FrequencyExclusion, SpectrogramMask
from view.main_window import MainWindowView
from view.reconstruction_window import ReconstructionWindow
from view.parameter_panels import ParameterPanels
from view.plot_renderers import PlotRenderer
from utils.helpers import round_to_nearest
from utils.ui_helpers import show_warning

logger = logging.getLogger(__name__)


class AppController(QObject):
    """Wires model and view together. Owns all signal-handling logic."""

    # Coalesce rapid sweep changes (e.g. dragging the slider): only the final
    # position triggers the expensive recompute, this many ms after the last tick.
    _SWEEP_DEBOUNCE_MS = 50

    def __init__(self):
        super().__init__()

        # Create components
        self.model = ShotModel()
        self.panels = ParameterPanels()
        self.view = MainWindowView()
        self.view.set_parameter_panels(self.panels)
        self.renderer = PlotRenderer()

        # Track open reconstruction windows
        self._reconstruction_windows = []

        qApp = QApplication.instance()
        if qApp is not None:
            qApp.aboutToQuit.connect(self._cleanup_all_reconstructions)

        # Suppression flags
        self._suppress_fft_updates = False
        self._suppress_model_sync = False

        # Debounce timer for the sweep recompute pipeline. Single-shot; each
        # sweep change restarts it, so a drag only computes the final position.
        self._sweep_timer = QTimer(self)
        self._sweep_timer.setSingleShot(True)
        self._sweep_timer.timeout.connect(self._do_sweep_recompute)

        # Wire signals — store lambda refs so they can be used with blockSignal
        self._connect_signals()

    def _connect_signals(self):
        """All signal connections in one place.

        Lambdas are stored as attributes so the same object can be passed
        to both connect() and blockSignal().
        """
        p = self.panels

        # File/shot
        self._h_shot_open = lambda: self._on_shot_changed('open')
        self._h_shot_number = lambda: self._on_shot_changed('shot')
        p.file.child('Open').sigValueChanged.connect(self._h_shot_open)
        p.file.child('Shot').sigValueChanged.connect(self._h_shot_number)

        # Initialization
        self._h_init_type = lambda: self._on_init_changed('type')
        self._h_init_value = lambda: self._on_init_changed('value')
        self._h_init_file = lambda: self._on_init_changed('file')
        p.init.child('Type').sigValueChanged.connect(self._h_init_type)
        p.init.child('Value').sigValueChanged.connect(self._h_init_value)
        p.init.child('File').sigValueChanged.connect(self._h_init_file)

        # Config
        p.config.child('Save').sigValueChanged.connect(self._on_save_config)
        p.config.child('Load').sigValueChanged.connect(self._on_load_config)

        # Detector
        self._h_det_band = lambda: self._on_detector_changed('band')
        self._h_det_side = lambda: self._on_detector_changed('side')
        p.detector.child('Band').sigValueChanged.connect(self._h_det_band)
        p.detector.child('Side').sigValueChanged.connect(self._h_det_side)

        # Sweep
        self._h_sweep_slider = lambda: self._on_sweep_changed('slider')
        self._h_sweep_number = lambda: self._on_sweep_changed('number')
        self._h_sweep_timestamp = lambda: self._on_sweep_changed('timestamp')
        p.sweep.child('Sweep').sigValueChanged.connect(self._h_sweep_slider)
        p.sweep.child('Sweep nº').sigValueChanged.connect(self._h_sweep_number)
        p.sweep.child('Timestamp').sigValueChanged.connect(self._h_sweep_timestamp)

        # FFT params
        self._h_fft_nperseg = lambda: self._on_fft_changed('nperseg')
        self._h_fft_noverlap = lambda: self._on_fft_changed('noverlap')
        self._h_fft_nfft = lambda: self._on_fft_changed('nfft')
        self._h_fft_burst = lambda: self._on_fft_changed('burst_size')
        self._h_fft_sub_bg = lambda: self._on_fft_changed('subtract_background')
        self._h_fft_fix_bg = lambda: self._on_fft_changed('fix_background_parameters')
        self._h_fft_bg_sweep = lambda: self._on_fft_changed('background_sweep')
        self._h_fft_bg_burst = lambda: self._on_fft_changed('background_burst_size')
        self._h_fft_sub_disp = lambda: self._on_fft_changed('subtract_dispersion')
        self._h_fft_low = lambda: self._on_fft_changed('low_filter')
        self._h_fft_high = lambda: self._on_fft_changed('high_filter')
        p.fft.child('nperseg').sigValueChanged.connect(self._h_fft_nperseg)
        p.fft.child('noverlap').sigValueChanged.connect(self._h_fft_noverlap)
        p.fft.child('nfft').sigValueChanged.connect(self._h_fft_nfft)
        p.fft.child('burst size (odd)').sigValueChanged.connect(self._h_fft_burst)
        p.fft.child('Scale').sigValueChanged.connect(self._on_scale_or_colormap_changed)
        p.fft.child('Subtract background').sigValueChanged.connect(self._h_fft_sub_bg)
        p.fft.child('Fix background sweep and burst size').sigValueChanged.connect(self._h_fft_fix_bg)
        p.fft.child('Background sweep').sigValueChanged.connect(self._h_fft_bg_sweep)
        p.fft.child('Background burst size (odd)').sigValueChanged.connect(self._h_fft_bg_burst)
        p.fft.child('Subtract dispersion').sigValueChanged.connect(self._h_fft_sub_disp)
        p.fft.child('Color Map').sigValueChanged.connect(self._on_scale_or_colormap_changed)
        p.fft.child('Filters').child('Low Filter').sigValueChanged.connect(self._h_fft_low)
        p.fft.child('Filters').child('High Filter').sigValueChanged.connect(self._h_fft_high)
        p.fft.child('Exclude frequencies').sigAddNew.connect(self._on_add_frequency_exclusion)
        p.fft.child('Spectrogram masks').sigAddNew.connect(self._on_add_spectrogram_mask)

        # Profiles
        p.profiles.child('Coordinates').sigValueChanged.connect(self._on_profile_coord_changed)

        # Reconstruct
        self._h_recon_start = lambda: self._on_reconstruct_params_changed('start')
        self._h_recon_end = lambda: self._on_reconstruct_params_changed('end')
        self._h_recon_step = lambda: self._on_reconstruct_params_changed('step')
        p.reconstruct.child('Start Time').sigValueChanged.connect(self._h_recon_start)
        p.reconstruct.child('End Time').sigValueChanged.connect(self._h_recon_end)
        p.reconstruct.child('Time Step').sigValueChanged.connect(self._h_recon_step)
        p.reconstruct.child('Reconstruct Shot').sigActivated.connect(self._on_request_reconstruct)
        #p.reconstruct.child('Apply Custom Density Cutoff').sigValueChanged.connect(self._on_cutoff_changed)
        p.reconstruct.child('Apply Custom Density Cutoff').sigValueChanged.connect(self._on_cutoff_changed)

    # --- Helpers to sync model ↔ panels ---

    def _sync_detector_from_panels(self):
        """Read current detector selection from panels into model."""
        self.model.detector.band = self.panels.detector.child('Band').value()
        self.model.detector.side = self.panels.detector.child('Side').value()

    def _sync_params_to_panels(self):
        """Update panel values to reflect current band/side's stored params."""
        m = self.model
        p = self.panels
        d = m.detector
        sp = m.spect_params[d.side][d.band]
        filt = m.filters[d.side][d.band]

        p.fft.child('Filters').child('Low Filter').setValue(filt.low, blockSignal=self._h_fft_low)
        p.fft.child('Filters').child('High Filter').setValue(filt.high, blockSignal=self._h_fft_high)
        p.fft.child('nperseg').setValue(sp.nperseg, blockSignal=self._h_fft_nperseg)

        # Update noverlap limits then value
        p.fft.child('noverlap').sigValueChanged.disconnect(self._h_fft_noverlap)
        p.fft.child('noverlap').setLimits((0, sp.nperseg - 1))
        p.fft.child('noverlap').sigValueChanged.connect(self._h_fft_noverlap)
        p.fft.child('noverlap').setValue(sp.noverlap, blockSignal=self._h_fft_noverlap)

        # Update nfft limits then value
        p.fft.child('nfft').sigValueChanged.disconnect(self._h_fft_nfft)
        p.fft.child('nfft').setLimits((sp.nperseg, np.inf))
        p.fft.child('nfft').sigValueChanged.connect(self._h_fft_nfft)
        p.fft.child('nfft').setValue(sp.nfft, blockSignal=self._h_fft_nfft)

        p.fft.child('Subtract background').setValue(sp.subtract_background, blockSignal=self._h_fft_sub_bg)
        p.fft.child('Background sweep').setValue(sp.background_sweep, blockSignal=self._h_fft_bg_sweep)
        p.fft.child('Background burst size (odd)').setValue(sp.background_burst_size, blockSignal=self._h_fft_bg_burst)
        p.fft.child('Subtract dispersion').setValue(
            sp.subtract_dispersion if sp.subtract_dispersion is not None else False,
            blockSignal=self._h_fft_sub_disp,
        )

        # Update frequency exclusion UI (per side)
        p.fft.child('Exclude frequencies').clearChildren()
        self._suppress_model_sync = True
        for excl in m.frequency_exclusions[d.side]:
            self._on_add_frequency_exclusion()
            children = p.fft.child('Exclude frequencies').children()
            children[-1].child('Enabled').setValue(excl.enabled, blockSignal=self._on_frequency_exclusion_changed)
            children[-1].child('from').setValue(excl.low, blockSignal=self._on_frequency_exclusion_changed)
            children[-1].child('to').setValue(excl.high, blockSignal=self._on_frequency_exclusion_changed)
        self._suppress_model_sync = False

        # Update 2D spectrogram mask UI (per band+side)
        p.fft.child('Spectrogram masks').clearChildren()
        self._suppress_model_sync = True
        for mask in m.spectrogram_masks[d.side][d.band]:
            self._on_add_spectrogram_mask()
            children = p.fft.child('Spectrogram masks').children()
            last = children[-1]
            last.child('Enabled').setValue(mask.enabled, blockSignal=self._on_spectrogram_mask_changed)
            last.child('time_min').setValue(mask.t_min, blockSignal=self._on_spectrogram_mask_changed)
            last.child('time_max').setValue(mask.t_max, blockSignal=self._on_spectrogram_mask_changed)
            last.child('f_prob_min').setValue(mask.f_prob_min, blockSignal=self._on_spectrogram_mask_changed)
            last.child('f_prob_max').setValue(mask.f_prob_max, blockSignal=self._on_spectrogram_mask_changed)
            last.child('f_beat_min').setValue(mask.f_beat_min, blockSignal=self._on_spectrogram_mask_changed)
            last.child('f_beat_max').setValue(mask.f_beat_max, blockSignal=self._on_spectrogram_mask_changed)
        self._suppress_model_sync = False

        # Update initialization panels
        iv = m.init_values[d.side]
        p.init.child('Type').setValue(iv.type, blockSignal=self._h_init_type)
        p.init.child('Value').setValue(iv.current_value, blockSignal=self._h_init_value)
        p.init.child('File').setValue(iv.file.name, blockSignal=self._h_init_file)

        if iv.type == 'Default (recommended)':
            p.init.child('Value').setOpts(readonly=True)
            p.init.child('File').setOpts(visible=False)
        elif iv.type == 'Custom':
            p.init.child('Value').setOpts(readonly=False)
            p.init.child('File').setOpts(visible=False)
        elif iv.type == 'From file':
            p.init.child('Value').setOpts(readonly=True)
            p.init.child('File').setOpts(visible=True)

        # Reconstruction time range
        p.reconstruct.child('Start Time').setValue(m.reconstruction_start_time, blockSignal=self._h_recon_start)
        p.reconstruct.child('End Time').setValue(m.reconstruction_end_time, blockSignal=self._h_recon_end)
        p.reconstruct.child('Time Step').setValue(m.reconstruction_time_step, blockSignal=self._h_recon_step)

    # --- Shot loading ---

    def _on_shot_changed(self, source):
        """Handles shot/file load."""
        p = self.panels
        m = self.model

        if source == 'open':
            file_path = p.file.child('Open').value()
            try:
                m.load_shot_from_path(file_path)
                p.file.child('Shot').setValue(m.shot, blockSignal=self._h_shot_number)
            except (ValueError, FileNotFoundError):
                show_warning()
                return
            finally:
                p.file.child('Open').setValue(m.file_path, blockSignal=self._h_shot_open)

        elif source == 'shot':
            shot = p.file.child('Shot').value()
            try:
                m.load_shot_from_number(shot)
                p.file.child('Open').setValue(m.file_path, blockSignal=self._h_shot_open)
            except FileNotFoundError:
                show_warning()
                p.file.child('Shot').setValue(m.shot, blockSignal=self._h_shot_number)
                return

        # Show all parameters
        self.view.show_post_load_params(self.panels)

        # Except parameters that default to hidden states
        #p.reconstruct.child('Apply Custom Density Cutoff').setOpts(readonly=True)
        #p.reconstruct.child('Apply Custom Density Cutoff').setOpts(visible=False)
        self._on_cutoff_changed()  # Update visibility of density cutoff value

        # Sync detector
        self._sync_detector_from_panels()
        d = m.detector

        # Post-load init
        m.post_load_init()
        m.initialize_limiters(d.side, p.sweep.child('Timestamp').value(), define_init_funcs=True)

        # Full recompute + draw
        self._recompute_and_draw_all()

        # Set parameter limits
        ts = m.time_stamps
        p.sweep.child('Sweep').setLimits((0, len(ts) - 1))
        p.sweep.child('Sweep nº').setLimits((0, len(ts) - 1))
        p.sweep.child('Timestamp').setLimits((ts[0], ts[-1]))
        p.sweep.child('Timestamp').setOpts(step=ts[1])
        bs = m.detector.burst_size
        p.fft.child('Background sweep').setLimits((bs // 2, len(ts) - bs // 2 - 1))
        p.fft.child('nperseg').setLimits((MIN_NPERSEG, len(m.current_sweep.data) // 2))
        p.fft.child('noverlap').setLimits((0, p.fft.child('nperseg').value() - 1))
        p.fft.child('nfft').setLimits((p.fft.child('nperseg').value(), MAX_NFFT))
        p.fft.child('Filters').child('Low Filter').setLimits((0, np.inf))
        fft = m.current_fft
        p.fft.child('Filters').child('High Filter').setLimits((abs(fft.f_beat[0] - fft.f_beat[1]), np.inf))
        p.reconstruct.child('Start Time').setLimits((0, len(ts) * (ts[1] - ts[0])))
        p.reconstruct.child('End Time').setLimits((0, len(ts) * (ts[1] - ts[0])))

        default_cfg_name = f"{DEFAULT_PREFIX_CONFIG}{self.model.shot}{DEFAULT_POSTFIX_CONFIG}"
        p.config.child('Save').setOpts(selectFile=default_cfg_name)

    # --- Initialization ---

    def _on_init_changed(self, source):
        """Handle initialization parameter changes."""
        p = self.panels
        m = self.model
        d = m.detector

        if source == 'type':
            value = p.init.child('Type').value()
            m.init_values[d.side].type = value

            if value == 'Default (recommended)':
                p.init.child('Value').setOpts(readonly=True)
                p.init.child('File').setOpts(visible=False)
            elif value == 'Custom':
                p.init.child('Value').setOpts(readonly=False)
                p.init.child('File').setOpts(visible=False)
            elif value == 'From file':
                p.init.child('Value').setOpts(readonly=True)
                p.init.child('File').setOpts(visible=True)

        elif source == 'value':
            m.init_values[d.side].custom_value = p.init.child('Value').value()

        elif source == 'file':
            path = p.init.child('File').value()
            data = np.atleast_2d(np.loadtxt(path, delimiter=','))
            m.init_values[d.side].file.time = data[:, 0]
            m.init_values[d.side].file.position = data[:, 1]
            p.init.child('File').setValue(path + ' (loaded)', blockSignal=self._h_init_file)
            m.init_values[d.side].file.name = p.init.child('File').value()

        m.initialize_limiters(d.side, p.sweep.child('Timestamp').value(), define_init_funcs=True)
        self._draw_group_delays()
        self._draw_profile()

    # --- Detector ---

    def _on_detector_changed(self, source):
        """Band/side change."""
        p = self.panels
        self._sync_detector_from_panels()
        self._sync_params_to_panels()

        self._recompute_sweep()
        self._recompute_fft_and_display()

    # --- Sweep navigation ---

    def _on_sweep_changed(self, source):
        """Sweep/timestamp change.

        Syncs the three sweep widgets (slider, number, timestamp) immediately so
        the UI tracks the input, then debounces the expensive recompute so that
        dragging the slider doesn't queue one full pipeline per integer step.
        """
        p = self.panels
        m = self.model
        ts = m.time_stamps

        if source == 'slider':
            value = p.sweep.child('Sweep').value()
            m.detector.sweep = value
            timestamp = ts[value]
            p.sweep.child('Sweep nº').setValue(value, blockSignal=self._h_sweep_number)
            p.sweep.child('Timestamp').setValue(timestamp, blockSignal=self._h_sweep_timestamp)

        elif source == 'number':
            value = int(p.sweep.child('Sweep nº').value())
            m.detector.sweep = value
            timestamp = ts[value]
            p.sweep.child('Sweep nº').setValue(value, blockSignal=self._h_sweep_number)
            p.sweep.child('Sweep').setValue(value, blockSignal=self._h_sweep_slider)
            p.sweep.child('Timestamp').setValue(timestamp, blockSignal=self._h_sweep_timestamp)

        elif source == 'timestamp':
            value = p.sweep.child('Timestamp').value()
            timestamp = round_to_nearest(value, ts)
            index = np.where(ts == timestamp)
            m.detector.sweep = index[0][0]
            p.sweep.child('Timestamp').setValue(timestamp, blockSignal=self._h_sweep_timestamp)
            p.sweep.child('Sweep').setValue(index[0][0], blockSignal=self._h_sweep_slider)
            p.sweep.child('Sweep nº').setValue(index[0][0], blockSignal=self._h_sweep_number)

        # Coalesce rapid changes: the heavy pipeline runs once the position settles.
        self._sweep_timer.start(self._SWEEP_DEBOUNCE_MS)

    def _do_sweep_recompute(self):
        """Run the expensive sweep pipeline for the current detector.sweep.

        Invoked (debounced) by ``_sweep_timer`` after the sweep position settles.
        """
        t1 = time.perf_counter()

        p = self.panels
        m = self.model

        # Pre-warm linearization cache for all 4 bands in parallel.
        # Each cache miss costs ~80ms; 4 sequential = ~320ms, parallel = ~80ms.
        m.prewarm_linearization_cache(m.detector.sweep)
        t1b = time.perf_counter()

        m.initialize_limiters(m.detector.side, p.sweep.child('Timestamp').value())
        t2 = time.perf_counter()

        self._recompute_sweep()
        t3 = time.perf_counter()

        if not self._suppress_fft_updates:
            self._recompute_fft_and_display()
            t4 = time.perf_counter()

            m.compute_all_beatf()
            t5 = time.perf_counter()

            self._draw_group_delays()
            t6 = time.perf_counter()

            self._draw_profile()
            t7 = time.perf_counter()

            logger.info(
                "SWEEP TIMING: prewarm=%.1fms init=%.1fms sweep=%.1fms fft+display=%.1fms "
                "all_beatf=%.1fms group_delays=%.1fms profile=%.1fms TOTAL=%.1fms",
                (t1b - t1) * 1000, (t2 - t1b) * 1000, (t3 - t2) * 1000, (t4 - t3) * 1000,
                (t5 - t4) * 1000, (t6 - t5) * 1000, (t7 - t6) * 1000,
                (t7 - t1) * 1000,
            )

    # --- FFT parameters ---

    def _on_fft_changed(self, source):
        """FFT param change."""
        p = self.panels
        m = self.model
        d = m.detector
        sp = m.spect_params[d.side][d.band]
        filt = m.filters[d.side][d.band]
        fft = m.current_fft

        if source == 'nperseg':
            value = int(p.fft.child('nperseg').value())
            sp.nperseg = value
            p.fft.child('nperseg').setValue(value, blockSignal=self._h_fft_nperseg)
            self._suppress_fft_updates = True
            p.fft.child('noverlap').setLimits((0, value - 1))
            p.fft.child('nfft').setLimits((value, np.inf))
            self._suppress_fft_updates = False

        elif source == 'noverlap':
            value = int(p.fft.child('noverlap').value())
            sp.noverlap = value
            p.fft.child('noverlap').setValue(value, blockSignal=self._h_fft_noverlap)

        elif source == 'nfft':
            value = int(p.fft.child('nfft').value())
            sp.nfft = value
            p.fft.child('nfft').setValue(value, blockSignal=self._h_fft_nfft)

        elif source == 'burst_size':
            value = int(p.fft.child('burst size (odd)').value())
            if value % 2 == 0:
                value -= 1
            p.fft.child('burst size (odd)').setValue(value, blockSignal=self._h_fft_burst)
            m.detector.burst_size = value

            lower_limit = value // 2
            upper_limit = len(m.time_stamps) - value // 2 - 1
            self._suppress_fft_updates = True
            p.sweep.child('Sweep').setLimits((lower_limit, upper_limit))
            p.sweep.child('Sweep nº').setLimits((lower_limit, upper_limit))
            p.sweep.child('Timestamp').setLimits((m.time_stamps[lower_limit], m.time_stamps[upper_limit]))
            p.fft.child('Background sweep').setLimits((lower_limit, upper_limit))
            self._suppress_fft_updates = False
            
            #update big step of slider to be the new burst size
            list(p.sweep.child('Sweep').items)[0].slider.setPageStep(value)

        elif source == 'subtract_background':
            sp.subtract_background = p.fft.child('Subtract background').value()

        elif source == 'fix_background_parameters':
            value = p.fft.child('Fix background sweep and burst size').value()
            _bg_sweep = int(p.fft.child('Background sweep').value())
            _bg_burst_size = int(p.fft.child('Background burst size (odd)').value())
            if value:
                sweep = sp.background_sweep
                for side in SIDES:
                    for band in BANDS:
                        m.spect_params[side][band].background_sweep = _bg_sweep
                        m.spect_params[side][band].background_burst_size = _bg_burst_size
            p.fft.child('Background sweep').setValue(sp.background_sweep, blockSignal=self._h_fft_bg_sweep)
            p.fft.child('Background burst size (odd)').setValue(sp.background_burst_size, blockSignal=self._h_fft_bg_burst)
            p.fft.child('Background sweep').setLimits((sp.background_burst_size // 2, len(m.time_stamps) - sp.background_burst_size // 2 - 1))

        elif source == 'background_sweep':
            fixed_bg_parameters = p.fft.child('Fix background sweep and burst size').value()
            value = int(p.fft.child('Background sweep').value())
            if fixed_bg_parameters:
                for side in SIDES:
                    for band in BANDS:
                        m.spect_params[side][band].background_sweep = value
            else:
                sp.background_sweep = value
            p.fft.child('Background sweep').setValue(sp.background_sweep, blockSignal=self._h_fft_bg_sweep)

        elif source == 'background_burst_size':
            fixed_bg_parameters = p.fft.child('Fix background sweep and burst size').value()
            value = int(p.fft.child('Background burst size (odd)').value())
            if value % 2 == 0:
                value -= 1
            if fixed_bg_parameters:
                for side in SIDES:
                    for band in BANDS:
                        m.spect_params[side][band].background_burst_size = value
            else:
                sp.background_burst_size = value
            p.fft.child('Background burst size (odd)').setValue(sp.background_burst_size, blockSignal=self._h_fft_bg_burst)
            p.fft.child('Background sweep').setLimits((value // 2, len(m.time_stamps) - value // 2 - 1))

        elif source == 'subtract_dispersion':
            sp.subtract_dispersion = p.fft.child('Subtract dispersion').value()

        elif source == 'low_filter':
            filt.low = p.fft.child('Filters').child('Low Filter').value()
            if fft.f_beat is not None and filt.low + abs(fft.f_beat[1] - fft.f_beat[0]) >= filt.high:
                filt.high = filt.low + abs(fft.f_beat[1] - fft.f_beat[0])
                p.fft.child('Filters').child('High Filter').setValue(filt.high, blockSignal=self._h_fft_high)

        elif source == 'high_filter':
            filt.high = p.fft.child('Filters').child('High Filter').value()
            if fft.f_beat is not None and filt.high - abs(fft.f_beat[1] - fft.f_beat[0]) <= filt.low:
                filt.low = filt.high - abs(fft.f_beat[1] - fft.f_beat[0])
                p.fft.child('Filters').child('Low Filter').setValue(filt.low, blockSignal=self._h_fft_low)

        # Trigger appropriate updates
        if not self._suppress_fft_updates:
            if source in ('low_filter', 'high_filter', 'subtract_background'):
                self._draw_spectrogram()
                m.compute_one_beatf(d.band, d.side)
            elif source in ('burst_size', 'background_burst_size', 'background_sweep', 'fix_background_parameters'):
                for side in SIDES:
                    for band in BANDS:
                        m.compute_background(band, side)
                self._recompute_fft_and_display()
                m.compute_all_beatf()
            else:
                m.compute_background(d.band, d.side)
                self._recompute_fft_and_display()
                m.compute_one_beatf(d.band, d.side)

            self._draw_group_delays()
            self._draw_profile()

    # --- Frequency exclusions (1D, dropped from the merged profile) ---

    def _on_add_frequency_exclusion(self):
        """Add a new frequency exclusion range."""
        p = self.panels
        m = self.model
        d = m.detector

        pos = len(p.fft.child('Exclude frequencies').children())

        # Don't allow more than N exclusions for shotfile compatibility
        if pos < FREQUENCY_EXCLUSIONS_MAX_N:
            p.fft.child('Exclude frequencies').addChild({
                'name': f'{pos + 1}', 'type': 'group', 'children': [
                    {'name': 'Enabled', 'type': 'bool', 'value': True},
                    {'name': 'from', 'type': 'float', 'value': 0, 'suffix': 'Hz', 'siPrefix': True, 'decimals': DECIMALS_EXCLUSIONS},
                    {'name': 'to', 'type': 'float', 'value': 0, 'suffix': 'Hz', 'siPrefix': True, 'decimals': DECIMALS_EXCLUSIONS},
                    {'name': 'Remove', 'type': 'action'},
                ]
            })

            child = p.fft.child('Exclude frequencies').child(f'{pos + 1}')
            child.child('Enabled').sigValueChanged.connect(self._on_frequency_exclusion_changed)
            child.child('from').sigValueChanged.connect(self._on_frequency_exclusion_changed)
            child.child('to').sigValueChanged.connect(self._on_frequency_exclusion_changed)
            child.child('Remove').sigActivated.connect(self._on_remove_frequency_exclusion)

            if not self._suppress_model_sync:
                m.frequency_exclusions[d.side].append(FrequencyExclusion(0.0, 0.0, True))

    def _on_remove_frequency_exclusion(self):
        """Remove a frequency exclusion range."""
        sender = self.sender()

        parent = sender.parent()
        num_of_parent = int(parent.name())
        p = self.panels
        m = self.model
        d = m.detector

        p.fft.child('Exclude frequencies').removeChild(parent)

        # Renumber remaining exclusions
        for i in range(num_of_parent, len(p.fft.child('Exclude frequencies').children()) + 1):
            p.fft.child('Exclude frequencies').child(f'{i + 1}').setName(f'{i}')

        # Remove from model
        m.frequency_exclusions[d.side].pop(num_of_parent - 1)

        self._draw_spectrogram()
        self._draw_group_delays()
        self._draw_profile()

    def _on_frequency_exclusion_changed(self):
        """Frequency exclusion value changed."""
        sender = self.sender()

        if sender.name() == 'from':
            if sender.value() > sender.parent().child('to').value():
                sender.parent().child('to').setValue(sender.value(), blockSignal=self._on_frequency_exclusion_changed)
        elif sender.name() == 'to':
            if sender.value() < sender.parent().child('from').value():
                sender.parent().child('from').setValue(sender.value(), blockSignal=self._on_frequency_exclusion_changed)

        exclusion_num = int(sender.parent().name())
        d = self.model.detector
        excl = self.model.frequency_exclusions[d.side][exclusion_num - 1]
        excl.low = sender.parent().child('from').value()
        excl.high = sender.parent().child('to').value()
        excl.enabled = sender.parent().child('Enabled').value()

        self._draw_spectrogram()
        self._draw_group_delays()
        self._draw_profile()

    # --- Spectrogram masks (2D, blanked before peak-finding) ---

    def _on_add_spectrogram_mask(self):
        """Add a new 2D spectrogram mask for the current detector."""
        p = self.panels
        m = self.model
        d = m.detector

        pos = len(p.fft.child('Spectrogram masks').children())
        if pos < SPECTROGRAM_MASKS_MAX_N:
            p.fft.child('Spectrogram masks').addChild({
                'name': f'{pos + 1}', 'type': 'group', 'children': [
                    {'name': 'Enabled', 'type': 'bool', 'value': True},
                    {'name': 'time_min', 'type': 'float', 'value': 0, 'suffix': 's', 'siPrefix': True, 'decimals': DECIMALS_EXCLUSIONS},
                    {'name': 'time_max', 'type': 'float', 'value': 0, 'suffix': 's', 'siPrefix': True, 'decimals': DECIMALS_EXCLUSIONS},
                    {'name': 'f_prob_min', 'type': 'float', 'value': 0, 'suffix': 'Hz', 'siPrefix': True, 'decimals': DECIMALS_EXCLUSIONS},
                    {'name': 'f_prob_max', 'type': 'float', 'value': 0, 'suffix': 'Hz', 'siPrefix': True, 'decimals': DECIMALS_EXCLUSIONS},
                    {'name': 'f_beat_min', 'type': 'float', 'value': 0, 'suffix': 'Hz', 'siPrefix': True, 'decimals': DECIMALS_EXCLUSIONS},
                    {'name': 'f_beat_max', 'type': 'float', 'value': 0, 'suffix': 'Hz', 'siPrefix': True, 'decimals': DECIMALS_EXCLUSIONS},
                    {'name': 'Remove', 'type': 'action'},
                ]
            })

            child = p.fft.child('Spectrogram masks').child(f'{pos + 1}')
            child.child('Enabled').sigValueChanged.connect(self._on_spectrogram_mask_changed)
            child.child('time_min').sigValueChanged.connect(self._on_spectrogram_mask_changed)
            child.child('time_max').sigValueChanged.connect(self._on_spectrogram_mask_changed)
            child.child('f_prob_min').sigValueChanged.connect(self._on_spectrogram_mask_changed)
            child.child('f_prob_max').sigValueChanged.connect(self._on_spectrogram_mask_changed)
            child.child('f_beat_min').sigValueChanged.connect(self._on_spectrogram_mask_changed)
            child.child('f_beat_max').sigValueChanged.connect(self._on_spectrogram_mask_changed)
            child.child('Remove').sigActivated.connect(self._on_remove_spectrogram_mask)

            if not self._suppress_model_sync:
                # Default the time gate to the full shot span so a new mask is
                # active for the currently displayed sweep right away.
                mask = SpectrogramMask()
                if m.time_stamps is not None and len(m.time_stamps):
                    mask.t_min = float(m.time_stamps[0])
                    mask.t_max = float(m.time_stamps[-1])
                    child.child('time_min').setValue(mask.t_min, blockSignal=self._on_spectrogram_mask_changed)
                    child.child('time_max').setValue(mask.t_max, blockSignal=self._on_spectrogram_mask_changed)
                m.spectrogram_masks[d.side][d.band].append(mask)

    def _on_remove_spectrogram_mask(self):
        """Remove a 2D spectrogram mask."""
        sender = self.sender()

        parent = sender.parent()
        num_of_parent = int(parent.name())
        p = self.panels
        m = self.model
        d = m.detector

        p.fft.child('Spectrogram masks').removeChild(parent)

        # Renumber remaining masks
        for i in range(num_of_parent, len(p.fft.child('Spectrogram masks').children()) + 1):
            p.fft.child('Spectrogram masks').child(f'{i + 1}').setName(f'{i}')

        # Remove from model
        m.spectrogram_masks[d.side][d.band].pop(num_of_parent - 1)

        self._recompute_spectrogram_masks()

    # min -> max partner for each mask bound pair
    _MASK_BOUND_PAIRS = {
        'time_min': 'time_max', 'time_max': 'time_min',
        'f_prob_min': 'f_prob_max', 'f_prob_max': 'f_prob_min',
        'f_beat_min': 'f_beat_max', 'f_beat_max': 'f_beat_min',
    }

    def _on_spectrogram_mask_changed(self):
        """A 2D spectrogram mask value changed."""
        sender = self.sender()

        # Keep each min <= max: editing a min above its max bumps the max up,
        # editing a max below its min bumps the min down (mirrors frequency exclusions).
        if sender.name() in self._MASK_BOUND_PAIRS:
            partner = sender.parent().child(self._MASK_BOUND_PAIRS[sender.name()])
            if sender.name().endswith('_min') and sender.value() > partner.value():
                partner.setValue(sender.value(), blockSignal=self._on_spectrogram_mask_changed)
            elif sender.name().endswith('_max') and sender.value() < partner.value():
                partner.setValue(sender.value(), blockSignal=self._on_spectrogram_mask_changed)

        mask_num = int(sender.parent().name())
        d = self.model.detector
        mask = self.model.spectrogram_masks[d.side][d.band][mask_num - 1]
        mask.enabled = sender.parent().child('Enabled').value()
        mask.t_min = sender.parent().child('time_min').value()
        mask.t_max = sender.parent().child('time_max').value()
        mask.f_prob_min = sender.parent().child('f_prob_min').value()
        mask.f_prob_max = sender.parent().child('f_prob_max').value()
        mask.f_beat_min = sender.parent().child('f_beat_min').value()
        mask.f_beat_max = sender.parent().child('f_beat_max').value()

        self._recompute_spectrogram_masks()

    def _recompute_spectrogram_masks(self):
        """Re-run peak-finding after a mask change. Masks blank the spectrogram
        before the max beat frequency is found, so a plain redraw is not enough."""
        self._draw_spectrogram()         # compute_current_display + draw (current detector)
        self.model.compute_all_beatf()   # refresh beat_frequencies for all detectors
        self._draw_group_delays()        # compute_aggregated_delays + draw
        self._draw_profile()

    # --- Visual-only changes ---

    def _on_scale_or_colormap_changed(self):
        """Scale or colormap change — just redraw, no recompute."""
        self._draw_spectrogram()

    def _on_profile_coord_changed(self):
        """Profile coordinate mode change."""
        self._draw_profile()

    # --- Reconstruct params ---

    def _on_reconstruct_params_changed(self, source):
        """Validate start/end times and sync them into the model."""
        p = self.panels
        if source == 'start':
            if p.reconstruct.child('Start Time').value() > p.reconstruct.child('End Time').value():
                p.reconstruct.child('End Time').setValue(
                    p.reconstruct.child('Start Time').value(),
                    blockSignal=self._h_recon_end,
                )
        elif source == 'end':
            if p.reconstruct.child('End Time').value() < p.reconstruct.child('Start Time').value():
                p.reconstruct.child('Start Time').setValue(
                    p.reconstruct.child('End Time').value(),
                    blockSignal=self._h_recon_start,
                )
        
        m = self.model
        m.reconstruction_start_time = p.reconstruct.child('Start Time').value()
        m.reconstruction_end_time = p.reconstruct.child('End Time').value()
        m.reconstruction_time_step = p.reconstruct.child('Time Step').value()

    def _on_cutoff_changed(self):
        """Toggle density cutoff visibility."""
        value = self.panels.reconstruct.child('Apply Custom Density Cutoff').value()
        if value == 'Custom':
            self.panels.reconstruct.child('Density Cutoff Value').setOpts(title='Density Cutoff Value')
            self.panels.reconstruct.child('Density Cutoff Value').setOpts(visible=True, suffix='m^-3', siPrefix=False, delay=0)
        elif value in ['H-0', 'H-1']:
            self.panels.reconstruct.child('Density Cutoff Value').setOpts(title=f'Density Cutoff Multiplier')
            self.panels.reconstruct.child('Density Cutoff Value').setOpts(visible=True, suffix='', siPrefix=False, delay=0)
        elif value == 'None':
            self.panels.reconstruct.child('Density Cutoff Value').setOpts(visible=False)

    # --- Reconstruction ---

    def _on_request_reconstruct(self):
        """Request reconstruction in separate thread."""
        m = self.model
        p = self.panels

        write_hdf5 = p.reconstruct.child('Reconstruction Output').child('HDF5').value()

        # If HDF5 is enabled, let the user choose the output directory and filename
        if write_hdf5:
            default_name = DEFAULT_FOLDER_HDF5 + DEFAULT_PREFIX_HDF5 + str(m.shot) + DEFAULT_POSTFIX_HDF5
            hdf5_destination_path, _ = QFileDialog.getSaveFileName(
                self.view, 'Save HDF5 File', default_name, 'HDF5 Files (*.h5)',
            )
            if not hdf5_destination_path:
                return  # User cancelled — abort reconstruction
            if not hdf5_destination_path.endswith('.h5'):
                hdf5_destination_path += '.h5'
  
        else:
            hdf5_destination_path = None

        params = ReconstructionInput(
            shot=m.shot,
            file_path=m.file_path,
            spect_params=m.spect_params,
            filters=m.filters,
            frequency_exclusions=m.frequency_exclusions,
            spectrogram_masks=m.spectrogram_masks,
            burst_size=m.detector.burst_size,
            start_time=p.reconstruct.child('Start Time').value(),
            end_time=p.reconstruct.child('End Time').value(),
            time_step=p.reconstruct.child('Time Step').value(),
            custom_density_cutoff=p.reconstruct.child('Apply Custom Density Cutoff').value(),
            density_cutoff_value=p.reconstruct.child('Density Cutoff Value').value(),
            write_private_shotfile=p.reconstruct.child('Reconstruction Output').child('Private Shotfile').value(),
            write_public_shotfile=p.reconstruct.child('Reconstruction Output').child('Public Shotfile').value(),
            write_hdf5=write_hdf5,
            hdf5_destination_path=hdf5_destination_path,
            get_init_lfs=lambda time: m.get_init('LFS', time),
            get_init_hfs=lambda time: m.get_init('HFS', time),
        )

        window = ReconstructionWindow(params)
        self._reconstruction_windows.append(window)
        window.destroyed.connect(
            lambda: self._reconstruction_windows.remove(window)
            if window in self._reconstruction_windows else None
        )
        window.show()

    def _cleanup_all_reconstructions(self):
        """Clean up all running reconstruction threads on app quit."""
        for window in list(self._reconstruction_windows):
            window.cleanup_thread()

    # --- Config ---

    def _on_save_config(self):
        """Save current configuration."""
        path = self.panels.config.child('Save').value()
        self.model.save_config(path)
        self.panels.config.child('Save').setValue(path + ' (saved)', blockSignal=self._on_save_config)
        self.panels.config.child('Load').setValue('', blockSignal=self._on_load_config)

    def _on_load_config(self):
        """Load configuration from file."""
        path = self.panels.config.child('Load').value()
        self.model.load_config(path)
        self._sync_params_to_panels()

        # Force change signal to handle sweep number and plot everything
        #TODO: should't this be in _sync_params_to_panels?
        p = self.panels
        p.fft.child('burst size (odd)').setValue(2, blockSignal=self._h_fft_burst)
        p.fft.child('burst size (odd)').setValue(self.model.detector.burst_size)

        p.config.child('Load').setValue(path + ' (loaded)', blockSignal=self._on_load_config)
        p.config.child('Save').setValue('', blockSignal=self._on_save_config)

    # --- Recompute + redraw orchestration ---

    def _recompute_and_draw_all(self):
        """Full cascade after shot load."""
        self._recompute_sweep()
        self._recompute_fft_and_display()
        self.model.compute_all_beatf()
        self._draw_group_delays()
        self._draw_profile()

    def _recompute_sweep(self):
        """Compute sweep data and render."""
        _t0 = time.perf_counter()
        self.model.compute_sweep()
        _t1 = time.perf_counter()
        sw = self.model.current_sweep
        self.renderer.draw_sweep(self.view.plot_sweep, sw.x_data, sw.data, sw.signal_type)
        _t2 = time.perf_counter()
        logger.info(
            "  _recompute_sweep: compute=%.1fms draw=%.1fms",
            (_t1 - _t0) * 1000, (_t2 - _t1) * 1000,
        )

    def _recompute_fft_and_display(self):
        """Compute FFT + display data and render spectrogram with all overlays."""
        self.model.compute_current_fft()
        self.model.compute_current_display()
        self._draw_spectrogram_only()

    def _draw_spectrogram(self):
        """Recompute display data then redraw spectrogram. For standalone callers."""
        self.model.compute_current_display()
        self._draw_spectrogram_only()

    def _draw_spectrogram_only(self):
        """Render spectrogram with all overlays. Assumes display data is current."""
        m = self.model
        d = m.detector
        fft = m.current_fft
        sp = m.spect_params[d.side][d.band]
        filt = m.filters[d.side][d.band]
        disp = m.current_display

        scale = self.panels.fft.child('Scale').value()
        colormap = self.panels.fft.child('Color Map').value()

        self.view.colorBar = self.renderer.draw_spectrogram(
            self.view.plot_spect, fft.Sxx, scale, colormap,
            sp.nperseg, sp.noverlap,
            fft.f, fft.f_beat, fft.f_probe, fft.fs, d.band,
            self.view.colorBar,
        )

        self.renderer.draw_dispersion_line(self.view.plot_spect, fft.f_probe, disp.y_dis)
        self.renderer.draw_filter_lines(self.view.plot_spect, fft.f_probe, disp.y_dis, filt.low, filt.high)
        self.renderer.draw_beatf_on_spectrogram(
            self.view.plot_spect, fft.f_probe, disp.y_beatf,
            m.frequency_exclusions[d.side], d.side,
        )

        # Shade the 2D spectrogram masks active for the current sweep.
        timestamp = m.time_stamps[d.sweep] if m.time_stamps is not None else 0
        self.renderer.draw_spectrogram_masks(
            self.view.plot_spect, m.spectrogram_masks[d.side][d.band],
            timestamp, fft.f_probe, fft.f_beat,
        )

    def _draw_group_delays(self):
        """Compute aggregated delays and render group delay plot."""
        m = self.model
        m.compute_aggregated_delays()
        self.renderer.draw_group_delays(
            self.view.plot_beatf, m.beat_frequencies, m.frequency_exclusions,
            m.aggregated_hfs, m.aggregated_lfs,
        )

    def _draw_profile(self):
        """Compute and render density profile."""
        m = self.model
        p = self.panels
        coord = p.profiles.child('Coordinates').value()
        timestamp = p.sweep.child('Timestamp').value()

        r_HFS, ne_HFS, r_LFS, ne_LFS = m.compute_profile(coord, timestamp)
        self.renderer.draw_profile(self.view.plot_profile, r_HFS, ne_HFS, r_LFS, ne_LFS, coord)
