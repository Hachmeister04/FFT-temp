import logging
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from scipy.fft import fftshift
import scipy.constants as cst
import rpspy

# Try to load from spectrofast for faster spectrogram computation; fall back to scipy if not available
try:
    from spectrofast import spectrogram as fast_spectrogram
    def spectrogram(args, **kwargs):
        return fast_spectrogram(args, **kwargs, planner="estimate", use_wisdom=False)
except ImportError:
    from scipy.signal import spectrogram

from constants import (
    BANDS, SIDES,
    DEFAULT_LINEARIZATION_SWEEP,
    PROFILE_INVERSION_RESOLUTION,
    DEFAULT_START_TIME, DEFAULT_END_TIME, DEFAULT_TIMESTEP, 
    DEFAULT_CUSTOM_DENSITY_CUTOFF, DEFAULT_CUSTOM_DENSITY_CUTOFF_VALUE
)
from model.state import (
    SpectrogramParams, FilterRange, FrequencyExclusion, SpectrogramMask,
    InitValues, InitFileData, BeatFrequencyData,
    CurrentSweepData, CurrentFFTData, CurrentDisplayData,
    AggregatedDelayData, DetectorSelection,
)
from utils.helpers import cached_get_auto_linearization_from_shares, r_to_rho

logger = logging.getLogger(__name__)


class ShotModel:
    """Pure data model for reflectometry analysis. No PyQt dependency."""

    def __init__(self):
        # Shot identity
        self.shot = 0
        self.file_path = ''
        self.time_stamps = None

        # Per-detector state: Dict[side][band]
        self.spect_params = {
            side: {
                band: SpectrogramParams(
                    subtract_dispersion=True if band == 'V' else False
                ) for band in BANDS
            } for side in SIDES
        }
        self.filters = {
            side: {band: FilterRange() for band in BANDS} for side in SIDES
        }
        self.beat_frequencies = {
            side: {band: BeatFrequencyData() for band in BANDS} for side in SIDES
        }
        self.background_spectrograms = {
            side: {band: None for band in BANDS} for side in SIDES
        }

        # Per-side 1D probing-frequency intervals dropped from the merged profile
        self.frequency_exclusions = {side: [] for side in SIDES}
        self.init_values = {side: InitValues() for side in SIDES}

        # Per-detector (band+side) 2D spectrogram boxes blanked before peak-finding
        self.spectrogram_masks = {side: {band: [] for band in BANDS} for side in SIDES}

        # Current detector selection
        self.detector = DetectorSelection()

        # Reconstruction time range
        self.reconstruction_start_time = DEFAULT_START_TIME
        self.reconstruction_end_time = DEFAULT_END_TIME
        self.reconstruction_time_step = DEFAULT_TIMESTEP

        # Current computed data
        self.current_sweep = CurrentSweepData()
        self.current_fft = CurrentFFTData()
        self.current_display = CurrentDisplayData()

        # Aggregated delay data
        self.aggregated_hfs = AggregatedDelayData()
        self.aggregated_lfs = AggregatedDelayData()

        # Cached shot-level data
        self.sampling_frequency = None

        # Persistent thread pool for parallel computation
        self._executor = ThreadPoolExecutor(max_workers=4)

        # Limiter/initialization
        self.inner_limiter = 0.0
        self.outer_limiter = 0.0
        self.hfs_gd_at_zero_fp = 0.0
        self.lfs_gd_at_zero_fp = 0.0
        self.get_init = None

        # Custom density cutoff
        self.custom_density_cutoff = DEFAULT_CUSTOM_DENSITY_CUTOFF
        self.custom_density_cutoff_value = DEFAULT_CUSTOM_DENSITY_CUTOFF_VALUE

    # --- Shot loading ---

    def load_shot_from_path(self, file_path):
        """Load shot from file path. Raises FileNotFoundError/ValueError on failure."""
        from utils.helpers import get_shot_from_path
        shot = get_shot_from_path(file_path)
        rpspy.get_timestamps(shot, file_path)
        self.shot = shot
        self.file_path = file_path

    def load_shot_from_number(self, shot):
        """Load shot by number. Raises FileNotFoundError on failure."""
        from utils.helpers import get_path_from_shot
        file_path = get_path_from_shot(shot)
        rpspy.get_timestamps(shot, file_path)
        self.shot = shot
        self.file_path = file_path

    def post_load_init(self):
        """Run after successful shot load: backgrounds, limiters, timestamps."""
        self.sampling_frequency = rpspy.get_sampling_frequency(self.shot, self.file_path)
        self.time_stamps = rpspy.get_timestamps(self.shot, self.file_path)
        self.compute_all_backgrounds()
        self._init_default_limiters()

    def _init_default_limiters(self):
        """Set default limiter positions from rpspy."""
        self.inner_limiter = rpspy.get_reflectometry_limiter_position('hfs')
        self.hfs_gd_at_zero_fp = 2 * (self.inner_limiter - rpspy.get_antenna_position('hfs')) / cst.c
        self.init_values['HFS'].current_value = self.inner_limiter

        self.outer_limiter = rpspy.get_reflectometry_limiter_position('lfs')
        self.lfs_gd_at_zero_fp = 2 * (rpspy.get_antenna_position('lfs') - self.outer_limiter) / cst.c
        self.init_values['LFS'].current_value = self.outer_limiter

    # --- Initialization ---

    def initialize_limiters(self, side, timestamp, update_all_sides=True, define_init_funcs=False):
        """Compute limiter positions based on init_values.

        Args:
            side: Current active side ('HFS' or 'LFS').
            timestamp: Current sweep timestamp.
            update_all_sides: If True, also process 'From file' init for both sides.
            define_init_funcs: If True, create self.get_init closure.
        """
        iv = self.init_values[side]

        if iv.type == 'Default (recommended)':
            if side == 'HFS':
                self.inner_limiter = rpspy.get_reflectometry_limiter_position('hfs')
                self.hfs_gd_at_zero_fp = 2 * (self.inner_limiter - rpspy.get_antenna_position('hfs')) / cst.c
            elif side == 'LFS':
                self.outer_limiter = rpspy.get_reflectometry_limiter_position('lfs')
                self.lfs_gd_at_zero_fp = 2 * (rpspy.get_antenna_position('lfs') - self.outer_limiter) / cst.c

        elif iv.type == 'Custom':
            if side == 'HFS':
                self.inner_limiter = iv.custom_value
                self.hfs_gd_at_zero_fp = 2 * (self.inner_limiter - rpspy.get_antenna_position('hfs')) / cst.c
            elif side == 'LFS':
                self.outer_limiter = iv.custom_value
                self.lfs_gd_at_zero_fp = 2 * (rpspy.get_antenna_position('lfs') - self.outer_limiter) / cst.c

        if update_all_sides:
            for s in SIDES:
                ivs = self.init_values[s]
                if ivs.type == 'From file' and ivs.file.name != '':
                    limiter = np.interp(timestamp, ivs.file.time, ivs.file.position)
                    if s == 'HFS':
                        self.inner_limiter = limiter
                        self.hfs_gd_at_zero_fp = 2 * (self.inner_limiter - rpspy.get_antenna_position('hfs')) / cst.c
                        ivs.current_value = self.inner_limiter
                    if s == 'LFS':
                        self.outer_limiter = limiter
                        self.lfs_gd_at_zero_fp = 2 * (rpspy.get_antenna_position('lfs') - self.outer_limiter) / cst.c
                        ivs.current_value = self.outer_limiter

        if side == 'HFS':
            self.init_values[side].current_value = self.inner_limiter
        elif side == 'LFS':
            self.init_values[side].current_value = self.outer_limiter

        if define_init_funcs:
            def get_init(s, time):
                if self.init_values[s].type == 'From file':
                    return np.interp(time, self.init_values[s].file.time, self.init_values[s].file.position)
                else:
                    return self.inner_limiter if s == 'HFS' else self.outer_limiter
            self.get_init = get_init

    def get_current_limiter_value(self, side):
        """Get the current limiter value for the given side."""
        if side == 'HFS':
            return self.inner_limiter
        return self.outer_limiter

    # --- Sweep computation ---

    def prewarm_linearization_cache(self, sweep):
        """Pre-warm the linearization cache for all 4 bands in parallel.

        Each rpspy.get_linearization call takes ~80ms on cache miss.
        Running 4 bands sequentially = ~320ms; in parallel = ~80ms.
        """
        futures = [
            self._executor.submit(
                cached_get_auto_linearization_from_shares,
                self.shot, band, sweep,
            )
            for band in BANDS
        ]
        for f in futures:
            f.result()

    def compute_sweep(self):
        """Compute linearized sweep data for current detector."""
        import time as _time
        d = self.detector
        signal_type = 'complex' if d.band == 'V' else 'real'

        _t0 = _time.perf_counter()
        data = rpspy.get_band_signal(
            self.shot, self.file_path, d.band, d.side, signal_type, d.sweep
        )[0]
        data -= 2**11 if signal_type == 'real' else (2**11 + 1j * 2**11)
        _t1 = _time.perf_counter()

        x_data = cached_get_auto_linearization_from_shares(self.shot, d.band, d.sweep)
        _t2 = _time.perf_counter()

        x_data, data = rpspy.linearize(x_data, data)
        _t3 = _time.perf_counter()

        logger.info(
            "  compute_sweep: get_band_signal=%.1fms linearization_cache=%.1fms linearize=%.1fms",
            (_t1 - _t0) * 1000, (_t2 - _t1) * 1000, (_t3 - _t2) * 1000,
        )

        self.current_sweep = CurrentSweepData(
            data=data,
            x_data=x_data,
            signal_type=signal_type,
        )

    # --- Spectrogram computation ---

    def compute_spectrogram(self, band, side, nperseg, noverlap, nfft, sweep, burst_size, subtract_dispersion):
        """Calculate spectrogram for a given band and side.

        Returns:
            tuple: (f, fs, f_beat, t, f_probe, Sxx)
        """
        signal_type = 'complex' if band == 'V' else 'real'

        burst = rpspy.get_band_signal(
            self.shot, self.file_path, band, side, signal_type, sweep - burst_size // 2, burst_size
        )

        f = cached_get_auto_linearization_from_shares(self.shot, band, sweep)

        f, linearized_burst = rpspy.linearize(f, burst)

        fs = self.sampling_frequency

        if band == 'V':
            linearized_burst -= (2**11 + 1j * 2**11)
        else:
            linearized_burst -= 2**11

        if subtract_dispersion is True:
            df_dt = (f[-1] - f[0]) / ((len(f) - 1) * (1 / fs))
            correction = rpspy.get_dispersion_phase(self.shot, band, side, f[0], df_dt, np.arange(len(f)) / fs)
            corrected_burst = linearized_burst * correction
        else:
            corrected_burst = linearized_burst

        two_sided = np.iscomplexobj(corrected_burst)

        f_beat, t, Sxx = spectrogram(
            corrected_burst,
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            return_onesided=not two_sided,
            detrend=False,
            window='boxcar',
        )

        f_probe = np.interp(t, np.arange(len(f)) / fs, f)

        if two_sided:
            f_beat = fftshift(f_beat)
            Sxx = fftshift(Sxx, axes=-2)

        # Average across burst
        Sxx = np.average(Sxx, axis=0)

        return f, fs, f_beat, t, f_probe, Sxx

    def compute_current_fft(self):
        """Compute full FFT pipeline for current detector selection."""
        d = self.detector
        sp = self.spect_params[d.side][d.band]
        f, fs, f_beat, t, f_probe, Sxx = self.compute_spectrogram(
            d.band, d.side, sp.nperseg, sp.noverlap, sp.nfft,
            d.sweep, d.burst_size, sp.subtract_dispersion
        )
        self.current_fft = CurrentFFTData(
            f=f, fs=fs, f_beat=f_beat, t=t, f_probe=f_probe, unfiltered_Sxx=Sxx,
        )

    # --- Dispersion ---

    def compute_dispersion(self, band, side, f_probe, t, subtract):
        """Compute dispersion for the selected band and side."""
        if subtract is True:
            return np.zeros(len(f_probe))
        else:
            k = (f_probe[-1] - f_probe[0]) / (t[-1] - t[0])
            return k * rpspy.aug_tgcorr2(band, side, f_probe * 1e-9, self.shot)

    # --- Beat frequency ---

    def compute_beatf(self, band, side, Sxx, y_dis, f_beat, fs, f_probe):
        """Compute beat frequency from spectrogram data."""
        filt = self.filters[side][band]

        Sxx[np.broadcast_to(f_beat[:, None], Sxx.shape) <= y_dis + filt.low] = Sxx.min()
        Sxx[np.broadcast_to(f_beat[:, None], Sxx.shape) >= y_dis + filt.high] = Sxx.min()

        sxx_min = Sxx.min()

        # FrequencyExclusion is intentionally NOT applied here: blanking a whole
        # f_beat column makes the quadratic peak fit return NaN. Those points are
        # dropped from the final curve in compute_aggregated_delays instead.

        # SpectrogramMask (per band/side, 2D box, gated by the current sweep timestamp)
        ts = self.time_stamps[self.detector.sweep]
        for mask in self.spectrogram_masks[side][band]:
            if not mask.enabled or not (mask.t_min <= ts <= mask.t_max):
                continue
            cols = (f_probe >= mask.f_prob_min) & (f_probe <= mask.f_prob_max)
            rows = (f_beat >= mask.f_beat_min) & (f_beat <= mask.f_beat_max)
            if cols.any() and rows.any():
                Sxx[np.ix_(rows, cols)] = sxx_min

        y_max, _ = rpspy.column_wise_max_with_quadratic_interpolation(Sxx)
        y_max *= abs(f_beat[1] - f_beat[0])

        # For two-sided (fftshifted) spectrograms, f_beat[0] = -fs/2; shift index→frequency.
        if f_beat[0] < 0:
            y_max += f_beat[0]

        return y_max

    def compute_one_beatf(self, band, side):
        """Update beat freq data for one band/side."""
        d = self.detector
        if side == d.side and band == d.band:
            fft = self.current_fft
            disp = self.current_display
            df_dt = (fft.f[-1] - fft.f[0]) / ((len(fft.f) - 1) * (1 / fft.fs))
            y_beat_time = (disp.y_beatf - disp.y_dis) / df_dt
            self.beat_frequencies[side][band] = BeatFrequencyData(
                f_probe=fft.f_probe, y_beatf=disp.y_beatf,
                y_beat_time=y_beat_time, df_dt=df_dt,
            )
        else:
            self.beat_frequencies[side][band] = self._compute_one_beatf_result(band, side)

    def _compute_one_beatf_result(self, band, side):
        """Compute beat freq for one non-current band/side. Returns result (thread-safe)."""
        d = self.detector
        sp = self.spect_params[side][band]
        f, fs, f_beat, t, f_probe, Sxx = self.compute_spectrogram(
            band, side, sp.nperseg, sp.noverlap, sp.nfft,
            d.sweep, d.burst_size, sp.subtract_dispersion,
        )

        if sp.subtract_background:
            Sxx = self.background_subtract(Sxx, band, side)

        y_dis = self.compute_dispersion(band, side, f_probe, t, sp.subtract_dispersion)
        y_beatf = self.compute_beatf(band, side, Sxx, y_dis, f_beat, fs, f_probe)

        df_dt = (f[-1] - f[0]) / ((len(f) - 1) * (1 / fs))
        y_beat_time = (y_beatf - y_dis) / df_dt

        return BeatFrequencyData(
            f_probe=f_probe, y_beatf=y_beatf,
            y_beat_time=y_beat_time, df_dt=df_dt,
        )

    def compute_all_beatf(self):
        """Update beat frequencies for all 8 detectors in parallel."""
        import time as _time
        _t_start = _time.perf_counter()
        d = self.detector

        # Current detector — instant, uses cached FFT/display data
        fft = self.current_fft
        disp = self.current_display
        df_dt = (fft.f[-1] - fft.f[0]) / ((len(fft.f) - 1) * (1 / fft.fs))
        y_beat_time = (disp.y_beatf - disp.y_dis) / df_dt
        self.beat_frequencies[d.side][d.band] = BeatFrequencyData(
            f_probe=fft.f_probe, y_beatf=disp.y_beatf,
            y_beat_time=y_beat_time, df_dt=df_dt,
        )

        # Remaining 7 detectors — run in parallel
        others = [
            (band, side) for side in SIDES for band in BANDS
            if not (side == d.side and band == d.band)
        ]

        futures = {
            self._executor.submit(self._compute_one_beatf_result, band, side): (band, side)
            for band, side in others
        }
        for future in futures:
            band, side = futures[future]
            self.beat_frequencies[side][band] = future.result()

        _t_end = _time.perf_counter()
        logger.info("  all_beatf: %.1fms (7 detectors parallel)", (_t_end - _t_start) * 1000)

    # --- Background ---

    def compute_background(self, band, side):
        """Calculate and store background spectrogram."""
        sp = self.spect_params[side][band]
        burst_size = sp.background_burst_size

        # Clamp background sweep to valid range so the burst window
        # [sweep - burst_size//2, sweep + burst_size//2] stays within the data.
        half = burst_size // 2
        n_sweeps = len(self.time_stamps)
        lower = half
        upper = max(lower, n_sweeps - half - 1)
        sweep = int(np.clip(sp.background_sweep, lower, upper))

        _, _, _, _, _, Sxx = self.compute_spectrogram(
            band, side, sp.nperseg, sp.noverlap, sp.nfft,
            sweep, burst_size, sp.subtract_dispersion,
        )
        self.background_spectrograms[side][band] = Sxx

    def compute_all_backgrounds(self):
        """Calculate backgrounds for all band/side combinations."""
        for side in SIDES:
            for band in BANDS:
                self.compute_background(band, side)

    def background_subtract(self, Sxx, band, side):
        """Subtract background from spectrogram."""
        return np.clip(
            Sxx - self.background_spectrograms[side][band],
            a_min=np.min(Sxx),
            a_max=np.inf,
        )

    # --- Aggregated delays ---

    def compute_aggregated_delays(self):
        """Build aggregated delay arrays for HFS and LFS, applying exclusions."""
        for agg_data, side, gd_at_zero in [
            (self.aggregated_hfs, 'HFS', self.hfs_gd_at_zero_fp),
            (self.aggregated_lfs, 'LFS', self.lfs_gd_at_zero_fp),
        ]:
            all_f_probe = np.concatenate((
                [0],
                self.beat_frequencies[side]['K'].f_probe,
                self.beat_frequencies[side]['Ka'].f_probe,
                self.beat_frequencies[side]['Q'].f_probe,
                self.beat_frequencies[side]['V'].f_probe,
            ))
            all_beat_time = np.concatenate((
                [gd_at_zero],
                self.beat_frequencies[side]['K'].y_beat_time,
                self.beat_frequencies[side]['Ka'].y_beat_time,
                self.beat_frequencies[side]['Q'].y_beat_time,
                self.beat_frequencies[side]['V'].y_beat_time,
            ))

            # Remove NaNs
            nan_mask = np.isnan(all_beat_time)
            all_beat_time = all_beat_time[~nan_mask]
            all_f_probe = all_f_probe[~nan_mask]

            # Apply FrequencyExclusions: drop these f_probe points from the final
            # curve. SpectrogramMasks are handled upstream by blanking the
            # spectrogram in compute_beatf, so they are not dropped here.
            for excl in self.frequency_exclusions[side]:
                if not excl.enabled:
                    continue
                mask = (all_f_probe >= excl.low) & (all_f_probe <= excl.high) & (all_f_probe != 0)
                all_f_probe = all_f_probe[~mask]
                all_beat_time = all_beat_time[~mask]

            agg_data.f_probe = all_f_probe
            agg_data.beat_time = all_beat_time

    # --- Profile ---

    def compute_profile(self, coordinate_mode, timestamp):
        """Compute density profile.

        Returns:
            tuple: (r_HFS, ne_HFS, r_LFS, ne_LFS)
        """
        # HFS
        gd_hfs_x = np.linspace(
            self.aggregated_hfs.f_probe[0], self.aggregated_hfs.f_probe[-1],
            PROFILE_INVERSION_RESOLUTION,
        )
        gd_hfs_y = np.interp(gd_hfs_x, self.aggregated_hfs.f_probe, self.aggregated_hfs.beat_time)

        # LFS
        gd_lfs_x = np.linspace(
            self.aggregated_lfs.f_probe[0], self.aggregated_lfs.f_probe[-1],
            PROFILE_INVERSION_RESOLUTION,
        )
        gd_lfs_y = np.interp(gd_lfs_x, self.aggregated_lfs.f_probe, self.aggregated_lfs.beat_time)

        r_HFS = rpspy.profile_inversion(
            gd_hfs_x,
            np.clip(gd_hfs_y - gd_hfs_y[0], a_min=-np.inf, a_max=np.inf),
            pwld_batch=True,
        )
        r_LFS = rpspy.profile_inversion(
            gd_lfs_x,
            np.clip(gd_lfs_y - gd_lfs_y[0], a_min=-np.inf, a_max=np.inf),
            pwld_batch=True,
        )

        r_HFS = r_HFS + self.inner_limiter
        r_LFS = -r_LFS + self.outer_limiter

        ne_HFS = rpspy.f_to_ne(gd_hfs_x)
        ne_LFS = rpspy.f_to_ne(gd_lfs_x)

        if coordinate_mode == 'rho-poloidal':
            r_HFS = r_to_rho(timestamp, r_HFS, self.shot, 'HFS')
            r_LFS = r_to_rho(timestamp, r_LFS, self.shot, 'LFS')

        return r_HFS, ne_HFS, r_LFS, ne_LFS

    # --- Display data (for current spectrogram view) ---

    def compute_current_display(self):
        """Compute dispersion and beat frequency for the current detector view."""
        d = self.detector
        fft = self.current_fft
        sp = self.spect_params[d.side][d.band]

        # Apply background subtraction
        if sp.subtract_background:
            Sxx = self.background_subtract(np.array(fft.unfiltered_Sxx), d.band, d.side)
        else:
            Sxx = np.array(fft.unfiltered_Sxx)
        self.current_fft.Sxx = Sxx

        y_dis = self.compute_dispersion(d.band, d.side, fft.f_probe, fft.t, sp.subtract_dispersion)
        y_beatf = self.compute_beatf(d.band, d.side, np.array(Sxx), y_dis, fft.f_beat, fft.fs, fft.f_probe)

        self.current_display = CurrentDisplayData(
            y_beatf=y_beatf,
            y_dis=y_dis,
            df_dt=(fft.f[-1] - fft.f[0]) / ((len(fft.f) - 1) * (1 / fft.fs)),
        )

    # --- Config serialization ---

    def save_config(self, path):
        """Serialize current params/filters/exclusions to JSON."""
        params_dict = {}
        for side in SIDES:
            params_dict[side] = {}
            for band in BANDS:
                params_dict[side][band] = self.spect_params[side][band].to_config_dict()

        filters_dict = {}
        for side in SIDES:
            filters_dict[side] = {}
            for band in BANDS:
                filters_dict[side][band] = self.filters[side][band].to_config_list()

        exclusions_dict = {}
        for side in SIDES:
            exclusions_dict[side] = [
                excl.to_config_list() for excl in self.frequency_exclusions[side] if excl.enabled
            ]

        masks_dict = {}
        for side in SIDES:
            masks_dict[side] = {}
            for band in BANDS:
                masks_dict[side][band] = [
                    mask.to_config_list() for mask in self.spectrogram_masks[side][band]
                ]


        data = {
            'parameters': params_dict,
            'filters': filters_dict,
            'burst_size': self.detector.burst_size,
            'frequency_exclusions': exclusions_dict,
            'spectrogram_masks': masks_dict,
            'reconstruction_times': {
                'start_time': self.reconstruction_start_time,
                'end_time': self.reconstruction_end_time,
                'time_step': self.reconstruction_time_step,
            },
            'custom_density_cutoff': {
                'custom_density_cutoff': self.custom_density_cutoff,
                'custom_density_cutoff_value': self.custom_density_cutoff_value,
            },
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=4)

        logger.info("Config saved to %s", path)

    def load_config(self, path):
        """Deserialize params/filters/exclusions from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)

        self.detector.burst_size = data.get('burst_size', 1)

        params = data.get('parameters', {})
        for side in SIDES:
            for band in BANDS:
                if side in params and band in params[side]:
                    self.spect_params[side][band] = SpectrogramParams.from_config_dict(params[side][band])
                    # if not in config file use burst size
                    if self.spect_params[side][band].background_burst_size is None:
                        self.spect_params[side][band].background_burst_size = self.detector.burst_size

        filters = data.get('filters', {})
        for side in SIDES:
            for band in BANDS:
                if side in filters and band in filters[side]:
                    self.filters[side][band] = FilterRange.from_config_list(filters[side][band])

        # These two were renamed: 'exclusion_filters' -> 'frequency_exclusions' and
        # 'exclusion_regions' -> 'spectrogram_masks'. Fall back to the old key names
        # so configs written before the rename still load.
        try:
            exclusions = data['frequency_exclusions']
        except KeyError:
            exclusions = data.get('exclusion_filters', {})
        for side in SIDES:
            if side in exclusions:
                self.frequency_exclusions[side] = [
                    FrequencyExclusion.from_config_list(e) for e in exclusions[side]
                ]

        try:
            masks = data['spectrogram_masks']
        except KeyError:
            masks = data.get('exclusion_regions', {})
        for side in SIDES:
            for band in BANDS:
                if side in masks and band in masks[side]:
                    self.spectrogram_masks[side][band] = [
                        SpectrogramMask.from_config_list(m) for m in masks[side][band]
                    ]


        # Older config files have no 'reconstruction_times' key; keep current values then.
        recon_times = data.get('reconstruction_times', {})
        self.reconstruction_start_time = recon_times.get('start_time', self.reconstruction_start_time)
        self.reconstruction_end_time = recon_times.get('end_time', self.reconstruction_end_time)
        self.reconstruction_time_step = recon_times.get('time_step', self.reconstruction_time_step)

        # Older config files have no 'custom_density_cutoff' key; keep current values then.
        custom_density_cutoff = data.get('custom_density_cutoff',  {})
        self.custom_density_cutoff = custom_density_cutoff.get('custom_density_cutoff', self.custom_density_cutoff)
        self.custom_density_cutoff_value = custom_density_cutoff.get('custom_density_cutoff_value', self.custom_density_cutoff_value)