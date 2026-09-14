from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from constants import (
    DEFAULT_NPERSEG, DEFAULT_NOVERLAP, DEFAULT_NFFT,
    DEFAULT_FILTER_LOW, DEFAULT_FILTER_HIGH, DEFAULT_BURST_SIZE,
    DEFAULT_START_TIME, DEFAULT_END_TIME, DEFAULT_TIMESTEP,
    DEFAULT_DENSITY_CUTOFF,
)


@dataclass
class SpectrogramParams:
    nperseg: int = DEFAULT_NPERSEG
    noverlap: int = DEFAULT_NOVERLAP
    nfft: int = DEFAULT_NFFT
    subtract_background: bool = False
    subtract_dispersion: Optional[bool] = None
    background_sweep: int = 0
    background_burst_size: int = DEFAULT_BURST_SIZE

    def to_config_dict(self):
        return {
            'nperseg': self.nperseg,
            'noverlap': self.noverlap,
            'nfft': self.nfft,
            'subtract background': self.subtract_background,
            'subtract dispersion': self.subtract_dispersion,
            'background sweep': self.background_sweep,
            'background burst size (odd)': self.background_burst_size
        }

    @classmethod
    def from_config_dict(cls, d):
        return cls(
            nperseg=d['nperseg'],
            noverlap=d['noverlap'],
            nfft=d['nfft'],
            subtract_background=d['subtract background'],
            subtract_dispersion=d['subtract dispersion'],
            background_sweep=d.get('background sweep', 0),
            background_burst_size=d.get('background burst size (odd)', None)
        )


@dataclass
class FilterRange:
    low: float = DEFAULT_FILTER_LOW
    high: float = DEFAULT_FILTER_HIGH

    def to_config_list(self):
        return [self.low, self.high]

    @classmethod
    def from_config_list(cls, lst):
        return cls(low=lst[0], high=lst[1])


@dataclass
class ExclusionRange:
    low: float = 0.0
    high: float = 0.0
    enabled: bool = True

    def to_config_list(self):
        return [self.low, self.high]

    @classmethod
    def from_config_list(cls, lst):
        # Loaded exclusions are always enabled.
        return cls(low=lst[0], high=lst[1], enabled=True)


@dataclass
class ExclusionRegion:
    """A 2D region of the beat-frequency spectrogram to ignore during peak-finding.

    The (t_min, t_max) range is shot/discharge time in seconds and gates which
    sweeps the region applies to. (f_prob_min, f_prob_max) is the probing-frequency
    extent (spectrogram x-axis) and (f_beat_min, f_beat_max) the beat-frequency
    extent (spectrogram y-axis).
    """
    t_min: float = 0.0
    t_max: float = 0.0
    f_prob_min: float = 0.0
    f_prob_max: float = 0.0
    f_beat_min: float = 0.0
    f_beat_max: float = 0.0
    enabled: bool = True

    def to_config_list(self):
        return [
            self.t_min, self.t_max,
            self.f_prob_min, self.f_prob_max,
            self.f_beat_min, self.f_beat_max,
            self.enabled,
        ]

    @classmethod
    def from_config_list(cls, lst):
        return cls(
            t_min=lst[0], t_max=lst[1],
            f_prob_min=lst[2], f_prob_max=lst[3],
            f_beat_min=lst[4], f_beat_max=lst[5],
            enabled=lst[6] if len(lst) > 6 else True,
        )


@dataclass
class InitFileData:
    name: str = ''
    time: Optional[np.ndarray] = field(default=None, repr=False)
    position: Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class InitValues:
    type: str = 'Default (recommended)'
    custom_value: float = 0.0
    file: InitFileData = field(default_factory=InitFileData)
    current_value: float = 0.0


@dataclass
class BeatFrequencyData:
    f_probe: Optional[np.ndarray] = field(default=None, repr=False)
    y_beatf: Optional[np.ndarray] = field(default=None, repr=False)
    y_beat_time: Optional[np.ndarray] = field(default=None, repr=False)
    df_dt: Optional[float] = None


@dataclass
class CurrentSweepData:
    data: Optional[np.ndarray] = field(default=None, repr=False)
    x_data: Optional[np.ndarray] = field(default=None, repr=False)
    signal_type: str = 'real'


@dataclass
class CurrentFFTData:
    f: Optional[np.ndarray] = field(default=None, repr=False)
    fs: Optional[float] = None
    f_beat: Optional[np.ndarray] = field(default=None, repr=False)
    t: Optional[np.ndarray] = field(default=None, repr=False)
    f_probe: Optional[np.ndarray] = field(default=None, repr=False)
    unfiltered_Sxx: Optional[np.ndarray] = field(default=None, repr=False)
    Sxx: Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class CurrentDisplayData:
    y_beatf: Optional[np.ndarray] = field(default=None, repr=False)
    y_dis: Optional[np.ndarray] = field(default=None, repr=False)
    df_dt: Optional[float] = None


@dataclass
class AggregatedDelayData:
    f_probe: Optional[np.ndarray] = field(default=None, repr=False)
    beat_time: Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class DetectorSelection:
    band: str = 'K'
    side: str = 'HFS'
    sweep: int = 0
    burst_size: int = DEFAULT_BURST_SIZE


@dataclass
class ReconstructionInput:
    shot: int = 0
    file_path: str = ''
    spect_params: dict = field(default_factory=dict)
    filters: dict = field(default_factory=dict)
    exclusion_filters: dict = field(default_factory=dict)
    exclusion_regions: dict = field(default_factory=dict)
    burst_size: int = DEFAULT_BURST_SIZE
    background_burst_size: int = DEFAULT_BURST_SIZE
    start_time: float = DEFAULT_START_TIME
    end_time: float = DEFAULT_END_TIME
    time_step: float = DEFAULT_TIMESTEP
    custom_density_cutoff: bool = False
    density_cutoff_value: float = DEFAULT_DENSITY_CUTOFF
    write_private_shotfile: bool = False
    write_public_shotfile: bool = False
    write_hdf5: bool = True
    hdf5_destination_path: Optional[str] = None
    get_init_hfs: object = None
    get_init_lfs: object = None
