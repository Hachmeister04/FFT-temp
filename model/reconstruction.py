import logging
import rpspy
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

from constants import DEFAULT_LINEARIZATION_SWEEP

logger = logging.getLogger(__name__)


class ReconstructionWorker(QObject):
    """Handles profile reconstruction in a separate thread.

    Receives a plain ReconstructionInput data object, not the view.
    """

    finished_signal = pyqtSignal()
    error_signal = pyqtSignal(str)

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent=None, **kwargs)

    @pyqtSlot(object)
    def reconstruct(self, params):
        """Perform full profile reconstruction.

        Args:
            params: ReconstructionInput dataclass with all needed values.
        """
        try:
            spectrogram_options = {}
            for side in ['HFS', 'LFS']:
                spectrogram_options[side] = {}
                for band in ['K', 'Ka', 'Q', 'V']:
                    sp = params.spect_params[side][band]
                    spectrogram_options[side][band] = {
                        'nperseg': sp.nperseg,
                        'noverlap': sp.noverlap,
                        'nfft': sp.nfft,
                    }

            subtract_background_on_bands = []
            subtract_dispersion_on_bands = []
            for side in ['HFS', 'LFS']:
                for band in ['K', 'Ka', 'Q', 'V']:
                    sp = params.spect_params[side][band]
                    if sp.subtract_background:
                        subtract_background_on_bands.append(f"{band}-{side}")
                    if sp.subtract_dispersion:
                        subtract_dispersion_on_bands.append(f"{band}-{side}")

            background_sweeps = {}
            for side in ['HFS', 'LFS']:
                background_sweeps[side] = {}
                for band in ['K', 'Ka', 'Q', 'V']:
                    background_sweeps[side][band] = params.spect_params[side][band].background_sweep
            
            background_burst_size = {}
            for side in ['HFS', 'LFS']:
                background_burst_size[side] = {}
                for band in ['K', 'Ka', 'Q', 'V']:
                    background_burst_size[side][band] = params.spect_params[side][band].background_burst_size

            # Convert filters from FilterRange objects to the list format rpspy expects
            filters_dict = {}
            for side in ['HFS', 'LFS']:
                filters_dict[side] = {}
                for band in ['K', 'Ka', 'Q', 'V']:
                    filt = params.filters[side][band]
                    filters_dict[side][band] = [filt.low, filt.high]

            # Convert exclusion filters from ExclusionRange objects to list format
            exclusion_dict = {}
            for side in ['HFS', 'LFS']:
                exclusion_dict[side] = [[e.low, e.high] for e in params.exclusion_filters[side] if e.enabled]

            # Convert exclusion regions from ExclusionRegion objects to list format
    #             t_min: float = 0.0
    # t_max: float = 0.0
    # f_prob_min: float = 0.0
    # f_prob_max: float = 0.0
    # f_beat_min: float = 0.0
    # f_beat_max: float = 0.0
    # enabled: bool = True
    # spectrogram_regions : dict, optional
    #     2D spectrogram exclusion regions applied per band/side *before*
    #     peak-finding. Unlike ``exclusion_regions`` (which drops probing-frequency
    #     columns from the merged curve), each region masks a rectangular
    #     (shot time, probing frequency, beat frequency) box of a band's
    #     beat-frequency spectrogram so it is ignored when finding the maximum
    #     beat frequency. Default is None (no regions for any band/side).
    #     Structure mirrors ``filters``; each band holds a list of flat 6-element
    #     regions ``[t_min, t_max, f_prob_min, f_prob_max, f_beat_min, f_beat_max]``,
    #     with shot time in seconds and frequencies in Hz. Example:
    #     {
    #         'HFS': {'K': [], 'Ka': [], 'Q': [[2.0, 3.0, 40e9, 45e9, 1e6, 2e6]], 'V': []},
    #         'LFS': {'K': [], 'Ka': [], 'Q': [], 'V': []},
    #     }
            spectrogram_regions = {'HFS': {}, "LFS": {}}
            for side in ['HFS', 'LFS']:
                for band in ["K", "Ka", "Q", "V"]:
                    spectrogram_regions[side][band] = [
                        [
                            e.t_min, e.t_max, 
                            e.f_prob_min, e.f_prob_max, 
                            e.f_beat_min, e.f_beat_max,
                        ] for e in params.exclusion_regions[side][band] if e.enabled
                    ]

            rpspy.full_profile_reconstruction(
                shot=params.shot,
                shotfile_dir=params.file_path,
                linearization_shotfile_dir=params.file_path,
                sweep_linearization=DEFAULT_LINEARIZATION_SWEEP,
                shot_linearization=params.shot,
                spectrogram_options=spectrogram_options,
                filters=filters_dict,
                exclusion_regions=exclusion_dict,
                spectrogram_regions=spectrogram_regions,
                subtract_background_on_bands=subtract_background_on_bands,
                subtract_dispersion_on_bands=subtract_dispersion_on_bands,
                start_time=params.start_time,
                end_time=params.end_time,
                time_step=params.time_step,
                burst=params.burst_size,
                write_private_shotfile=params.write_private_shotfile,
                write_public_shotfile=params.write_public_shotfile,
                write_hdf5=params.write_hdf5,
                overwrite_existing_hdf5_file=True,
                hdf5_destination_path=params.hdf5_destination_path,
                return_profiles=False,
                initialization_lfs=params.get_init_lfs,
                initialization_hfs=params.get_init_hfs,
                custom_density_cutoff=params.custom_density_cutoff if params.custom_density_cutoff else None,
                density_cutoff_value=params.density_cutoff_value if params.density_cutoff_value else None,
                background_sweeps=background_sweeps,
                background_burst_size=background_burst_size,
            )

            self.finished_signal.emit()
        except Exception as e:
            logger.exception("Reconstruction failed")
            self.error_signal.emit(str(e))
