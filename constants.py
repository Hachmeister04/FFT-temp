import numpy as np
import getpass
# Detector configurations
BANDS = ('K', 'Ka', 'Q', 'V')
SIDES = ('HFS', 'LFS')

# Window
WINDOW_SIZE = (1600, 800)
PARAMETER_TREE_WIDTH_PROPORTION = 0.3
GRAPH_WIDTH_PROPORTION = 0.35

# Parameter Tree
DEFAULT_SECTION_SIZE = 200

# Spectrogram Params
MAX_BURST_SIZE = 285
DEFAULT_NPERSEG = 256
DEFAULT_NOVERLAP = 220
DEFAULT_NFFT = 512
MIN_NPERSEG = 10
MAX_NFFT = np.inf
DEFAULT_BURST_SIZE = 1

# Linearization Params
DEFAULT_LINEARIZATION_SWEEP = int(2 / 35e-6)

# Filter Params
DEFAULT_FILTER_LOW = 0  # Hz
DEFAULT_FILTER_HIGH = 10 * 1e6  # Hz

# Frequency exclusions (1D probing-frequency intervals dropped from the profile).
# Capped at 50 for shotfile compatibility.
FREQUENCY_EXCLUSIONS_MAX_N = 50

# Spectrogram masks (2D boxes blanked before peak-finding).
SPECTROGRAM_MASKS_MAX_N = 10

# Reconstruct Params
DEFAULT_START_TIME = 0  # s
DEFAULT_END_TIME = 10  # s
DEFAULT_TIMESTEP = 1e-3  # s
DECIMALS_TIMES = 3
DEFAULT_DENSITY_CUTOFF = 6e19  # m^-3

# Plot Params
DECIMALS_SWEEP_NUM = 6
DECIMALS_TIMESTAMP = 8
DECIMALS_NPERSEG = 6
DECIMALS_NOVERLAP = 6
DECIMALS_NFFT = 6
DECIMALS_EXCLUSIONS = 6
DECIMALS_INIT = 6

# Profile Properties
PROFILE_INVERSION_RESOLUTION = 150  # points

# Profile Colors
HFS_COLOR = 'r'
HFS_EXCLUSION_COLOR = (250, 160, 160)
LFS_COLOR = 'b'
LFS_EXCLUSION_COLOR = (137, 207, 240)

# Folders
if getpass.getuser().lower()=='vamar':
    DEFAULT_FOLDER_CONFIG = f"/shares/departments/AUG/users/{getpass.getuser().lower()}/configs_reflecto-lab/"
    DEFAULT_PREFIX_CONFIG = f"configurations_"
    DEFAULT_POSTFIX_CONFIG = f".json"

    DEFAULT_FOLDER_HDF5 = f"/shares/departments/AUG/users/{getpass.getuser().lower()}/reconstruction_shots/"
    DEFAULT_PREFIX_HDF5 = "RPS_"
    DEFAULT_POSTFIX_HDF5 = ".h5"
else:
    DEFAULT_FOLDER_CONFIG = f"/shares/departments/AUG/users/{getpass.getuser().lower()}/"
    DEFAULT_PREFIX_CONFIG = f"configuraions_"
    DEFAULT_POSTFIX_CONFIG = f".json"

    DEFAULT_FOLDER_HDF5 = f"/shares/departments/AUG/users/{getpass.getuser().lower()}/"
    DEFAULT_PREFIX_HDF5 = "RPS_"
    DEFAULT_POSTFIX_HDF5 = ".h5"