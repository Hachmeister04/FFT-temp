from pyqtgraph.parametertree import Parameter
import getpass
from constants import (
    DEFAULT_NPERSEG, DEFAULT_NOVERLAP, DEFAULT_NFFT, DEFAULT_BURST_SIZE, DEFAULT_POSTFIX_CONFIG,
    MAX_BURST_SIZE,
    DEFAULT_FILTER_LOW, DEFAULT_FILTER_HIGH,
    DEFAULT_START_TIME, DEFAULT_END_TIME, DEFAULT_TIMESTEP, DECIMALS_TIMES, DEFAULT_DENSITY_CUTOFF,
    DECIMALS_SWEEP_NUM, DECIMALS_TIMESTAMP,
    DECIMALS_NPERSEG, DECIMALS_NOVERLAP, DECIMALS_NFFT,
    DECIMALS_EXCLUSIONS, DECIMALS_INIT, DEFAULT_FOLDER_CONFIG
)


class ParameterPanels:
    """Holds all Parameter groups for the application."""

    def __init__(self):
        self.file = Parameter.create(name='File', type='group', children=[
            {'name': 'Open', 'type': 'file', 'value': None, 'fileMode': 'Directory'},
            {'name': 'Shot', 'type': 'int'},
        ])
        self.file.child('Open').setValue('')

        self.init = Parameter.create(name='Initialization', type='group', visible=False, children=[
            {'name': 'Type', 'type': 'list', 'limits': ['Default (recommended)', 'Custom', 'From file'], 'delay': 0},
            {'name': 'File', 'type': 'file', 'value': None, 'fileMode': 'AnyFile', 'acceptMode': 'AcceptOpen', 'nameFilter': 'CSV Files (*.csv)', 'visible': False, 'delay': 0},
            {'name': 'Value', 'type': 'float', 'value': 0, 'suffix': 'm', 'siPrefix': True, 'decimals': DECIMALS_INIT, 'readonly': True, 'delay': 0},
        ])
        self.init.child('File').setValue('')

        self.config = Parameter.create(name='Configuration', type='group', visible=False, children=[
            {'name': 'Save', 'type': 'file', 'value': None, 'fileMode': 'AnyFile', 'acceptMode': 'AcceptSave', 'nameFilter': 'JSON Files (*.json)', 'directory': DEFAULT_FOLDER_CONFIG, 'selectFile': f"test22.json"},
            {'name': 'Load', 'type': 'file', 'value': None, 'fileMode': 'AnyFile', 'acceptMode': 'AcceptOpen', 'nameFilter': 'JSON Files (*.json)', 'directory': DEFAULT_FOLDER_CONFIG},
        ])
        self.config.child('Save').setValue('')
        self.config.child('Load').setValue('')

        self.detector = Parameter.create(name='Detector', type='group', visible=False, children=[
            {'name': 'Band', 'type': 'list', 'limits': ['K', 'Ka', 'Q', 'V']},
            {'name': 'Side', 'type': 'list', 'limits': ['HFS', 'LFS']},
        ])

        self.sweep = Parameter.create(name='Sweep', type='group', visible=False, children=[
            {'name': 'Sweep nº', 'type': 'float', 'value': 0, 'decimals': DECIMALS_SWEEP_NUM, 'delay': 0},
            {'name': 'Sweep', 'title': ' ', 'type': 'slider', 'limits': (0, 0)},
            {'name': 'Timestamp', 'type': 'float', 'value': 0, 'suffix': 's', 'decimals': DECIMALS_TIMESTAMP, 'siPrefix': True, 'delay': 0},
        ])

        self.fft = Parameter.create(name='Spectrogram', type='group', visible=False, children=[
            {'name': 'nperseg', 'type': 'float', 'value': DEFAULT_NPERSEG, 'decimals': DECIMALS_NPERSEG, 'delay': 0},
            {'name': 'noverlap', 'type': 'float', 'value': DEFAULT_NOVERLAP, 'decimals': DECIMALS_NOVERLAP, 'delay': 0},
            {'name': 'nfft', 'type': 'float', 'value': DEFAULT_NFFT, 'decimals': DECIMALS_NFFT, 'delay': 0},
            {'name': 'burst size (odd)', 'type': 'float', 'value': DEFAULT_BURST_SIZE, 'limits': (1, MAX_BURST_SIZE), 'step': 2, 'delay': 0},
            {'name': 'Scale', 'type': 'checklist', 'limits': ['Normalized', 'Linear', 'Logarithmic'], 'exclusive': True, 'delay': 0},
            {'name': 'Subtract background', 'type': 'bool', 'value': False, 'delay': 0},
            {'name': 'Fix background sweep and burst size', 'type': 'bool', 'value': True, 'delay': 0},
            {'name': 'Background sweep', 'type': 'float', 'value': 0, 'decimals': DECIMALS_SWEEP_NUM, 'delay': 0},
            {'name': 'Background burst size (odd)', 'type': 'float', 'value': DEFAULT_BURST_SIZE, 'limits': (1, MAX_BURST_SIZE), 'step': 2, 'delay': 0},
            {'name': 'Subtract dispersion', 'type': 'bool', 'value': False, 'delay': 0},
            {'name': 'Color Map', 'type': 'cmaplut', 'value': 'plasma'},
            {'name': 'Filters', 'title': 'Filters (above dispersion)', 'type': 'group', 'children': [
                {'name': 'Low Filter', 'type': 'float', 'value': DEFAULT_FILTER_LOW, 'suffix': 'Hz', 'siPrefix': True, 'delay': 0},
                {'name': 'High Filter', 'type': 'float', 'value': DEFAULT_FILTER_HIGH, 'suffix': 'Hz', 'siPrefix': True, 'delay': 0},
            ]},
            {'name': 'Exclude frequencies', 'type': 'group', 'addText': 'Add'},
            {'name': 'Spectrogram masks', 'type': 'group', 'addText': 'Add'},
        ])

        self.profiles = Parameter.create(name='Profiles', type='group', visible=False, children=[
            {'name': 'Coordinates', 'type': 'checklist', 'limits': ['R (m)', 'rho-poloidal'], 'exclusive': True, 'delay': 0},
        ])

        self.reconstruct = Parameter.create(name='Reconstruct Shot', type='group', visible=False, children=[
            {'name': 'Start Time', 'type': 'float', 'value': DEFAULT_START_TIME, 'suffix': 's', 'siPrefix': True, 'decimals': DECIMALS_TIMES, 'delay': 0},
            {'name': 'End Time', 'type': 'float', 'value': DEFAULT_END_TIME, 'suffix': 's', 'siPrefix': True, 'decimals': DECIMALS_TIMES, 'delay': 0},
            {'name': 'Time Step', 'type': 'float', 'value': DEFAULT_TIMESTEP, 'suffix': 's', 'siPrefix': True, 'delay': 0},
            {'name': 'Apply Custom Density Cutoff', 'type': 'bool', 'value': False, 'delay': 0},
            {'name': 'Density Cutoff', 'type': 'float', 'value': DEFAULT_DENSITY_CUTOFF, 'suffix': 'm^-3', 'siPrefix': False, 'delay': 0},
            {'name': 'Reconstruct Shot', 'type': 'action'},
            {'name': 'Reconstruction Output', 'title': 'Reconstruction Output', 'type': 'group', 'children': [
                {'name': 'Private Shotfile', 'type': 'bool', 'value': False, 'delay': 0},
                {'name': 'Public Shotfile', 'type': 'bool', 'value': False, 'delay': 0},
                {'name': 'HDF5', 'type': 'bool', 'value': True, 'delay': 0},
            ]},
        ])

    def all_groups(self):
        return [self.file, self.init, self.config, self.detector,
                self.sweep, self.fft, self.profiles, self.reconstruct]

    def hideable_groups(self):
        return [self.init, self.config, self.detector, self.sweep,
                self.fft, self.profiles, self.reconstruct]
