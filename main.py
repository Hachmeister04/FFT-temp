import sys
import numpy as np
import json
from scipy.signal import spectrogram
from scipy.fft import fftshift
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QSplitter
from PyQt5.QtCore import pyqtSignal, QObject, QThread, pyqtSlot
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
from pyqtgraph.parametertree import Parameter, ParameterTree
import rpspy
import func_aux
import time

#TODO: Correct get_linearization (24 in parameters)

#TODO: Remove hardcoded values and add them here
MAX_BURST_SIZE = 285
DEFAULT_NPERSEG = 256
DEFAULT_NOVERLAP = 220
DEFAULT_NFFT = 512
MIN_NPERSEG = 10
MAX_NFFT = np.inf
DEFAULT_FILTER_LOW = 0 #Hz
DEFAULT_FILTER_HIGH = 10*1e6 #Hz
DEFAULT_START_TIME = 0 #s
DEFAULT_END_TIME = 10 #s
DEFAULT_TIMESTEP = 1e-3 #s

#TODO: Segment the code
class PlotWindow(QMainWindow):
    request_signal = pyqtSignal()
    
    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)

        #Initiate separate thread
        self._thread = QThread()
        self._threaded = Threaded()
        self._threaded.finished_signal.connect(self.finished_reconstruct)
        self.request_signal.connect(self._threaded.reconstruct)
        self._threaded.moveToThread(self._thread)

        qApp = QApplication.instance()
        if qApp is not None:
            qApp.aboutToQuit.connect(self._thread.quit)
        self._thread.start()

        # Define some useful attributes
        self.data = None
        self.burst = None
        self.nperseg = None
        self.noverlap = None
        self.nfft = None
        self.colormap = None
        self.filter_low = None
        self.filter_high = None
        self.params_added = False
        
        #Store the filters
        self.filters = {
            'HFS': {
                'K': [DEFAULT_FILTER_LOW, DEFAULT_FILTER_HIGH],
                'Ka': [DEFAULT_FILTER_LOW, DEFAULT_FILTER_HIGH],
                'Q': [DEFAULT_FILTER_LOW, DEFAULT_FILTER_HIGH],
                'V': [DEFAULT_FILTER_LOW, DEFAULT_FILTER_HIGH]
            },
            'LFS': {
                'K': [DEFAULT_FILTER_LOW, DEFAULT_FILTER_HIGH],
                'Ka': [DEFAULT_FILTER_LOW, DEFAULT_FILTER_HIGH],
                'Q': [DEFAULT_FILTER_LOW, DEFAULT_FILTER_HIGH],
                'V': [DEFAULT_FILTER_LOW, DEFAULT_FILTER_HIGH]
            }
        }

        #---------------------------------------------------------------------------

        # Set the application icon
        self.setWindowIcon(QtGui.QIcon('reflecto-lab.png'))

        # Set up the main window
        self.setWindowTitle('ReflectoLab')
        self.setGeometry(100, 100, 1600, 800)
        
        # Create layouts and widgets------------------------------------------------

        # Create a central widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QHBoxLayout(self.central_widget)
        
        # Create a QSplitter for adjustable layout
        self.splitter = QSplitter(self.central_widget)

        # Create the main graphics layout widget
        self.graph_layout = pg.GraphicsLayoutWidget()
        
        #TODO: Fix the displayed value in the parameters
        # Create the parameter tree
        self.params_file = Parameter.create(name='File', type='group', children=[
            {'name': 'Open', 'type': 'file', 'value': None, 'fileMode': 'Directory'},
            {'name': 'Shot', 'type': 'int'}
        ])
        self.params_detector = Parameter.create(name='Detector', type='group', children=[
            {'name': 'Band', 'type': 'list', 'limits': ['K', 'Ka', 'Q', 'V']},
            {'name': 'Side', 'type': 'list', 'limits': ['HFS', 'LFS']}
        ])
        self.params_sweep = Parameter.create(name='Sweep', type='group', children=[
            {'name': 'Sweep', 'type': 'slider', 'limits': (1, 1)},
            {'name': 'Sweep nº', 'type': 'float', 'value': 1},
            {'name': 'Timestamp', 'type': 'float', 'value': 0, 'suffix': 's', 'siPrefix': True},
        ])
        self.params_fft = Parameter.create(name='Spectrogram', type='group', children=[
            {'name': 'nperseg', 'type': 'float', 'value': DEFAULT_NPERSEG},
            {'name': 'noverlap', 'type': 'float', 'value': DEFAULT_NOVERLAP},
            {'name': 'nfft', 'type': 'float', 'value': DEFAULT_NFFT},
            {'name': 'burst size (odd)', 'type': 'float', 'value': 1, 'limits': (1, MAX_BURST_SIZE)},
            {'name': 'Color Map', 'type': 'cmaplut', 'value': 'plasma'}
        ])
        self.params_filter = Parameter.create(name='Filters (above dispersion)', type='group', children=[
            {'name': 'Low Filter', 'type': 'float', 'value': DEFAULT_FILTER_LOW, 'suffix': 'Hz', 'siPrefix': True},
            {'name': 'High Filter', 'type': 'float', 'value': DEFAULT_FILTER_HIGH, 'suffix': 'Hz', 'siPrefix': True}
        ])
        self.params_profile = Parameter.create(name='Profile', type='group', children=[
            {'name': 'Calculate Profile', 'type': 'action'},
        ])
        self.params_reconstruct = Parameter.create(name='Reconstruct Shot', type='group', children=[
            {'name': 'Start Time', 'type': 'float', 'value': DEFAULT_START_TIME, 'suffix': 's', 'siPrefix': True},
            {'name': 'End Time', 'type': 'float', 'value': DEFAULT_END_TIME, 'suffix': 's', 'siPrefix': True},
            {'name': 'Time Step', 'type': 'float', 'value': DEFAULT_TIMESTEP, 'suffix': 's', 'siPrefix': True},
            {'name': 'Reconstruct Shot', 'type': 'action'}
        ])
        self.param_tree = ParameterTree()
        self.param_tree.addParameters(self.params_file)

        # Create the first and second plots
        self.plot_sweep = self.graph_layout.addPlot(title="Sweep")
        self.plot_profile = self.graph_layout.addPlot(title="Profile")

        self.graph_layout.nextRow()

        # Create the third plot below the others
        self.plot_fft = self.graph_layout.addPlot(title="Spectrogram", colspan=2)

        # Add widgets and layouts---------------------------------------------------

        # Add widgets to the splitter
        self.splitter.addWidget(self.param_tree)
        self.splitter.addWidget(self.graph_layout)
        
        # Add the splitter to the main layout
        self.layout.addWidget(self.splitter)

        #---------------------------------------------------------------------------

        # Connect the file to the function that displays the rest of the parameters and the graphs
        self.params_file.child('Open').sigValueChanged.connect(self.update_shot)
        self.params_file.child('Shot').sigValueChanged.connect(self.update_shot)
    
    def update_shot(self):
        sender = self.sender()

        # Name the path to the directory
        if sender == self.params_file.child('Open'):
            self.file_path = self.params_file.child('Open').value()
            self.shot = func_aux.get_shot_from_path(self.file_path)
            self.params_file.child('Shot').setValue(self.shot, blockSignal=self.update_shot)
        elif sender == self.params_file.child('Shot'):
            self.file_path = func_aux.get_path_from_shot(self.params_file.child('Shot').value())
            self.shot = self.params_file.child('Shot').value()
            self.params_file.child('Open').setValue(self.file_path, blockSignal=self.update_shot)
        
        
        if self.params_added == False:
            self.param_tree.addParameters(self.params_detector)
            self.param_tree.addParameters(self.params_sweep)
            self.param_tree.addParameters(self.params_fft)
            self.param_tree.addParameters(self.params_filter)
            self.param_tree.addParameters(self.params_profile)
            self.param_tree.addParameters(self.params_reconstruct)
            self.params_added = True

        # Connect the parameters to the functions-----------------------------------

        # Connect the lists to update the plot
        self.params_detector.child('Band').sigValueChanged.connect(self.update_plot_params)
        self.params_detector.child('Side').sigValueChanged.connect(self.update_plot_params)
        self.params_detector.child('Side').sigValueChanged.connect(self.draw_profile)

        # Connect the slider, sweep, and timestamp to update the plot
        self.params_sweep.child('Sweep').sigValueChanged.connect(self.update_plot_params)
        self.params_sweep.child('Sweep nº').sigValueChanged.connect(self.update_plot_params)
        self.params_sweep.child('Timestamp').sigValueChanged.connect(self.update_plot_params)

        #Connect the fft params to update the fft
        self.params_fft.child('nperseg').sigValueChanged.connect(self.update_fft_params)
        self.params_fft.child('noverlap').sigValueChanged.connect(self.update_fft_params)
        self.params_fft.child('nfft').sigValueChanged.connect(self.update_fft_params)
        self.params_fft.child('burst size (odd)').sigValueChanged.connect(self.update_fft_params)
        self.params_fft.child('Color Map').sigValueChanged.connect(self.update_fft_params)

        #Connect the filter params to update the fft
        self.params_filter.child('Low Filter').sigValueChanged.connect(self.update_fft_params)
        self.params_filter.child('High Filter').sigValueChanged.connect(self.update_fft_params)

        #Connect the button to update the profile
        self.params_profile.child('Calculate Profile').sigActivated.connect(self.update_profile)

        #Connect the start and end times to eachother
        self.params_reconstruct.child('Start Time').sigValueChanged.connect(self.update_reconstruct_params)
        self.params_reconstruct.child('End Time').sigValueChanged.connect(self.update_reconstruct_params)

        #Connect the button to reconstruct shot
        self.params_reconstruct.child('Reconstruct Shot').sigActivated.connect(self.request_reconstruct)

        #---------------------------------------------------------------------------

        self.update_plot()
        self.update_fft()

        # Set limits to the parameters----------------------------------------------

        self.params_sweep.child('Sweep').setLimits((1, len(rpspy.get_timestamps(self.shot, self.file_path))))
        self.params_sweep.child('Sweep nº').setLimits((1, len(rpspy.get_timestamps(self.shot, self.file_path))))
        self.params_fft.child('nperseg').setLimits((MIN_NPERSEG, len(self.data)))
        self.params_fft.child('noverlap').setLimits((0, self.params_fft.child('nperseg').value() - 1))
        self.params_fft.child('nfft').setLimits((self.params_fft.child('nperseg').value(), MAX_NFFT))
        self.params_filter.child('Low Filter').setLimits((0, np.inf))
        self.params_filter.child('High Filter').setLimits((abs(self.f_beat[0] - self.f_beat[1]), np.inf))
        self.params_reconstruct.child('Start Time').setLimits((0, len(rpspy.get_timestamps(self.shot, self.file_path)) * (rpspy.get_timestamps(self.shot, self.file_path)[1] - rpspy.get_timestamps(self.shot, self.file_path)[0])))
        self.params_reconstruct.child('End Time').setLimits((0, len(rpspy.get_timestamps(self.shot, self.file_path)) * (rpspy.get_timestamps(self.shot, self.file_path)[1] - rpspy.get_timestamps(self.shot, self.file_path)[0])))
        
    def update_plot(self):
        self.band = self.params_detector.child('Band').value()
        self.side = self.params_detector.child('Side').value()
        self.sweep = int(self.params_sweep.child('Sweep nº').value()) - 1

        if self.band == 'V':
            self.signal = 'complex'
        else:
            self.signal = 'real'
        new_data = rpspy.get_band_signal(self.shot, self.file_path, self.band, self.side, self.signal, self.sweep)[0]
        x = func_aux.cached_get_linearization(self.shot, 24, self.band, shotfile_dir=self.file_path)
        x, new_data = rpspy.linearize(x, new_data)
        if not(np.array_equal(self.data, new_data)):
            self.data = new_data
            y_real = np.real(self.data)
            y_complex = np.imag(self.data)
            # Plot the data
            self.plot_sweep.clear()  # Clears the plot
            self.plot_sweep.plot(x, y_real, pen=pg.mkPen(color='r', width=2)) #Plot real part
            if not(np.array_equiv(y_complex, 0)):
                self.plot_sweep.plot(x, y_complex, pen=pg.mkPen(color='b', width=2)) #Plot complex part
            
            self.plot_sweep.setLimits(xMin=x[0],
                                    xMax=x[-1],
                                    maxXRange=x[-1]-x[0],
                                    yMin=0,
                                    yMax=2**12,
                                    maxYRange=2**12)
            self.plot_sweep.setLabel('bottom', 'Probing Frequency', units='Hz')
            print("plot")
    
    def update_fft(self):
        start_time = time.time()

        new_nperseg = int(self.params_fft.child('nperseg').value())
        new_noverlap = int(self.params_fft.child('noverlap').value())
        new_nfft = int(self.params_fft.child('nfft').value())
        new_burst_size = int(self.params_fft.child('burst size (odd)').value())
        new_colormap = self.params_fft.child('Color Map').value()
        new_filter_low = self.params_filter.child('Low Filter').value()
        new_filter_high = self.params_filter.child('High Filter').value()

        new_burst = rpspy.get_band_signal(self.shot, self.file_path, self.band, self.side, self.signal, self.sweep - new_burst_size // 2, new_burst_size)
        x = func_aux.cached_get_linearization(self.shot, 24, self.band, shotfile_dir=self.file_path)
        x, new_burst = rpspy.linearize(x, new_burst)
        if (not(np.array_equal(self.burst, new_burst)) or
            new_nperseg != self.nperseg or
            new_noverlap != self.noverlap or
            new_nfft != self.nfft or
            new_colormap != self.colormap or
            new_filter_low != self.filter_low or
            new_filter_high != self.filter_high):

            self.burst = new_burst
            self.nperseg = new_nperseg
            self.noverlap = new_noverlap
            self.nfft = new_nfft
            self.colormap = new_colormap
            self.filter_low = new_filter_low
            self.filter_high = new_filter_high

            fs = rpspy.get_sampling_frequency(self.shot, self.file_path)  # Sampling frequency

            self.f_beat, t, Sxx = spectrogram(
                self.burst, 
                fs=fs, 
                nperseg=self.nperseg, 
                noverlap=self.noverlap, 
                nfft=self.nfft,
                return_onesided=False if self.burst.dtype == complex else True
                )

            f_probe = np.interp(t, np.arange(len(x))/fs, x)

            if self.burst.dtype == complex:
                self.f_beat = fftshift(self.f_beat)
                Sxx = fftshift(Sxx, axes=-2)

            # Calculate average of the burst
            Sxx = np.average(Sxx, axis=0)

            # Example: Transformed display of ImageItem
            tr = QtGui.QTransform() # prepare ImageItem transformation
            #Translation when x is time
            #alpha_x = (self.nperseg-self.noverlap)/fs
            #tr.translate(self.noverlap/2/fs, -fs/2 if self.burst.dtype == complex else 0)
            #tr.scale(alpha_x, abs(f[1]-f[0])) # scale horizontal and vertical axes
            #Translation when x is frequency
            alpha_x = (self.nperseg-self.noverlap)*abs(x[1]-x[0])
            tr.translate(self.noverlap/2*abs(x[1]-x[0]) + x[0], -fs/2 if self.burst.dtype == complex else 0)
            tr.scale(alpha_x, abs(self.f_beat[1]-self.f_beat[0])) # scale horizontal and vertical axes

            i1 = pg.ImageItem(image=np.log(Sxx).T) # Note: `Sxx` needs to be transposed to fit the display format
            i1.setTransform(tr) # assign transform
            
            self.plot_fft.clear() # Clear previous plot
            self.plot_fft.addItem(i1)
            
            # Set up color bar
            try:
                self.colorBar.setImageItem(i1)
                self.colorBar.setColorMap(self.colormap)
            except AttributeError:
                self.colorBar = self.plot_fft.addColorBar(i1, colorMap=self.colormap, values=(np.min(np.log(Sxx)), np.max(np.log(Sxx))))

            #Generate dispersion line
            k = (f_probe[-1] - f_probe[0]) / (t[-1] - t[0])
            y_dis = k * rpspy.aug_tgcorr2(self.band, self.side, f_probe*1e-9, self.shot)
            self.plot_fft.plot(f_probe, y_dis, pen=pg.mkPen(color='g', width=2))

            #TODO: Don't redraw the fft when redrawing filters
            #Generate low filter line
            y_low = y_dis + self.filter_low
            self.plot_fft.plot(f_probe, y_low, pen=pg.mkPen(color='b', width=2))

            #Generate high filter line
            y_high = y_dis + self.filter_high
            self.plot_fft.plot(f_probe, y_high, pen=pg.mkPen(color='w', width=2))

            # Apply filters to spectogram
            Sxx_copy = np.array(Sxx)
            Sxx_copy[np.broadcast_to(self.f_beat[:, None], Sxx_copy.shape) <= y_dis + self.filter_low] = Sxx_copy.min()
            Sxx_copy[np.broadcast_to(self.f_beat[:, None], Sxx_copy.shape) >= y_dis + self.filter_high] = Sxx_copy.min()
            
            # Generate the line through the max of the graph
            y_max, _ = rpspy.column_wise_max_with_quadratic_interpolation(Sxx_copy)  # Y coordinates
            y_max *= abs(self.f_beat[1]-self.f_beat[0])
            if self.burst.dtype == complex:
                y_max += -fs/2
            self.plot_fft.plot(f_probe, y_max, pen=pg.mkPen(color='r', width=2))

            # Configure plot appearance
            self.plot_fft.setMouseEnabled(x=True, y=True)
            self.plot_fft.setLimits(xMin=f_probe[0]-(f_probe[1]-f_probe[0])/2, 
                                    xMax=f_probe[-1]+(f_probe[1]-f_probe[0])/2, 
                                    maxXRange=f_probe[-1]-f_probe[0]+f_probe[1]-f_probe[0], 
                                    yMin=self.f_beat[0]-(self.f_beat[1]-self.f_beat[0])/2,
                                    yMax=self.f_beat[-1]+(self.f_beat[1]-self.f_beat[0])/2,
                                    maxYRange=self.f_beat[-1]-self.f_beat[0]+self.f_beat[1]-self.f_beat[0])
            self.plot_fft.setLabel('bottom', 'Probing Frequency', units='Hz')
            self.plot_fft.setLabel('left', 'Beat Frequency', units='Hz')
            print("fft")
            print("--- %s seconds ---" % (time.time() - start_time))

    def update_profile(self):
        _, self.density, self.r_hfs, self.r_lfs = func_aux.cached_full_profile_reconstruction(
            shot=self.shot, 
            #destination_dir: str = '.', 
            shotfile_dir=self.file_path, 
            linearization_shotfile_dir=self.file_path, 
            sweep_linearization=None, 
            shot_linearization=self.shot,
            spectrogram_options=json.dumps({
            'K': {'nperseg': self.nperseg, 'noverlap':self.noverlap, 'nfft': self.nfft},
            'Ka': {'nperseg': self.nperseg, 'noverlap':self.noverlap, 'nfft': self.nfft},
            'Q': {'nperseg': self.nperseg, 'noverlap':self.noverlap, 'nfft': self.nfft},
            'V': {'nperseg': self.nperseg, 'noverlap':self.noverlap, 'nfft': self.nfft},
            }), 
            filters=json.dumps(self.filters),
            subtract_on_bands=None,
            start_time = self.params_sweep.child('Timestamp').value(), 
            end_time = self.params_sweep.child('Timestamp').value(), 
            #time_step = 1e-3,
            burst = int(self.params_fft.child('burst size (odd)').value()), 
            write_dump = False, 
            return_profiles = True,
            )
        
        self.draw_profile()

    def draw_profile(self):
        try:
            self.plot_profile.clear()
            if self.side == 'HFS':
                x = self.r_hfs[0]
            else:
                x = self.r_lfs[0]
            self.plot_profile.plot(x, self.density*1e-19, pen=pg.mkPen(color='r', width=2))
            self.plot_profile.setLimits(xMin=min(x), 
                                        xMax=max(x), 
                                        maxXRange=max(x)-min(x), 
                                        yMin=min(self.density*1e-19),
                                        yMax=max(self.density*1e-19),
                                        maxYRange=max(self.density*1e-19)-min(self.density*1e-19))
            self.plot_profile.setLabel('bottom', 'radius', units='m')
            self.plot_profile.setLabel('left', 'density', units='1e19 m^-3')
        except AttributeError:
            pass

    def update_plot_params(self):
        sender = self.sender()

        if sender == self.params_sweep.child('Sweep'):
            value = self.params_sweep.child('Sweep').value()
            timestamp = rpspy.get_timestamps(self.shot, self.file_path)[value - 1]
            self.params_sweep.child('Sweep nº').setValue(value, blockSignal=self.update_plot_params)
            self.params_sweep.child('Timestamp').setValue(timestamp, blockSignal=self.update_plot_params)

        elif sender == self.params_sweep.child('Sweep nº'):
            value = int(self.params_sweep.child('Sweep nº').value())
            timestamp = rpspy.get_timestamps(self.shot, self.file_path)[value - 1]
            self.params_sweep.child('Sweep nº').setValue(value, blockSignal=self.update_plot_params)
            self.params_sweep.child('Sweep').setValue(value, blockSignal=self.update_plot_params)
            self.params_sweep.child('Timestamp').setValue(timestamp, blockSignal=self.update_plot_params)
            
        elif sender == self.params_sweep.child('Timestamp'):
            value = self.params_sweep.child('Timestamp').value()
            timestamp = func_aux.round_to_nearest(value, rpspy.get_timestamps(self.shot, self.file_path))
            index = np.where(rpspy.get_timestamps(self.shot, self.file_path) == timestamp)
            self.params_sweep.child('Timestamp').setValue(timestamp, blockSignal=self.update_plot_params)
            self.params_sweep.child('Sweep').setValue(index[0][0] + 1, blockSignal=self.update_plot_params)
            self.params_sweep.child('Sweep nº').setValue(index[0][0] + 1, blockSignal=self.update_plot_params)
        
        elif sender == self.params_detector.child('Band') or sender == self.params_detector.child('Side'):
            self.params_filter.child('Low Filter').setValue(self.filters[self.params_detector.child('Side').value()][self.params_detector.child('Band').value()][0], blockSignal=self.update_fft_params)
            self.params_filter.child('High Filter').setValue(self.filters[self.params_detector.child('Side').value()][self.params_detector.child('Band').value()][1], blockSignal=self.update_fft_params)

        self.update_plot()
        self.update_fft()

    def update_fft_params(self):
        sender = self.sender()

        if sender == self.params_fft.child('burst size (odd)'):
            value = int(self.params_fft.child('burst size (odd)').value())
            if value % 2 == 0:
                self.params_fft.child('burst size (odd)').setValue(value - 1, blockSignal=self.update_fft_params)
            else:
                self.params_fft.child('burst size (odd)').setValue(value, blockSignal=self.update_fft_params)
            lower_limit = int(1 + self.params_fft.child('burst size (odd)').value() // 2)
            upper_limit = int(len(rpspy.get_timestamps(self.shot, self.file_path)) - self.params_fft.child('burst size (odd)').value() // 2)
            self.params_sweep.child('Sweep').setLimits([lower_limit, upper_limit])
            self.params_sweep.child('Sweep nº').setLimits([lower_limit, upper_limit])
            self.params_sweep.child('Timestamp').setLimits([rpspy.get_timestamps(self.shot, self.file_path)[lower_limit - 1], rpspy.get_timestamps(self.shot, self.file_path)[upper_limit - 1]])

        elif sender == self.params_fft.child('nperseg'):
            value = int(self.params_fft.child('nperseg').value())
            self.params_fft.child('nperseg').setValue(value, blockSignal=self.update_fft_params)
            self.params_fft.child('noverlap').setLimits((0, self.params_fft.child('nperseg').value() - 1))
            self.params_fft.child('nfft').setLimits((self.params_fft.child('nperseg').value(), np.inf))
        
        elif sender == self.params_fft.child('noverlap'):
            value = int(self.params_fft.child('noverlap').value())
            self.params_fft.child('noverlap').setValue(value, blockSignal=self.update_fft_params)
        
        elif sender == self.params_fft.child('nfft'):
            value = int(self.params_fft.child('nfft').value())
            self.params_fft.child('nfft').setValue(value, blockSignal=self.update_fft_params)
        
        elif sender == self.params_filter.child('Low Filter'):
            self.filters[self.side][self.band][0] = self.params_filter.child('Low Filter').value()
            if self.filters[self.side][self.band][0] + abs(self.f_beat[1] - self.f_beat[0]) >= self.filters[self.side][self.band][1]:
                self.filters[self.side][self.band][1] = self.filters[self.side][self.band][0] + abs(self.f_beat[1] - self.f_beat[0])
                self.params_filter.child('High Filter').setValue(self.filters[self.side][self.band][1], blockSignal=self.update_fft_params)
        
        elif sender == self.params_filter.child('High Filter'):
            self.filters[self.side][self.band][1] = self.params_filter.child('High Filter').value()
            if self.filters[self.side][self.band][1] - abs(self.f_beat[1] - self.f_beat[0]) <= self.filters[self.side][self.band][0]:
                self.filters[self.side][self.band][0] = self.filters[self.side][self.band][1] - abs(self.f_beat[1] - self.f_beat[0])
                self.params_filter.child('Low Filter').setValue(self.filters[self.side][self.band][0], blockSignal=self.update_fft_params)
        
        self.update_fft()
    
    def update_reconstruct_params(self):
        sender = self.sender()

        if sender == self.params_reconstruct.child('Start Time'):
            if self.params_reconstruct.child('Start Time').value() > self.params_reconstruct.child('End Time').value():
                self.params_reconstruct.child('End Time').setValue(self.params_reconstruct.child('Start Time').value(), blockSignal=self.update_reconstruct_params)

        elif sender == self.params_reconstruct.child('End Time'):
            if self.params_reconstruct.child('End Time').value() < self.params_reconstruct.child('Start Time').value():
                self.params_reconstruct.child('Start Time').setValue(self.params_reconstruct.child('End Time').value(), blockSignal=self.update_reconstruct_params)

    @pyqtSlot()
    def request_reconstruct(self):
        self.request_signal.emit()
        self.params_reconstruct.child('Start Time').setOpts(enabled=False)
        self.params_reconstruct.child('End Time').setOpts(enabled=False)
        self.params_reconstruct.child('Time Step').setOpts(enabled=False)
        self.params_reconstruct.child('Reconstruct Shot').setOpts(enabled=False)
    
    @pyqtSlot()
    def finished_reconstruct(self):
        self.params_reconstruct.child('Start Time').setOpts(enabled=True)
        self.params_reconstruct.child('End Time').setOpts(enabled=True)
        self.params_reconstruct.child('Time Step').setOpts(enabled=True)
        self.params_reconstruct.child('Reconstruct Shot').setOpts(enabled=True)


class Threaded(QObject):
    finished_signal = pyqtSignal()

    def __init__(self, parent=None, **kwargs):
        # intentionally not setting the parent
        super().__init__(parent=None, **kwargs)

    @pyqtSlot()
    def reconstruct(self):
        rpspy.full_profile_reconstruction(
            shot=main_window.shot, 
            destination_dir = 'reconstruction_shots', 
            shotfile_dir=main_window.file_path, 
            linearization_shotfile_dir=main_window.file_path, 
            sweep_linearization=None, 
            shot_linearization=main_window.shot,
            spectrogram_options={
            'K': {'nperseg': main_window.nperseg, 'noverlap':main_window.noverlap, 'nfft': main_window.nfft},
            'Ka': {'nperseg': main_window.nperseg, 'noverlap':main_window.noverlap, 'nfft': main_window.nfft},
            'Q': {'nperseg': main_window.nperseg, 'noverlap':main_window.noverlap, 'nfft': main_window.nfft},
            'V': {'nperseg': main_window.nperseg, 'noverlap':main_window.noverlap, 'nfft': main_window.nfft},
            }, 
            filters=main_window.filters,
            subtract_on_bands=None,
            start_time = main_window.params_reconstruct.child('Start Time').value(), 
            end_time = main_window.params_reconstruct.child('End Time').value(), 
            time_step = main_window.params_reconstruct.child('Time Step').value(),
            burst = int(main_window.params_fft.child('burst size (odd)').value()), 
            write_dump = True,
            return_profiles = False,
            )
        self.finished_signal.emit()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = PlotWindow()
    main_window.show()
    sys.exit(app.exec())