import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from constants import (
    BANDS, SIDES,
    HFS_COLOR, HFS_EXCLUSION_COLOR, LFS_COLOR, LFS_EXCLUSION_COLOR,
)

# Max exclusion filters per side (matches UI limit in app_controller)
_MAX_EXCL = 10


class PlotRenderer:
    """Rendering methods with persistent plot curves for fast updates.

    Group delay and profile plots use pre-allocated PlotDataItem curves
    that are updated via setData() instead of clear() + plot(). This avoids
    expensive Qt scene graph item creation/destruction on every slider tick.
    """

    def __init__(self):
        # Group delay persistent curves (lazy-initialized on first draw)
        self._gd_agg_hfs = None
        self._gd_agg_lfs = None
        self._gd_band = {}      # (side, band) -> PlotDataItem
        self._gd_excl = {}      # (side, band, idx) -> PlotDataItem
        self._gd_ready = False

        # Profile persistent curves (lazy-initialized on first draw)
        self._prof_hfs = None
        self._prof_lfs = None
        self._prof_ready = False

        # Spectrogram persistent items (lazy-initialized on first draw).
        # Updated via setImage()/setData()/setRect() instead of clear()+recreate
        # so dragging the sweep slider doesn't churn the Qt scene graph.
        self._spec_img = None          # ImageItem
        self._spec_disp = None         # dispersion line
        self._spec_filt_low = None     # low filter line
        self._spec_filt_high = None    # high filter line
        self._spec_beatf = None        # beat-frequency curve
        self._spec_excl = []           # pool of frequency-exclusion overlay curves
        self._spec_mask_rects = []     # pool of QGraphicsRectItem for shaded spectrogram masks

    @staticmethod
    def draw_sweep(plot_widget, x_data, data, signal_type):
        """Draw linearized sweep in the Sweep plot."""
        y_real = np.real(data)
        y_imag = np.imag(data)

        plot_widget.clear()
        plot_widget.plot(x_data, y_real, pen=pg.mkPen(color='r', width=2))

        if not np.array_equiv(y_imag, 0):
            plot_widget.plot(x_data, y_imag, pen=pg.mkPen(color='b', width=2))

        plot_widget.setLimits(
            xMin=x_data[0], xMax=x_data[-1],
            yMin=-2**11, yMax=2**11,
        )
        plot_widget.setRange(
            xRange=(x_data[0], x_data[-1]),
            yRange=(-2**11, 2**11),
        )
        plot_widget.setLabel('bottom', 'Probing Frequency', units='Hz')

    def draw_spectrogram(self, plot_widget, Sxx, scale, colormap, nperseg, noverlap,
                         f, f_beat, f_probe, fs, band, existing_colorbar):
        """Draw spectrogram image. Returns updated colorBar reference.

        Reuses a persistent ImageItem (created on first draw) and updates it via
        setImage()/setTransform() rather than clear()+recreate.
        """
        # Scale the spectrogram
        if scale == 'Normalized':
            Sxx_display = np.array(Sxx)
            max_vals = np.max(Sxx_display, axis=0)
            max_vals[max_vals == 0] = 1
            Sxx_display = Sxx_display / max_vals
        elif scale == 'Linear':
            Sxx_display = np.array(Sxx)
        elif scale == 'Logarithmic':
            Sxx_display = np.log(np.array(Sxx))
        else:
            Sxx_display = np.array(Sxx)

        # Prepare ImageItem transformation
        f_probe_step = abs(f_probe[1] - f_probe[0])
        f_beat_step = abs(f_beat[1] - f_beat[0])

        left_frequency_limit = f_probe[0] - f_probe_step / 2
        right_frequency_limit = f_probe[-1] + f_probe_step / 2

        lower_frequency_limit = f_beat[0] - f_beat_step / 2
        upper_frequency_limit = f_beat[-1] + f_beat_step / 2

        transform = QtGui.QTransform()
        alpha_x = (right_frequency_limit - left_frequency_limit) / len(f_probe)
        alpha_y = (upper_frequency_limit - lower_frequency_limit) / len(f_beat)
        transform.translate(
            left_frequency_limit,
            lower_frequency_limit,
        )
        transform.scale(alpha_x, alpha_y)

        # Reuse the persistent ImageItem (added once, at the bottom of the z-stack)
        if self._spec_img is None:
            self._spec_img = pg.ImageItem()
            plot_widget.addItem(self._spec_img)
        self._spec_img.setImage(Sxx_display.T, autoLevels=False)
        self._spec_img.setTransform(transform)

        # Color bar
        levels = (np.min(Sxx_display), np.max(Sxx_display))
        try:
            existing_colorbar.setImageItem(self._spec_img)
            existing_colorbar.setColorMap(colormap)
            existing_colorbar.setLevels(values=levels)
            colorbar = existing_colorbar
        except (AttributeError, TypeError):
            colorbar = plot_widget.addColorBar(
                self._spec_img, colorMap=colormap, values=levels,
            )

        # Configure plot appearance
        plot_widget.setMouseEnabled(x=True, y=True)
        plot_widget.setLimits(
            xMin=f_probe[0] - (f_probe[1] - f_probe[0]) / 2,
            xMax=f_probe[-1] + (f_probe[1] - f_probe[0]) / 2,
            yMin=f_beat[0] - (f_beat[1] - f_beat[0]) / 2,
            yMax=f_beat[-1] + (f_beat[1] - f_beat[0]) / 2,
        )
        plot_widget.setLabel('bottom', 'Probing Frequency', units='Hz')
        plot_widget.setLabel('left', 'Beat Frequency', units='Hz')

        return colorbar

    def draw_dispersion_line(self, plot_widget, f_probe, y_dis):
        """Overlay dispersion curve on spectrogram."""
        if self._spec_disp is None:
            self._spec_disp = plot_widget.plot(pen=pg.mkPen(color='g', width=2))
        self._spec_disp.setData(f_probe, y_dis)

    def draw_filter_lines(self, plot_widget, f_probe, y_dis, filter_low, filter_high):
        """Draw low and high filter lines on spectrogram."""
        if self._spec_filt_low is None:
            self._spec_filt_low = plot_widget.plot(pen=pg.mkPen(color='b', width=2))
            self._spec_filt_high = plot_widget.plot(pen=pg.mkPen(color='w', width=2))
        self._spec_filt_low.setData(f_probe, y_dis + filter_low)
        self._spec_filt_high.setData(f_probe, y_dis + filter_high)

    def draw_beatf_on_spectrogram(self, plot_widget, f_probe, y_beatf, frequency_exclusions, side):
        """Draw beat frequency curve + frequency-exclusion overlays on spectrogram."""
        if self._spec_beatf is None:
            self._spec_beatf = plot_widget.plot(pen=pg.mkPen(color='r', width=2))
            self._spec_excl = [
                plot_widget.plot(pen=pg.mkPen(color='w', width=2))
                for _ in range(_MAX_EXCL)
            ]
        self._spec_beatf.setData(f_probe, y_beatf)

        for i, curve in enumerate(self._spec_excl):
            if i < len(frequency_exclusions) and frequency_exclusions[i].enabled:
                excl = frequency_exclusions[i]
                mask = (f_probe >= excl.low) & (f_probe <= excl.high)
                curve.setData(f_probe[mask], y_beatf[mask])
            else:
                curve.setData([], [])

    def draw_spectrogram_masks(self, plot_widget, masks, timestamp, f_probe, f_beat):
        """Shade the 2D spectrogram masks that are active for the current sweep.

        Only masks that are enabled and whose time gate [t_min, t_max] contains
        the current sweep ``timestamp`` are drawn (matching what compute_beatf
        actually blanks). Rectangles are pooled and reused via setRect()/setVisible()
        so repeated redraws don't churn the scene graph.
        """
        x_lo, x_hi = f_probe.min(), f_probe.max()
        y_lo, y_hi = f_beat.min(), f_beat.max()

        # Build the list of visible boxes for currently-active masks.
        boxes = []
        for mask in masks:
            if not mask.enabled or not (mask.t_min <= timestamp <= mask.t_max):
                continue
            # Clamp to the visible spectrogram extents.
            x0 = max(mask.f_prob_min, x_lo)
            x1 = min(mask.f_prob_max, x_hi)
            y0 = max(mask.f_beat_min, y_lo)
            y1 = min(mask.f_beat_max, y_hi)
            if x1 <= x0 or y1 <= y0:
                continue
            boxes.append((x0, y0, x1 - x0, y1 - y0))

        # Grow the persistent rect pool to cover the active boxes.
        while len(self._spec_mask_rects) < len(boxes):
            rect = QtWidgets.QGraphicsRectItem()
            rect.setBrush(pg.mkBrush(128, 128, 128, 128))   # neutral gray, alpha 0.5
            rect.setPen(pg.mkPen('w', width=1, style=QtCore.Qt.DashLine))
            plot_widget.addItem(rect)
            self._spec_mask_rects.append(rect)

        # Position the needed rects, hide the rest.
        for i, rect in enumerate(self._spec_mask_rects):
            if i < len(boxes):
                x, y, w, h = boxes[i]
                rect.setRect(x, y, w, h)
                rect.setVisible(True)
            else:
                rect.setVisible(False)

    def draw_group_delays(self, plot_widget, beat_frequencies, frequency_exclusions,
                          aggregated_hfs, aggregated_lfs):
        """Draw the group delay plot with all beat frequencies.

        Uses persistent PlotDataItem curves updated via setData() to avoid
        expensive clear() + plot() cycles.
        """
        if not self._gd_ready:
            # First call: create all persistent curves in correct z-order
            self._gd_agg_hfs = plot_widget.plot(pen=pg.mkPen(color='w', width=2))
            self._gd_agg_lfs = plot_widget.plot(pen=pg.mkPen(color='w', width=2))

            for side in SIDES:
                color = HFS_COLOR if side == 'HFS' else LFS_COLOR
                excl_color = HFS_EXCLUSION_COLOR if side == 'HFS' else LFS_EXCLUSION_COLOR
                for band in BANDS:
                    self._gd_band[(side, band)] = plot_widget.plot(
                        pen=pg.mkPen(color=color, width=2))
                    for i in range(_MAX_EXCL):
                        self._gd_excl[(side, band, i)] = plot_widget.plot(
                            pen=pg.mkPen(color=excl_color, width=2))

            plot_widget.setLabel('bottom', 'Probing Frequency', units='Hz')
            plot_widget.setLabel('left', 'Time Delay', units='s')
            self._gd_ready = True

        # Update aggregated lines
        self._gd_agg_hfs.setData(aggregated_hfs.f_probe, aggregated_hfs.beat_time)
        self._gd_agg_lfs.setData(aggregated_lfs.f_probe, aggregated_lfs.beat_time)

        # Update per-band lines and frequency-exclusion overlays
        for side in SIDES:
            excls = frequency_exclusions[side]
            for band in BANDS:
                bf = beat_frequencies[side][band]
                self._gd_band[(side, band)].setData(bf.f_probe, bf.y_beat_time)

                for i in range(_MAX_EXCL):
                    curve = self._gd_excl[(side, band, i)]
                    if i < len(excls) and excls[i].enabled:
                        excl = excls[i]
                        mask = (bf.f_probe >= excl.low) & (bf.f_probe <= excl.high)
                        curve.setData(bf.f_probe[mask], bf.y_beat_time[mask])
                    else:
                        curve.setData([], [])

    def draw_profile(self, plot_widget, r_HFS, ne_HFS, r_LFS, ne_LFS, coordinate_mode):
        """Draw density profile.

        Uses persistent PlotDataItem curves updated via setData().
        """
        if not self._prof_ready:
            self._prof_hfs = plot_widget.plot(pen=pg.mkPen(color=HFS_COLOR, width=2))
            self._prof_lfs = plot_widget.plot(pen=pg.mkPen(color=LFS_COLOR, width=2))
            plot_widget.setLabel('left', 'density', units='1e19 m^-3')
            self._prof_ready = True

        self._prof_hfs.setData(r_HFS, ne_HFS * 1e-19)
        self._prof_lfs.setData(r_LFS, ne_LFS * 1e-19)

        x_label = 'radius'
        x_units = 'm' if coordinate_mode == 'R (m)' else ''
        plot_widget.setLabel('bottom', x_label, units=x_units)
