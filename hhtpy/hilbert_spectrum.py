# -------------------------
# Copyright (C) Signal Analysis Lab AS - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by Signal Analysis Lab AS, June 2017
# -------------------------

import sys
import time
from copy import copy
import matplotlib.pylab as plt
import numpy as np
from matplotlib import colors
from matplotlib.collections import LineCollection

from hhtpy import IntrinsicModeFunction


def plot_hilbert_spectrum(
    imfs: list[IntrinsicModeFunction],
    time_variable: np.ndarray,
    log_color=True,
    log_freq=False,
    fig=None,
    ax=None,
    c_lim=None,
    x_lim=None,
    y_lim=None,
):
    """
    Hilbert spectrum plotting time-frequency against amplitude/power [Huang98]_.

    Args:
        imfs (list[IntrinsicModeFunction]): The IMFs to plot.
        time_variable (np.ndarray): The time variable.
        log_color (bool):              If True, normalize the colors to the log10 of the amplitude.
        log_freq (bool):               If True display the log10 of the frequencies.
        fig (matplotlib.figure): Module that provides the top-level Artist, the Figure, which contains all the plot elements.
        ax (matplotlib.axes.Axes): Contains most of the figure elements: Axis, Tick, Line2D, Text, Polygon, etc., and sets the coordinate system.
        c_lim (tuple): Set the limit of the color axis. Chooses based on HilbertHuangTransform if None.
        x_lim (tuple): Set the limit of the x-axis. Chooses based on HilbertHuangTransform if None.
        y_lim (tuple): Set the limit of the y-axis. Chooses based on HilbertHuangTransform if None.
    """
    if len(imfs) == 0:
        raise ValueError("No IMFs to plot.")

    fig = plt.figure() if fig is None else fig
    ax = plt.subplot2grid((1, 1), (0, 0)) if ax is None else ax

    for i, imf in enumerate(imfs):
        frequency = copy(imf.instantaneous_frequency)

        if frequency is None:
            continue

        if log_freq:
            frequency[frequency <= 0] = None
            frequency = np.log10(frequency)

        points = np.array([time_variable, frequency]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        if log_color:
            norm = colors.LogNorm(c_lim[0], c_lim[1], clip=True)
        else:
            norm = plt.Normalize(c_lim[0], c_lim[1], clip=True)

        lc = LineCollection(
            segments=segments, array=imf.instantaneous_amplitude, norm=norm
        )
        ax.add_collection(lc)

    clb = fig.colorbar(ax.collections[-1], aspect=20, fraction=0.1, pad=0.02)

    clb.set_label("Amplitude")

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    if log_freq:
        ax.set_yscale("log")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")

    return fig, ax, clb
