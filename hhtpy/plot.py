import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.collections import LineCollection
from hhtpy import IntrinsicModeFunction
from dataclasses import dataclass, field

from typing import Union, Optional, Any


@dataclass
class HilbertSpectrumConfig:
    time_variable: Optional[np.ndarray] = None
    log_color: bool = True
    log_freq: bool = False
    min_amplitude_lim: float = 1e-2
    amplitude_unit: Optional[str] = None
    max_number_of_imfs: Optional[int] = None
    fig: Optional[Any] = None
    ax: Optional[Any] = None


def plot_hilbert_spectrum(
    imfs: list[IntrinsicModeFunction],
    config: HilbertSpectrumConfig = HilbertSpectrumConfig(),
):
    """
    Hilbert spectrum plotting time-frequency against amplitude/power [Huang98]_.

    Args:
        imfs (list[IntrinsicModeFunction]): The IMFs to plot.
        config (HilbertSpectrumConfig): Configuration for the plot.
    """
    if len(imfs) == 0:
        raise ValueError("No IMFs to plot.")

    if config.max_number_of_imfs:
        num_imfs = np.min([len(imfs), config.max_number_of_imfs])
    else:
        num_imfs = len(imfs)

    fig = plt.figure() if config.fig is None else config.fig
    ax = plt.subplot2grid((1, 1), (0, 0)) if config.ax is None else config.ax

    time_variable = (
        np.arange(len(imfs[0].signal)) / imfs[0].sampling_frequency
        if config.time_variable is None
        else config.time_variable
    )

    x_lim = [0, time_variable[-1]]
    y_lim = _get_ylim(imfs)
    c_lim = _get_clim(imfs, config.min_amplitude_lim)

    for imf in imfs[:num_imfs]:
        frequency = imf.instantaneous_frequency

        if frequency is None:
            continue

        if config.log_freq:
            frequency[frequency <= 0] = None
            frequency = np.log10(frequency)

        points = np.array([time_variable, frequency]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        if config.log_color:
            norm = colors.LogNorm(c_lim[0], c_lim[1], clip=True)
        else:
            norm = plt.Normalize(c_lim[0], c_lim[1], clip=True)

        lc = LineCollection(
            segments=segments, array=imf.instantaneous_amplitude, norm=norm
        )
        ax.add_collection(lc)

    clb = fig.colorbar(ax.collections[-1], aspect=20, fraction=0.1, pad=0.02)

    clb.set_label(
        f"Amplitude {f'[{config.amplitude_unit}]' if config.amplitude_unit else ''}"
    )

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    if config.log_freq:
        ax.set_yscale("log")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")

    return fig, ax, clb


def _get_ylim(imfs):
    freqs = []
    for imf in imfs:
        freq = imf.instantaneous_frequency[~np.isnan(imf.instantaneous_frequency)]
        freqs.extend([np.min(freq), np.max(freq)])

    freqs = np.array(freqs)
    return np.min(freqs) * 0.9, np.max(freqs) * 1.1


def _get_clim(imfs, min_amplitude):
    amps = []
    for imf in imfs:
        amp = imf.instantaneous_amplitude[~np.isnan(imf.instantaneous_amplitude)]
        amps.extend([np.min(amp), np.max(amp)])

    amps = np.array(amps)
    return np.max((np.min(amps), min_amplitude)), np.max(amps)


def plot_imfs(
    imfs: Union[np.ndarray, list[IntrinsicModeFunction]],
    signal: np.ndarray = None,
    residue: np.ndarray = None,
    x_axis: np.ndarray = None,
    max_number_of_imfs: int = None,
):
    """
    Plot the IMFs and the _residue of the EMD decomposition.

    Args:
        imfs (np.ndarray): The IMFs to plot.
        signal (np.ndarray): The original signal.
        residure (np.ndarray): The _residue of the EMD decomposition.
    """
    imfs = (
        [imf.signal for imf in imfs]
        if isinstance(imfs[0], IntrinsicModeFunction)
        else imfs
    )

    if max_number_of_imfs:
        num_imfs = np.min([len(imfs), max_number_of_imfs])

    fig, axs = plt.subplots(num_imfs + 2, 1)
    if signal is not None:
        axs[0].plot(signal) if x_axis is None else axs[0].plot(x_axis, signal)
    for i in range(num_imfs):
        axs[i + 1].plot(imfs[i]) if x_axis is None else axs[i + 1].plot(x_axis, imfs[i])
    if residue is not None:
        axs[-1].plot(residue) if x_axis is None else axs[-1].plot(x_axis, residue)

    return fig, axs
