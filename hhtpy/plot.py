import numpy as np
import matplotlib.pyplot as plt

def plot_imfs(imfs: np.ndarray, signal: np.ndarray=None, residue: np.ndarray=None, x_axis: np.ndarray=None, show_plot: bool=True, save_path: str=None):
    """
    Plot the IMFs and the _residue of the EMD decomposition.

    Args:
        imfs (np.ndarray): The IMFs to plot.
        signal (np.ndarray): The original signal.
        residure (np.ndarray): The _residue of the EMD decomposition.
    """
    num_imfs = len(imfs)
    fig, axs = plt.subplots(num_imfs + 2, 1)
    if signal is not None:
        axs[0].plot(signal) if x_axis is None else axs[0].plot(x_axis, signal)
    for i in range(num_imfs):
        axs[i + 1].plot(imfs[i]) if x_axis is None else axs[i + 1].plot(x_axis, imfs[i])
    if residue is not None:
        axs[-1].plot(residue) if x_axis is None else axs[-1].plot(x_axis, residue)

    plt.show() if show_plot else None

    return fig, axs