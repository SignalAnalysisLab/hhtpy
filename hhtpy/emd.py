from typing import Union, List
import numpy as np
from scipy.interpolate import CubicSpline as CubicSpline
from hhtpy._emd_utils import include_endpoints_in_extrema, is_monotonic, is_imf, find_local_extrema
from typing import Callable


class EmpiricalModeDecomposition:
    def __init__(self, signal: np.ndarray, stopping_criterion: Callable = None, max_sift_iterations: int = 15):
        if signal.size == 0:
            raise ValueError("Input _signal_normalized must not be empty.")
        if signal.ndim != 1:
            raise ValueError("Input _signal_normalized must be one-dimensional.")

        self._imfs: List[np.ndarray] = []
        self._residue: np.ndarray = None

        self.max_sift_iterations: int = max_sift_iterations
        self.get_max_amount_imfs = lambda: int(
            np.log2(
                len(self._signal_normalized))) - 1  # Calculates maximum possible IMFs from white noise characteristics

        # Default stopping criterion if none provided
        if stopping_criterion is None:
            self.stopping_criterion = self._count_to_max_then_stop_sifting
        else:
            self.stopping_criterion = stopping_criterion

        self._std_signal = np.std(signal)
        self._mean_signal = np.mean(signal)
        self._signal_normalized = (signal.copy() - self._mean_signal) / self._std_signal

        self._sift_counter: int = 0

    @property
    def signal(self):
        return self._signal_normalized * self._std_signal + self._mean_signal

    @property
    def imfs(self):
        return np.array(self._imfs) * self._std_signal

    @property
    def residue(self):
        return (self._std_signal * np.array(self._residue)) + self._mean_signal if self._residue is not None else None

    def _sift(self, mode: Union[List[float], np.ndarray]):
        while not self.stopping_criterion(mode):
            maxima_indices, minima_indices = find_local_extrema(mode)
            x_max, y_max = include_endpoints_in_extrema(maxima_indices, mode, extrema_type='maxima')
            x_min, y_min = include_endpoints_in_extrema(minima_indices, mode, extrema_type='minima')

            n = np.arange(len(mode))
            upper_envelope = CubicSpline(x_max, y_max)(n)
            lower_envelope = CubicSpline(x_min, y_min)(n)

            mean_envelope = 0.5 * (upper_envelope + lower_envelope)

            mode -= mean_envelope
        return mode

    def decompose(self):
        residue = self._signal_normalized.copy()
        for i in range(self.get_max_amount_imfs()):
            imf = self._sift(residue.copy())
            residue -= imf
            self._imfs.append(imf)

            # Always check if the residue before checking for IMF
            if is_monotonic(residue):
                self._residue = residue
                break

            if is_imf(residue):
                self._imfs.append(residue)
                break

    def set_stopping_criterion(self, stopping_criterion_func: Callable[
        [Union[List[float], np.ndarray], float], bool]) -> None:
        self._stopping_condition_func = stopping_criterion_func

    def _count_to_max_then_stop_sifting(self, _):
        """
        Stoppage criterion: count to `max_sift_iterations` and stop sifting.
        """
        self._sift_counter += 1
        if self._sift_counter >= self.max_sift_iterations:
            self._sift_counter = 0  # Reset for the next IMF
            return True  # Signal to stop sifting
        else:
            return False  # Continue sifting
