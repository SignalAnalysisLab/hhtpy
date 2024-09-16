from typing import List
import numpy as np
from hhtpy._emd_utils import is_monotonic, is_imf, sift
from sift_stopping_criteria import (
    get_stopping_criterion_fixed_number_of_sifts,
    SiftStoppingCriterion,
)


def decompose(
    signal: np.ndarray,
    stopping_criterion: SiftStoppingCriterion = get_stopping_criterion_fixed_number_of_sifts(
        15
    ),
):
    """
    Perform the Empirical Mode Decomposition on a given signal.
    Args:
        signal: The input signal to decompose.
        stopping_criterion: The stopping criterion to use for the SIFT process.
        max_sift_iterations: The maximum number of SIFT iterations to perform.

    Returns:
        List[np.ndarray]: The Intrinsic Mode Functions (IMFs) of the signal.
        np.ndarray: The residue of the signal after decomposition.
    """
    if signal.size == 0:
        raise ValueError("Input signal must not be empty.")
    if signal.ndim != 1:
        raise ValueError("Input signal must be one-dimensional.")

    signal_std = np.std(signal)
    signal_mean = np.mean(signal)

    signal_normalized = (signal - signal_mean) / signal_std

    max_imfs = int(
        np.log2(len(signal)) - 1
    )  # The maximum possible IMFs from white noise characteristics

    residue = signal_normalized
    imfs: List[np.ndarray] = []

    for i in range(max_imfs):
        if is_imf(residue):
            imfs.append(residue)
            residue = np.zeros_like(residue)
            break

        if is_monotonic(residue):
            break

        mode = residue
        total_sifts_performed = 0

        while not stopping_criterion(mode, total_sifts_performed):
            mode = sift(mode)
            total_sifts_performed += 1

        residue -= mode
        imfs.append(mode)

    return np.array(imfs) * signal_std, residue * signal_std + signal_mean
