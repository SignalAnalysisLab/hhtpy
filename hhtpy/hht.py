from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import median_filter
from hhtpy._emd_utils import find_local_extrema
from hhtpy.emd import decompose


@dataclass
class IntrinsicModeFunction:
    """
    Dataclass to store the intrinsic mode function (IMF) and its instantaneous frequency.
    """

    signal: np.ndarray
    instantaneous_frequency: np.ndarray
    instantaneous_amplitude: np.ndarray
    sampling_frequency: float


def calculate_instantaneous_frequency_quadrature(
    imf: np.ndarray,
    sampling_frequency: float,
    normalize: bool = True,
    median_filter_window_pct: float = 0.05,
) -> np.ndarray:
    """
    Calculate the instantaneous frequency using the quadrature method.

    Parameters:
        imf (np.ndarray): Input intrinsic mode function (IMF).
        sampling_frequency (float): Sampling frequency of the signal.
        normalize (bool): Whether to normalize the IMF. Default is True.

    Returns:
        np.ndarray: Instantaneous frequency array.

    Raises:
        ValueError: If normalization fails after a certain number of attempts.
    """
    imf = imf.copy()

    if normalize:
        imf = normalize_imf(imf, max_attempts=150)

    frequency = quadrature_method(imf, sampling_frequency)
    frequency = median_filter(
        frequency, size=int(sampling_frequency * median_filter_window_pct)
    )

    return frequency


def normalize_imf(imf: np.ndarray, max_attempts: int) -> np.ndarray:
    """
    Normalize the IMF by iteratively dividing by its instantaneous amplitude spline.

    Parameters:
        imf (np.ndarray): Input intrinsic mode function (IMF).
        max_attempts (int): Maximum number of normalization attempts.

    Returns:
        np.ndarray: Normalized IMF.

    Raises:
        ValueError: If the maximum value of the IMF remains greater than 1 after max_attempts.
    """
    for _ in np.arange(max_attempts):
        if np.max(imf) <= 1:
            break

        imf /= calculate_instantaneous_amplitude_spline(imf)
    else:
        raise ValueError(
            f"Normalization of the IMF failed after {max_attempts} attempts. "
            f"Maximum value is still greater than 1 (current max: {np.max(imf):.3f})."
        )

    return imf


def calculate_instantaneous_amplitude_spline(imf: np.ndarray) -> np.ndarray:
    x_max, _ = find_local_extrema(np.abs(imf))
    x_max = np.concatenate(([0], x_max, [len(imf) - 1]))

    n = np.arange(len(imf))
    return CubicSpline(x_max, abs(imf[x_max]))(n)


def _quadrature_phase(monocomponent_normalized: np.ndarray) -> np.ndarray:
    """
    Calculates the quadrature phase of the normalized IMF signal.

    The quadrature phase is given by:

    .. math::
        \\theta(t) = \\arctan{\\frac{q(t)}{x(t)}}

    where `q(t)` is the quadrature of the signal.

    Args:
        monocomponent_normalized (np.ndarray): IMF normalized between -1 and 1.

    Returns:
        np.ndarray: Quadrature phase :math:`\\theta(t)`.
    """
    if not isinstance(monocomponent_normalized, np.ndarray):
        raise ValueError("Input must be a NumPy array.")

    if not np.all(np.abs(monocomponent_normalized) <= 1):
        raise ValueError("Input values must be normalized between -1 and 1.")

    quadrature = _calculate_quadrature(monocomponent_normalized)
    z = monocomponent_normalized + 1j * quadrature
    return np.angle(z)


def quadrature_method(
    monocomponent_normalized: np.ndarray, sampling_frequency: float
) -> np.ndarray:
    """
    Calculates the instantaneous frequency using the quadrature method.

    Assuming an analytic signal of the form:

    .. math::
        z(t) = x(t) + i q(t)

    The instantaneous frequency is given by:

    .. math::
        \\omega(t) = \\frac{F_s}{2 \\pi} \\cdot \\frac{d}{dt} \\measuredangle{z(t)}

    Args:
        monocomponent_normalized (np.ndarray): IMF normalized between -1 and 1.
        sampling_frequency (float): Sampling frequency in Hz.

    Returns:
        np.ndarray: Instantaneous frequency.
    """
    if not isinstance(monocomponent_normalized, np.ndarray):
        raise ValueError("Input must be a NumPy array.")
    if not isinstance(sampling_frequency, (int, float)):
        raise ValueError("Sampling frequency must be a float or integer.")
    if not np.all(np.abs(monocomponent_normalized) <= 1):
        raise ValueError("Input values must be normalized between -1 and 1.")

    phase = _quadrature_phase(monocomponent_normalized)
    frequency = sampling_frequency / (2 * np.pi) * np.abs(np.gradient(phase))

    return frequency


def _calculate_quadrature(monocomponent: np.ndarray) -> np.ndarray:
    """
    Calculates the quadrature of the normalized IMF signal.

    The quadrature is calculated as:

    .. math::
        q(t) = \\text{sign}\\left(\\frac{dx(t)}{dt}\\right) \\cdot \\sqrt{1 - x^2(t)}

    Args:
        monocomponent (np.ndarray): IMF normalized between -1 and 1.

    Returns:
        np.ndarray: Quadrature :math:`q(t)`.
    """
    if not isinstance(monocomponent, np.ndarray):
        raise ValueError("Input must be a NumPy array.")
    if not np.all(np.abs(monocomponent) <= 1):
        raise ValueError("Input values must be normalized between -1 and 1.")

    # Calculate the sign based on the derivative of the signal
    sign = np.zeros_like(monocomponent)
    sign[:-1] = -np.sign(np.diff(monocomponent))
    sign[-1] = sign[-2]  # Handle the last element by copying the second last

    # Calculate the quadrature with numerical stability (handling small values)
    quadrature = sign * np.sqrt(np.maximum(0, 1 - monocomponent**2))

    return quadrature


FrequencyCalculationMethod = Callable[[np.ndarray, float], np.ndarray]
AmplitudeCalculationMethod = Callable[[np.ndarray], np.ndarray]


def hilbert_huang_transform(
    signal: np.ndarray,
    sampling_frequency: float,
    frequency_calculation_method: FrequencyCalculationMethod = calculate_instantaneous_frequency_quadrature,
    amplitude_calculation_method: AmplitudeCalculationMethod = calculate_instantaneous_amplitude_spline,
) -> (list[IntrinsicModeFunction], np.ndarray):
    """
    Perform the Hilbert-Huang Transform on the input signal.

    Parameters:

        signal (np.ndarray): Input signal.
        sampling_frequency (float): Sampling frequency of the signal.
        amplitude_calculation_method:
        frequency_calculation_method:

    Returns:
        np.ndarray: Instantaneous frequency array.
    """

    imfs, residue = decompose(signal)

    return [
        IntrinsicModeFunction(
            signal=imf,
            instantaneous_frequency=frequency_calculation_method(
                imf, sampling_frequency
            ),
            instantaneous_amplitude=amplitude_calculation_method(imf),
            sampling_frequency=sampling_frequency,
        )
        for imf in imfs
    ], residue
