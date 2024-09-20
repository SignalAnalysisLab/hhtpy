from scipy.signal import find_peaks
from numpy.typing import ArrayLike
from typing import Tuple
from typing import Union, List
import numpy as np
from scipy.interpolate import CubicSpline as CubicSpline


def sift(mode: Union[List[float], np.ndarray]):
    """
    TODO: Document the sifting process here. What part does it play in the EMD algorithm?
    """
    maxima_indices, minima_indices = find_local_extrema(mode)
    x_max, y_max = include_endpoints_in_extrema(
        maxima_indices, mode, extrema_type="maxima"
    )
    x_min, y_min = include_endpoints_in_extrema(
        minima_indices, mode, extrema_type="minima"
    )

    n = np.arange(len(mode))
    upper_envelope = CubicSpline(x_max, y_max)(n)
    lower_envelope = CubicSpline(x_min, y_min)(n)

    mean_envelope = 0.5 * (upper_envelope + lower_envelope)

    return mode - mean_envelope


def include_endpoints_in_extrema(
    x_extrema: np.ndarray, data: np.ndarray, extrema_type: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Include the start and end points in the extrema indices and values,
    correcting for end effects in cubic spline interpolation.

    Args:
        x_extrema (np.ndarray): Indices of the extrema used for interpolation.
        signal (np.ndarray): The input signal array.
        extrema_type (str): Type of extrema, either 'maxima' or 'minima'.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Updated x_extrema and y_extrema with end points included.
    """
    if extrema_type not in ["maxima", "minima"]:
        raise ValueError("extrema_type must be 'maxima' or 'minima'.")

    y_extrema = data[x_extrema]
    num_extrema = len(x_extrema)
    data_length = len(data)

    if num_extrema > 4:
        # Handle the start point
        if x_extrema[0] != 0:
            # Predict the value at the start using linear interpolation
            predicted_value = linear_interpolation_at_x(
                x_value=0,
                point_1=(x_extrema[0], y_extrema[0]),
                point_2=(x_extrema[1], y_extrema[1]),
            )
            x_extrema = np.insert(x_extrema, 0, 0)
            y_extrema = np.insert(y_extrema, 0, predicted_value)

        # Handle the end point
        if x_extrema[-1] != data_length - 1:
            # Predict the value at the end using linear interpolation
            predicted_value = linear_interpolation_at_x(
                x_value=data_length - 1,
                point_1=(x_extrema[-2], y_extrema[-2]),
                point_2=(x_extrema[-1], y_extrema[-1]),
            )
            x_extrema = np.append(x_extrema, data_length - 1)
            y_extrema = np.append(y_extrema, predicted_value)

    elif num_extrema >= 1:
        # Handle start point
        if x_extrema[0] != 0:
            x_extrema = np.insert(x_extrema, 0, 0)
            y_extrema = np.insert(y_extrema, 0, data[0])

        # Handle end point
        if x_extrema[-1] != data_length - 1:
            x_extrema = np.append(x_extrema, data_length - 1)
            y_extrema = np.append(y_extrema, data[-1])

    else:
        # No extrema found; use start and end points
        x_extrema = np.array([0, data_length - 1])
        y_extrema = np.array([data[0], data[-1]])

    return x_extrema, y_extrema


def linear_interpolation_at_x(
    x_value: float, point_1: Tuple[float, float], point_2: Tuple[float, float]
) -> float:
    """
    Perform linear interpolation between two points.

    Parameters:
    - x_value (float): The x-coordinate at which to evaluate the interpolated y-value.
    - point_1 (_Tuple[float, float]): The first point as (x, y).
    - point_2 (_Tuple[float, float]): The second point as (x, y).

    Returns:
    - float: Interpolated y-value at x_value.
    """

    # Extract coordinates
    x1, y1 = point_1
    x2, y2 = point_2

    # Ensure that x1 != x2 to avoid division by zero
    if x1 == x2:
        raise ValueError(
            "x-coordinates of point_1 and point_2 must be different for interpolation."
        )

    # Perform linear interpolation using numpy
    return np.interp(x_value, [x1, x2], [y1, y2])


def get_extrema_indices(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find the indices of local maxima and minima in the _signal_normalized, handling saddle points.

    Parameters:
    - _signal_normalized (np.ndarray): The input _signal_normalized array.

    Returns:
    - _Tuple[np.ndarray, np.ndarray]: Indices of local maxima and minima, respectively.
    """
    # Identify where the slope changes for maxima and minima
    maxima_indices = (data[:-2] < data[1:-1]) & (data[2:] < data[1:-1])
    minima_indices = (data[:-2] > data[1:-1]) & (data[2:] > data[1:-1])

    # Find indices of the maxima and minima
    maxima_indices = np.where(maxima_indices)[0] + 1
    minima_indices = np.where(minima_indices)[0] + 1

    # Handle equal points (saddle points)
    maxima_indices, minima_indices = handle_saddle_points(
        data, maxima_indices, minima_indices
    )

    return np.sort(maxima_indices), np.sort(minima_indices)


def handle_saddle_points(
    data: np.ndarray, maxima_indices: np.ndarray, minima_indices: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Handle saddle points where adjacent values in the _signal_normalized are equal.

    Parameters:
    - _signal_normalized (np.ndarray): The input _signal_normalized array.
    - maxima_indices (np.ndarray): Detected maxima indices.
    - minima_indices (np.ndarray): Detected minima indices.

    Returns:
    - _Tuple[np.ndarray, np.ndarray]: Updated maxima and minima indices considering saddle points.
    """
    equal_indices = np.where(np.diff(data) == 0)[0]

    for eq_index in equal_indices:
        # Check the surrounding values to determine if it's part of a saddle point
        left_value = data[eq_index - 1] if eq_index - 1 >= 0 else np.inf
        right_value = data[eq_index + 2] if eq_index + 2 < len(data) else np.inf

        if data[eq_index] > left_value and data[eq_index + 1] > right_value:
            maxima_indices = np.append(maxima_indices, eq_index + 1)
        elif data[eq_index] < left_value and data[eq_index + 1] < right_value:
            minima_indices = np.append(minima_indices, eq_index + 1)

    return maxima_indices, minima_indices


def is_monotonic(signal: ArrayLike) -> bool:
    diff = np.diff(signal)
    return np.all(diff >= 0) or np.all(diff <= 0)


def is_imf(
    signal: ArrayLike, tolerance: float = 0.01, strick_mode: bool = False
) -> bool:
    """
    Check if the input array satisfies the criteria for an Intrinsic Mode Function (IMF).

    An IMF must satisfy two conditions:
    1. The number of extrema and the number of zero crossings must differ at most by one.
    2. The mean of the envelopes defined by the local maxima and minima is approximately zero.

    Args:
        signal (ArrayLike): Input array to check.
        tolerance (float): Tolerance for the mean envelope to be considered zero.

    Returns:
        bool: True if the array is an IMF, False otherwise.
    """
    # Calculate the differences between consecutive elements
    signal = np.asarray(signal)
    n = len(signal)

    if n < 3:
        return False  # An IMF requires at least 3 points

    signal -= np.mean(signal)
    diff = np.diff(signal)

    # Find where the first derivative changes sign: maxima and minima
    maxima = (diff[:-1] > 0) & (diff[1:] <= 0)  # Maxima points
    minima = (diff[:-1] < 0) & (diff[1:] >= 0)  # Minima points

    # Get the actual maxima and minima values
    maxima_values = signal[1:-1][maxima]
    minima_values = signal[1:-1][minima]

    # Check if all maxima are positive and all minima are negative
    if not (np.all(maxima_values > 0) and np.all(minima_values < 0)):
        return False

    if strick_mode:
        # Interpolate upper and lower envelopes
        x = np.arange(n)
        from scipy.interpolate import CubicSpline

        maxima_indices, minima_indices = find_local_extrema(signal)
        x_max, y_max = include_endpoints_in_extrema(
            maxima_indices, signal, extrema_type="maxima"
        )
        x_min, y_min = include_endpoints_in_extrema(
            minima_indices, signal, extrema_type="minima"
        )

        n = np.arange(len(signal))
        upper_envelope = CubicSpline(x_max, y_max)(n)
        lower_envelope = CubicSpline(x_min, y_min)(n)

        mean_envelope = 0.5 * (upper_envelope + lower_envelope)

        # Check if the mean envelope is approximately zero
        mean_abs = np.abs(mean_envelope)
        signal_amplitude = np.abs(signal)
        with np.errstate(divide="ignore", invalid="ignore"):
            normalized_mean = np.where(
                signal_amplitude != 0, mean_abs / signal_amplitude, 0
            )

        if np.max(normalized_mean) > tolerance:
            return False

    return True


def find_local_extrema(signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find the indices of local maxima and minima in the signal,
    handling saddle points (flat regions).

    Parameters:
        signal (np.ndarray): The input signal array.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Indices of local maxima and minima.
    """
    # Find maxima, including plateaus
    maxima_indices, _ = find_peaks(signal, plateau_size=1)

    # Find minima by inverting the signal
    minima_indices, _ = find_peaks(-signal, plateau_size=1)

    return maxima_indices, minima_indices


def get_freq_lim(imfs, padding=0.1):
    freqs = []
    for imf in imfs:
        freq = imf.instantaneous_frequency[~np.isnan(imf.instantaneous_frequency)]
        freqs.extend([np.min(freq), np.max(freq)])

    freqs = np.array(freqs)
    return np.min(freqs) * (1 - padding), np.max(freqs) * (1 + padding)
