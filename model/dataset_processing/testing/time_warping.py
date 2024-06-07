import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d


def time_warp(signal, warp_factor):
    """
    Apply time-warping to an EEG signal by stretching/compressing segments.
    Args:
        signal (np.ndarray): Original EEG signal.
        warp_factor (float): Factor by which to stretch (>1) or compress (<1) the signal.
    Returns:
        np.ndarray: Time-warped EEG signal.
    """
    length = len(signal)
    t = np.linspace(0, length - 1, length)
    warped_length = int(length * warp_factor)
    warped_indices = np.linspace(0, length - 1, warped_length)
    warped_signal = interp1d(warped_indices, np.interp(warped_indices, t, signal), kind='linear',
                             fill_value='extrapolate')(t)
    return warped_signal


# Example usage:
np.random.seed(42)
original_signal = np.random.randn(200)  # Example 1-second EEG signal at 200 Hz
warp_factor = np.random.uniform(0.9, 1.1)  # Subtle warp factor
warped_signal = time_warp(original_signal, warp_factor)

# Plotting the original and warped signals
plt.figure(figsize=(12, 6))
plt.plot(original_signal, label='Original Signal')
plt.plot(warped_signal, label='Warped Signal', linestyle='--')
plt.title('EEG Signal Before and After Time-Warping')
plt.xlabel('Sample Points')
plt.ylabel('Amplitude')
plt.legend()
plt.show()
