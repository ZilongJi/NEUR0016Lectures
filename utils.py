import numpy as np
from scipy.ndimage import gaussian_filter

def compute_firing_rate_map(positions, spikes, timestamps, bin_size=2.5, smoothing=True, gaussian_sigma=2):
    """
    Compute a 2D firing rate map from position and spike data.

    Parameters
    ----------
    positions : ndarray, shape (n, 2)
        X and Y coordinates of the animal's position.
    spikes : ndarray, shape (n,)
        Spike counts aligned to each position sample.
    timestamps : ndarray, shape (n,)
        Time of each position sample (must be sorted).
    bin_size : float, optional
        Size of spatial bins in cm. Default is 2.5.
    smoothing : {"unsmoothed", "gaussian"}, optional
        Smoothing method. "unsmoothed" (default) or "gaussian".

    Returns
    -------
    firing_rate_map : 2D ndarray
        Firing rate (Hz) map over space.
    """
    x, y = positions[:, 0], positions[:, 1]

    # Define spatial bins
    x_bins = np.arange(x.min(), x.max() + bin_size, bin_size)
    y_bins = np.arange(y.min(), y.max() + bin_size, bin_size)

    # Bin indices (0-based)
    x_idx = np.digitize(x, x_bins) - 1
    y_idx = np.digitize(y, y_bins) - 1

    # Estimate dwell times
    t_diff = np.diff(timestamps, prepend=timestamps[0])
    median_diff = np.median(t_diff)
    t_diff = np.minimum(t_diff, median_diff)  # cap outliers

    # Accumulate dwell times and spike counts
    dwell_times = np.zeros((len(x_bins), len(y_bins)))
    spike_counts = np.zeros((len(x_bins), len(y_bins)))
    np.add.at(dwell_times, (x_idx, y_idx), t_diff)
    np.add.at(spike_counts, (x_idx, y_idx), spikes)

    # Compute firing rate map
    if not smoothing:
        firing_rate_map = np.divide(
            spike_counts, dwell_times,
            out=np.zeros_like(spike_counts, dtype=float),
            where=dwell_times > 0
        )
    elif smoothing:
        dwell_smoothed = gaussian_filter(dwell_times, sigma=gaussian_sigma)
        spikes_smoothed = gaussian_filter(spike_counts, sigma=gaussian_sigma)
        firing_rate_map = np.divide(
            spikes_smoothed, dwell_smoothed,
            out=np.zeros_like(spikes_smoothed, dtype=float),
            where=dwell_smoothed > 0
        )

    return firing_rate_map
