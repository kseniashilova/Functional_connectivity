import numpy as np


def bin_spike_trains_fixed_size(spike_train, num_bins):
    bin_size = int(len(spike_train) / num_bins)
    binned_spike_counts = np.zeros(num_bins, dtype=int)

    for i in range(num_bins):
        bin_start = i * bin_size
        bin_end = bin_start + bin_size
        binned_spike_counts[i] = spike_train[bin_start:bin_end].sum()

    return binned_spike_counts


def transform_spike_trains(spike_trains, num_bins):
    mat_bins = np.zeros((len(spike_trains), num_bins))
    for i in range(len(spike_trains)):
        mat_bins[i] = bin_spike_trains_fixed_size(spike_trains[i], num_bins)
    return mat_bins


