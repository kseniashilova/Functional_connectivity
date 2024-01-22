import numpy as np


def transform_to_bins(data, n_bins):
    new_data = []
    for i in range(data.shape[0]):
        bin_step = int(len(data[i]) / n_bins)
        new_train = np.add.reduceat(data[i], np.arange(0, len(data[i]), bin_step))
        new_data.append(new_train)
    return new_data
