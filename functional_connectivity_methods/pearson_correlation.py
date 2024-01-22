import numpy as np


def pearson_connectivity_map(data, n_bins=20):
    new_data=[]
    for i in range(data.shape[0]):
        bin_step = int(len(data[i]) / n_bins)
        new_train = np.add.reduceat(data[i], np.arange(0, len(data[i]), bin_step))
        new_data.append(new_train)
    corr_map = np.corrcoef(new_data)
    return corr_map
