import numpy as np
from .helper import *

def pearson_connectivity_map(data, n_bins=20):
    new_data=transform_to_bins(data, n_bins)

    corr_map = np.corrcoef(new_data)
    return corr_map
