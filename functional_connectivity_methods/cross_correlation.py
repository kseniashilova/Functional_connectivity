import numpy as np
from .helper import *


def cross_correlation(data, n_bins, window_size, bin_size):
    new_data = transform_to_bins(data, n_bins)

