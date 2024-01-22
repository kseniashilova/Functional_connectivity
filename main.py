import numpy as np
import matplotlib.pyplot as plt
from synthetic_dataset import *
from functional_connectivity_methods import *

if __name__ == '__main__':
    generation_method = 'poisson'
    connectivity_method = 'pearson corr'
    dataset = Dataset(100, generation_method)
    dataset.generate_data()
    correlation_map = pearson_connectivity_map(dataset.spike_trains)
    print(dataset.spike_trains)
    print(correlation_map.shape)
    plt.imshow(correlation_map)
    plt.title(f'Generation method: {generation_method}, connectivity method: {connectivity_method}')
    plt.colorbar()
    plt.show()

