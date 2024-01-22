from .spike_train import *


class Dataset:
    def __init__(self, n_neurons=100, method='poisson'):
        self.spike_trains = []
        self.n_neurons = n_neurons
        self.method = method

    def generate_data(self):
        if self.method == 'poisson':
            for i in range(self.n_neurons):
                self.spike_trains.append(generate_poisson_spikes())
        self.spike_trains=np.array(self.spike_trains)

