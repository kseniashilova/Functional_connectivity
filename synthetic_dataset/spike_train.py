import numpy as np


def generate_poisson_spikes(firing_rate=10, duration=60, dt=0.001):
    spikes = []
    t = 0
    while t <= duration:
        if np.random.rand() < firing_rate * dt:
            spikes.append(1)
        else:
            spikes.append(0)
        t += dt

    return spikes


def generate_integrate_and_fire_spikes():
    pass

