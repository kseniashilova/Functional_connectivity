import itertools

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from network import ER_network


def create_table(ax, draw):
    sizes = np.array([5, 10, 20, 30, 40, 50, 60, 70, 80, 100, 150])
    distribs = ['Gaussian']
    methods = ['Pearson Correlation', 'Transfer Entropy', 'CCG', 'GLM']
    probs = [0.15]
    neuron_types = ['iaf_psc_alpha']
    weightsCoefs = [20000]
    whitenoises = [False]
    ax = [ax]

    results = []
    for combination in itertools.product(ax, probs, neuron_types, distribs, methods,
                                         weightsCoefs, whitenoises, sizes):
        ax, prob, neuron_type, distrib, method, weightsCoef, whitenoise, size = combination
        f1 = ER_network.start_network_simulation(*combination, upperBound=size + 1, draw=draw)

        results.append({'prob': prob, 'neuron_type': neuron_type, 'distrib': distrib,
                        'method': method, 'weightsCoef': weightsCoef, 'whitenoise': whitenoise,
                        'size': size, 'f1': f1[0]})

    df = pd.DataFrame(data=results,
                      columns=['prob', 'neuron_type', 'distrib',
                               'method', 'weightsCoef', 'whitenoise',
                               'size', 'f1'])

    return df


def save_table(ax, draw):
    table = create_table(ax, draw)
    print(table)
    table.to_csv('f1(1).csv', index=False)

# save_table(None, False)


df = create_table(None, draw=False)
for method in ['Pearson Correlation', 'Transfer Entropy', 'CCG', 'GLM']:
    plt.plot(df[df['method']==method]['size'], df[df['method']==method]['f1'], label=method)
plt.legend()
plt.savefig('f1(1).png')