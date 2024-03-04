import itertools

import numpy as np
import pandas as pd
from network import ER_network


def create_table():
    sizes = np.array([5, 10, 20, 30, 50, 70, 100])
    #sizes = np.array([20])
    distribs = ['Gaussian', 'Uniform']
    methods = ['Pearson Correlation', 'Transfer Entropy', 'CCG', 'GLM']
    probs = [0.2, 0.3, 0.5, 0.7]
    neuron_types = ['iaf_psc_alpha']
    weightsCoefs = [1000]
    whitenoises = [True, False]
    ax = [None]

    results = []
    for combination in itertools.product(ax, probs, neuron_types, distribs, methods,
                                         weightsCoefs, whitenoises, sizes):
        ax, prob, neuron_type, distrib, method, weightsCoef, whitenoise, size = combination
        f1 = ER_network.start_network_simulation(*combination, upperBound=size + 1, draw=False)
        # results.append([f1, combination])

        # df = df.append({'prob': prob, 'neuron_type': neuron_type, 'distrib': distrib,
        #                 'method': method, 'weightsCoef': weightsCoef, 'whitenoise': whitenoise,
        #                 'size': size, 'f1': f1}, ignore_index=True)

        results.append({'prob': prob, 'neuron_type': neuron_type, 'distrib': distrib,
                        'method': method, 'weightsCoef': weightsCoef, 'whitenoise': whitenoise,
                        'size': size, 'f1': f1[0]})

    df = pd.DataFrame(data=results,
                      columns=['prob', 'neuron_type', 'distrib', 'method', 'weightsCoef', 'whitenoise', 'size', 'f1'])

    return df

    # for size in sizes:
    #     for prob in probs:
    #         for neuron_type in neuron_types:
    #             for distrib in distribs:
    #                 for method in methods:
    #                     for weightsCoef in weightsCoefs:
    #                         for whitenoise in whitenoises:
    #
    #
    #                             dct_table = {
    #                                 "size": size,
    #                                 "prob": prob,
    #                                 "neuron_type": neuron_type,
    #                                 'distrib': distrib,
    #                                 'method': method,
    #                                 'weightsCoef': weightsCoef,
    #                                 'whitenoise': whitenoise
    #                             }
    #
    #                             f1_score = ER_network.start_network_simulation(ax, prob,
    #                                                                 neuron_type,
    #                                                                 distrib,
    #                                                                 [method],
    #                                                                 weightsCoef,
    #                                                                 whitenoise, size, size + 1, draw=False)
    #
    #                             dct_table['f1_score'] = f1_score
    #                             table.append(dct_table)
    # return table


def save_table():
    table = create_table()
    print(table)
    table.to_csv('f1.csv', index=False)
    table

save_table()