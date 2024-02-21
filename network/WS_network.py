import numpy as np
import nest

import sklearn
import scipy
import numpy as np
import nest
import nest.raster_plot
import matplotlib.pyplot as plt
from scipy import signal
import pylab
from tqdm import tqdm

from functional_connectivity_methods import entropy_methods, cross_correlation


def generate_ws_network(N, k, p):

    # N: number of nodes
    # k: each node is connected to k nearest neighbors in ring topology
    # p: probability of rewiring each edge

    adjacency_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(1, k // 2 + 1):
            adjacency_matrix[i, (i + j) % N] = 1
            adjacency_matrix[(i + j) % N, i] = 1

    # Rewiring
    for i in range(N):
        for j in range(i + 1, i + k // 2 + 1):  # Only look ahead to avoid duplicate work
            if np.random.rand() < p:
                tmp_arr = np.hstack([np.arange(i - k // 2, i + k // 2 + 1), i])
                tmp_arr = tmp_arr[tmp_arr<N]
                possible_targets = np.delete(np.arange(N), tmp_arr)
                new_target = np.random.choice(possible_targets)
                adjacency_matrix[i, j % N] = 0
                adjacency_matrix[j % N, i] = 0
                adjacency_matrix[i, new_target] = 1
                adjacency_matrix[new_target, i] = 1
    return adjacency_matrix




def plot_voltages(neurons, data):
    times = data["times"]
    V_m = data["V_m"]
    senders = data["senders"]

    plt.figure(figsize=(12, 8))
    for neuron_id in neurons:
        neuron_data = V_m[senders == neuron_id]
        plt.plot(times[senders == neuron_id], neuron_data, label=f"Neuron {neuron_id}")
    plt.xlabel("Time (ms)")
    plt.ylabel("Membrane potential (mV)")
    plt.title("Membrane Potential of Neurons in the Network")
    plt.legend()
    plt.show()


def threshold_matrix(m, threshold):
    bin_m = (m > threshold).astype(int)
    return bin_m


def calculate_f1_score(m1, m2):
    # Ensure matrices are binary
    assert np.array_equal(m1, m1.astype(bool)) and np.array_equal(m2, m2.astype(
        bool)), "Matrices must be binary"

    tp = np.sum(np.logical_and(m1 == 1, m2 == 1))
    fp = np.sum(np.logical_and(m1 == 1, m2 == 0))
    fn = np.sum(np.logical_and(m1 == 0, m2 == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0



def plot_matrices(anatomical_matrix, functional_matrix):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    img1 = axes[0].imshow(anatomical_matrix, cmap='viridis')
    axes[0].set_title('Anatomical')
    axes[0].figure.colorbar(img1, ax=axes[0])

    img2 = axes[1].imshow(functional_matrix, cmap='viridis')
    axes[1].set_title('Functional')
    axes[1].figure.colorbar(img2, ax=axes[1])

    plt.show()


def get_spike_trains(number_neurons, simulation_time, senders, times, bin_size=1.0):

    num_bins = int(simulation_time / bin_size)
    binary_spike_trains = np.zeros((number_neurons, num_bins))

    for neuron_id, spike_time in zip(senders, times):
        neuron_index = neuron_id - min(senders)
        bin_index = int(spike_time / bin_size) - 1
        if bin_index < num_bins:
            binary_spike_trains[neuron_index, bin_index] = 1
    return binary_spike_trains


def simulate_WS_network(num_neurons=10, k=10, p=0.1, method='Transfer Entropy', neuron_type="iaf_psc_alpha"):

    nest.ResetKernel()
    nest.set_verbosity('M_FATAL')
    neurons = nest.Create(neuron_type, num_neurons)


    ws_adjacency_matrix = generate_ws_network(num_neurons, k, p)

    for i in range(num_neurons):
        for j in range(num_neurons):
            if ws_adjacency_matrix[i, j] == 1:
                nest.Connect(neurons[i], neurons[j], syn_spec={"weight": 1000.0})


    multimeter0 = nest.Create("multimeter", params={"record_from": ["V_m"]})
    dc_gen = nest.Create("dc_generator", params={"amplitude": 5000.0})
    nest.Connect(dc_gen, neurons[0])
    nest.Connect(multimeter0, neurons)

    spikerecorder = nest.Create('spike_recorder')
    nest.Connect(neurons, spikerecorder)

    # Simulate
    sim_time = 1000.0
    nest.Simulate(sim_time)

    data_multimeter = nest.GetStatus(multimeter0)[0]["events"]

    events_spikes = spikerecorder.get("events")
    senders = events_spikes["senders"]
    ts = events_spikes["times"]

    plot_voltages(neurons, data_multimeter)

    spikes_matrix = get_spike_trains(num_neurons, sim_time, senders, ts, bin_size=1.0)

    conn_matrix = np.zeros((num_neurons, num_neurons))
    for neuron_id_1 in range(num_neurons):
        for neuron_id_2 in range(neuron_id_1 + 1, num_neurons):
            #data1 = ts[senders == neurons[neuron_id_1]]
            #data2 = ts[senders == neurons[neuron_id_2]]
            data1 = spikes_matrix[neuron_id_1, :]
            data2 = spikes_matrix[neuron_id_2, :]
            if np.all(data1 == 0) or np.all(data2 == 0):
                continue
            if method == 'Transfer Entropy':
                conn_matrix[neuron_id_1, neuron_id_2] = entropy_methods.tr_ent(data1, data2, 3)
                conn_matrix[neuron_id_2, neuron_id_1] = entropy_methods.tr_ent(data2, data1, 3)
            elif method == 'CCG':
                corr_val = cross_correlation(data1, data2)
                conn_matrix[neuron_id_1, neuron_id_2] = corr_val
                conn_matrix[neuron_id_2, neuron_id_1] = conn_matrix[neuron_id_1, neuron_id_2]
            elif method == 'Pearson Correlation':
                conn_matrix[neuron_id_1, neuron_id_2] = np.corrcoef(data1, data2)[0, 1]
                conn_matrix[neuron_id_2, neuron_id_1] = conn_matrix[neuron_id_1, neuron_id_2]

    adj_matrix = np.zeros((num_neurons, num_neurons))
    for neuron in neurons:
        conns = nest.GetConnections(neuron)
        try:
            if conns is not None:
                for conn in conns:
                    src, tgt = conn.source, conn.target
                    if (src-1 < num_neurons) and (tgt-1 < num_neurons):
                        adj_matrix[src - 1, tgt - 1] = 1
                        adj_matrix[tgt - 1, src - 1] = 1
        except TypeError as e:
            print(e)


    #plot_matrices(adj_matrix, conn_matrix)
    f1_score = 0
    for thr in np.arange(0, 1, 0.1):
        bin_conn_matrix = threshold_matrix(conn_matrix, thr)
        tmp = calculate_f1_score(adj_matrix, bin_conn_matrix)
        if tmp >= f1_score:
            f1_score = tmp


    print('f1 score: ', f1_score)
    return adj_matrix, bin_conn_matrix, f1_score



def start_network_simulation(ax, lowerBound, upperBound, prob, neuron_type, distribution_current):
    range_steps = range(lowerBound, upperBound)
    for method in ['Pearson Correlation', 'Transfer Entropy', 'CCG']:
        #'Pearson Correlation', 'Transfer Entropy', 'CCG'
        print(method)
        f1_lst_averaged = []
        for trial_n in tqdm(range(1)):
            f1_lst = []
            for num_neuron_for_WS in range_steps:
                anatomical, functional, f1 = simulate_WS_network(num_neuron_for_WS, int(num_neuron_for_WS*0.1),
                                                                 prob,
                                                                 method,
                                                                 neuron_type,
                                                                 distribution_current)
                f1_lst.append(f1)
            f1_lst_averaged.append(f1_lst)
        f1_lst_averaged = np.mean(f1_lst_averaged, axis=0)

        ax.plot(range_steps, f1_lst_averaged, label=method)

    ax.legend()
    ax.set_title('WS network')
    ax.set_xlabel('network size')
    ax.set_ylabel('f1 score')
    #plt.show()


