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
from network import GLM, spiketrains


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
    bin_m = (abs(m) > threshold).astype(int)
    return bin_m


def calculate_cosine_similarity(m1, m2):
    a = m1.flatten()
    b = m2.flatten()
    cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return cos_sim




def calculate_f1_score(m1, m2):
    tp = np.sum(np.logical_and(m1 == 1, m2 == 1))
    fp = np.sum(np.logical_and(m1 == 1, m2 == 0))
    fn = np.sum(np.logical_and(m1 == 0, m2 == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0


def plot_matrices(adj_matrix, conn_matrix):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    ax1 = axes[0]
    ax2 = axes[1]

    img1 = ax1.imshow(adj_matrix, cmap='viridis')
    ax1.set_title('Anatomical')
    ax1.figure.colorbar(img1, ax=ax1)

    img2 = ax2.imshow(conn_matrix, cmap='viridis')
    ax2.set_title('Functional')
    ax2.figure.colorbar(img2, ax=ax2)

    # plt.savefig(''.join(np.random.choice(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'], size=5)) + '.png')
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


def find_node(adj, number, mode='highest_degree'):
    if mode == 'highest_degree':
        degree_counts = np.sum(adj, axis=0)
        highest_degrees_indices = np.argsort(degree_counts)[(-1 * number):]
        nodes = highest_degrees_indices[::-1]
    elif mode == 'random':
        nodes = np.random.choice(range(len(adj)), size=number, replace=False)
    return nodes


def simulate_ER_network(num_neurons=10, conn_prob=0.1, method='Transfer Entropy', neuron_type="iaf_psc_alpha",
                        distribution_current='Gaussian', weightsCoef=100, whitenoise=True):
    nest.ResetKernel()
    nest.set_verbosity('M_FATAL')
    neurons = nest.Create(neuron_type, num_neurons)

    # ER network
    adj_matrix = np.zeros((num_neurons, num_neurons))
    for i in range(len(neurons)):
        n_source = neurons[i]
        for j in range(i + 1, len(neurons)):
            n_target = neurons[j]
            if n_source != n_target and np.random.rand() < conn_prob:
                nest.Connect(n_source, n_target, syn_spec={"weight": weightsCoef/num_neurons})
                nest.Connect(n_target, n_source, syn_spec={"weight": weightsCoef/num_neurons})
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1

    node_input = find_node(adj_matrix, number=1, mode='highest_degree')
    print(node_input)
    multimeter0 = nest.Create("multimeter", params={"record_from": ["V_m"]})
    # poisson_gen = nest.Create("poisson_generator", 1, {"rate": 80000.0})

    # generate input stimulus
    # if distribution_current == 'Gaussian':
    #     weights = np.random.normal(5000, 100, len(neurons))
    # elif distribution_current == 'Uniform':
    #     weights = np.random.uniform(400, 500, len(neurons))
    # else:
    #     print('Incorrect distribution type')
    #     return

    # Add white noise or not
    # if whitenoise:
    #     noise = np.random.normal(0, 10, len(neurons))
    #     weights += noise

    for node in node_input:
        dc_gen = nest.Create("dc_generator", params={"amplitude": 1000})
        nest.Connect(dc_gen, neurons[node])

    # for i in range(len(neurons)):
    #     dc_gen = nest.Create("dc_generator", params={"amplitude": weights[i]})
    #     nest.Connect(dc_gen, neurons[i])
    #     break # we need only 1-neuron input

    nest.Connect(multimeter0, neurons)

    spikerecorder = nest.Create('spike_recorder')
    nest.Connect(neurons, spikerecorder)

    # Simulate
    sim_time = 100.0
    nest.Simulate(sim_time)

    data_multimeter = nest.GetStatus(multimeter0)[0]["events"]
    # times = data_multimeter["times"]
    # V_m = data_multimeter["V_m"]
    # senders = data_multimeter["senders"]

    events_spikes = spikerecorder.get("events")
    senders = events_spikes["senders"]
    ts = events_spikes["times"]

    plot_voltages(neurons, data_multimeter)

    spikes_matrix = get_spike_trains(num_neurons, sim_time, senders, ts, bin_size=1.0)

    conn_matrix = np.zeros((num_neurons, num_neurons))
    spikes_matrix = spiketrains.transform_spike_trains(spikes_matrix, 50)
    conn_matrix = GLM.estimate_W(num_neurons, spikes_matrix.T)
    if method == 'GLM':
        # spikes_matrix = spiketrains.transform_spike_trains(spikes_matrix, 50)
        # conn_matrix = GLM.estimate_W(num_neurons, spikes_matrix.T)
        print('GLM')
    else:
        for neuron_id_1 in range(num_neurons):
            for neuron_id_2 in range(neuron_id_1 + 1, num_neurons):
                # data1 = ts[senders == neurons[neuron_id_1]]
                # data2 = ts[senders == neurons[neuron_id_2]]
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

    conn_matrix = (conn_matrix - conn_matrix.min()) / (conn_matrix.max() - conn_matrix.min())

    f1_score = 0
    # thr_max = 0
    # for thr in np.arange(0, 1.1, 0.1):
    #     bin_conn_matrix = threshold_matrix(conn_matrix, thr)
    #     tmp = calculate_f1_score(adj_matrix, bin_conn_matrix)
    #     if tmp >= f1_score:
    #         f1_score = tmp
    #         thr_max = thr

    thr = 0.8
    #bin_conn_matrix = threshold_matrix(conn_matrix, thr)
    #f1_score = calculate_f1_score(adj_matrix, bin_conn_matrix)
    f1_score = calculate_cosine_similarity(adj_matrix, conn_matrix)
    plot_matrices(adj_matrix, conn_matrix)
    # print('thr:  ', thr_max, ',  f1 score: ', f1_score)
    # print(' f1 score: ', f1_score)
    return adj_matrix, conn_matrix, f1_score


def start_network_simulation(ax, prob, neuron_type, distribution_current, method,
                             weightsCoef, whitenoise, lowerBound, upperBound, draw=True):
    print('Starting network simulation')
    print(ax, prob, neuron_type, distribution_current, method,
          weightsCoef, whitenoise, lowerBound, upperBound, draw)
    range_steps = range(lowerBound, upperBound)
    # for method in methods:
    # 'Pearson Correlation', 'Transfer Entropy', 'CCG'
    print(method)
    f1_lst_averaged = []
    for trial_n in tqdm(range(5)):
        f1_lst = []
        for num_neuron_for_ER in range_steps:
            anatomical, functional, f1 = simulate_ER_network(num_neuron_for_ER, prob,
                                                             method,
                                                             neuron_type,
                                                             distribution_current,
                                                             weightsCoef,
                                                             whitenoise)
            f1_lst.append(f1)
        f1_lst_averaged.append(f1_lst)
    f1_lst_averaged = np.mean(f1_lst_averaged, axis=0)
    if draw:
        ax.plot(range_steps, f1_lst_averaged, label=method)

    if draw:
        ax.legend()
        ax.set_title('ER network')
        ax.set_xlabel('network size')
        #ax.set_ylabel('f1 score')
        ax.set_ylabel('cosine similarity')

    return f1_lst_averaged

# poisson spikes
# IF model
# slide 42
# STIMULUS-DEPENDENT FUNCTIONAL NETWORK TOPOLOGY IN MOUSE VISUAL CORTEX

# white noise
