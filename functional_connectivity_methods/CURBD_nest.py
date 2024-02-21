#!/usr/bin/env python
"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% example script to generate simulated interacting brain regions and
% perform Current-Based Decomposition (CURBD). Ref:
%
% Perich MG et al. Inferring brain-wide interactions using data-constrained
% recurrent neural network models. bioRxiv. DOI: https://doi.org/10.1101/2020.12.18.423348
%
% Written by Matthew G. Perich and Eugene Carter. Updated December 2020.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

import numpy as np
import pylab
import matplotlib.pyplot as plt
import curbd
import nest


def get_spike_trains(number_neurons, simulation_time, senders, times, bin_size=1.0):

    num_bins = int(simulation_time / bin_size)
    binary_spike_trains = np.zeros((number_neurons, num_bins))

    for neuron_id, spike_time in zip(senders, times):
        neuron_index = neuron_id - min(senders)
        bin_index = int(spike_time / bin_size) - 1
        if bin_index < num_bins:
            binary_spike_trains[neuron_index, bin_index] = 1
    return binary_spike_trains




def simulate_ER_network(num_neurons=10, conn_prob=0.1, method='CURBD', neuron_type="iaf_psc_alpha"):
    nest.ResetKernel()
    nest.set_verbosity('M_FATAL')
    neurons = nest.Create(neuron_type, num_neurons)

    # ER network
    for i in range(len(neurons)):
        n_source = neurons[i]
        for j in range(i+1, len(neurons)):
            n_target = neurons[j]
            if n_source != n_target and np.random.rand() < conn_prob:
                nest.Connect(n_source, n_target, syn_spec={"weight": 1000.0})
                #{'synapse_model': 'my_stdp_synapse'})

    # neurons = nest.Create(neuron_type, 2)
    # nest.Connect(neurons[0], neurons[1], syn_spec={"weight": 2000.0})
    multimeter0 = nest.Create("multimeter", params={"record_from": ["V_m"]})
    #poisson_gen = nest.Create("poisson_generator", 1, {"rate": 80000.0})
    dc_gen = nest.Create("dc_generator", params={"amplitude": 5000.0})
    nest.Connect(dc_gen, neurons[0])
    nest.Connect(multimeter0, neurons)

    spikerecorder = nest.Create('spike_recorder')
    nest.Connect(neurons, spikerecorder)

    # Simulate
    sim_time = 1000.0
    nest.Simulate(sim_time)

    data_multimeter = nest.GetStatus(multimeter0)[0]["events"]
    #times = data_multimeter["times"]
    #V_m = data_multimeter["V_m"]
    #senders = data_multimeter["senders"]

    events_spikes = spikerecorder.get("events")
    senders = events_spikes["senders"]
    ts = events_spikes["times"]

    #plot_voltages(neurons, data_multimeter)

    spikes_matrix = get_spike_trains(num_neurons, sim_time, senders, ts, bin_size=1.0)

    print(curbd_net(spikes_matrix))
    return
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

    # thr = 0.2
    # bin_conn_matrix = threshold_matrix(conn_matrix, thr)
    # f1_score = calculate_f1_score(adj_matrix, bin_conn_matrix)
    print('f1 score: ', f1_score)
    return adj_matrix, bin_conn_matrix, f1_score




def curbd_net(activity):
    # sim = curbd.threeRegionSim(number_units=3, plotSim=True)
    # activity = np.concatenate((sim['Ra'], sim['Rb'], sim['Rc']), 0)
    #
    #
    # Na = sim['params']['Na']
    # Nb = sim['params']['Nb']
    # Nc = sim['params']['Nc']
    #
    # regions = []
    # regions.append(['Region A', np.arange(0, Na)])
    # regions.append(['Region B', np.arange(Na, Na + Nb)])
    # regions.append(['Region C', np.arange(Na + Nb, Na + Nb + Nc)])

    # print(regions)
    regions = []
    for i in range(activity.shape[0]):
        regions.append([i, np.arange(i-1, i)])
    regions = np.array(regions, dtype=object)
    model = curbd.trainMultiRegionRNN(activity,
                                      dtData=1,
                                      dtFactor=5,
                                      regions=regions,
                                      tauRNN=0.01,
                                      nRunTrain=500,
                                      verbose=False,
                                      plotStatus=False,
                                      nRunFree=5)

    [curbd_arr, curbd_labels] = curbd.computeCURBD(model)

    return curbd_arr, curbd_labels

    # n_regions = curbd_arr.shape[0]
    # n_region_units = curbd_arr[0, 0].shape[0]
    #
    # fig = pylab.figure(figsize=[8, 8])
    # count = 1
    # for iTarget in range(n_regions):
    #     for iSource in range(n_regions):
    #         axn = fig.add_subplot(n_regions, n_regions, count)
    #         count += 1
    #         axn.pcolormesh(model['tRNN'], range(n_region_units),
    #                        curbd_arr[iTarget, iSource])
    #         axn.set_xlabel('Time (s)')
    #         axn.set_ylabel('Neurons in {}'.format(regions[iTarget, 0]))
    #         axn.set_title(curbd_labels[iTarget, iSource])
    #         axn.title.set_fontsize(8)
    #         axn.xaxis.label.set_fontsize(8)
    #         axn.yaxis.label.set_fontsize(8)
    # fig.subplots_adjust(hspace=0.4, wspace=0.3)
    # fig.show()


simulate_ER_network(20, 0.5, 'CURBD')