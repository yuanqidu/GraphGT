# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 18:53:40 2020

@author: xguo7
"""
import tensorflow as tf
# tf.disable_v2_behavior()
from sklearn.metrics import mean_squared_error

flags = tf.compat.v1.flags
# flags = tf.app.flags
FLAGS = flags.FLAGS
import sys
from tqdm import tqdm
import numpy as np
import sklearn
import networkx as nx
import scipy
import os
import pickle
from random import sample
import matplotlib.pyplot as plt


# import seaborn as sns

def make_discretizer(target, num_bins=2):
    """Wrapper that creates discretizers."""
    return np.digitize(target, np.histogram(target, num_bins)[1][:-1])


def MI(z, f):
    z = np.array(z)
    f = np.array(f)
    m = []
    for i in range(z.shape[-1]):
        discretized_z = make_discretizer(z[:, :, i].reshape(-1))
        m.append(sklearn.metrics.normalized_mutual_info_score(discretized_z, f))
    return np.max(m)


def MI_factor(f):
    res = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            res[i][j] = sklearn.metrics.normalized_mutual_info_score(f[i].reshape(-1), f[j].reshape(-1))
    print('factor mutual info matrix', res)
    return res


# def MI_matrix(z, f):
#   z = np.concatenate(z,axis=-1)
#   print (z.shape)
#   pred=np.zeros((z.shape[-1],4))
#   print (pred.shape)
#   for i in range(z.shape[-1]):
#     for j in range(3):
#       for k in range(8):
#         pred[i,j] += MI(np.expand_dims(z[:,:,k,i],axis=-1),f[j][:,k])
#   pred /= 8
#   ax = sns.heatmap(pred)
#   plt.savefig('disentangle_heatmap_syn2_time_variant_only_time_constant_more_data_stop1e2max0.1.png')
#   plt.clf()
#   for i in range(z.shape[-1]):
#     j = 3
#     for k in range(z.shape[0]):
#       mi_score = MI(np.expand_dims(z[k,:,:,i],axis=-1),f[j][k,:])
#       pred[i,j] += mi_score
#     pred[i][j] /= 800
#   ax = sns.heatmap(pred)
#   plt.savefig('disentangle_heatmap_syn2_time_variant_only_time_constant_more_data_stop1e2max0.1.png')

def compute_kld_(generated, real):
    generated = [i / generated.count(i) for i in generated]
    real = [i / real.count(i) for i in real]
    return scipy.stats.entropy(generated, real)


def compute_KLD_from_graph(metric, generated_graph_list, real_graph_list):
    generated_l = list()
    real_l = list()

    for i in tqdm(range(len(generated_graph_list))):
        generated_G = generated_graph_list[i]
        real_G = real_graph_list[i]
        if metric == 'avg_clustering_dist':
            try:
                temp_value = nx.average_clustering(generated_G)
                generated_l.append(temp_value)
            except:
                generated_l.append(1.0)
            real_l.append(nx.average_clustering(real_G))
        if metric == 'density':
            generated_l.append(nx.density(generated_G))
            real_l.append(nx.density(real_G))
        if metric == 'avg_node_connectivity':
            generated_l.append(nx.average_node_connectivity(generated_G))
            real_l.append(nx.average_node_connectivity(real_G))

            # transfer to descrete:
    generated_discrete = make_discretizer(generated_l, num_bins=50)
    real_discrete = make_discretizer(real_l, num_bins=50)
    kld = compute_kld_(list(generated_discrete), list(real_discrete))
    return kld


def compute_KLD(metric, generated_adj, real_adj):
    generated_l = []
    real_l = []
    for i in tqdm(range(len(generated_adj))):
        generated_G = nx.from_numpy_matrix(generated_adj[i])
        real_G = nx.from_numpy_matrix(real_adj[i])
        if metric == 'avg_clustering_dist':
            generated_l.append(nx.average_clustering(generated_G))
            real_l.append(nx.average_clustering(real_G))
        if metric == 'density':
            generated_l.append(nx.density(generated_G))
            real_l.append(nx.density(real_G))
        if metric == 'avg_node_connectivity':
            generated_l.append(nx.average_node_connectivity(generated_G))
            real_l.append(nx.average_node_connectivity(real_G))

            # transfer to descrete:
    generated_discrete = make_discretizer(generated_l, num_bins=50)
    real_discrete = make_discretizer(real_l, num_bins=50)
    kld = compute_kld_(list(generated_discrete), list(real_discrete))
    return kld


def reconstruct_evaluation(generated_adj, generated_nodes, generated_spatial, real_adj, real_nodes, real_spatial):
    num_node = real_adj.shape[2]
    num_graph = real_adj.shape[0]
    num_time = real_adj.shape[1]
    generated_adj = generated_adj.reshape((num_graph, num_time, num_node, num_node))
    # evaluate the mse of node atrributes:
    mse_node = mean_squared_error(real_nodes.reshape(-1), generated_nodes.reshape(-1))
    # evaluate the mse of spatial atrributes:
    mse_spatial = mean_squared_error(real_spatial.reshape(-1), generated_spatial.reshape(-1))
    # evaluate the prediction acc of the adjacent matrix
    mse_adj = np.sum(np.sum(np.sum(np.sum(np.equal(generated_adj, real_adj))))) / (
            num_graph * num_time * num_node * num_node)
    # evaluate the KLD of graphs

    return mse_node, mse_spatial, mse_adj


def reconstruct_evaluation_simple(generated_adj, real_adj):
    num_node = real_adj.shape[2]
    num_graph = real_adj.shape[0]
    num_time = real_adj.shape[1]
    generated_adj = generated_adj.reshape((num_graph, num_time, num_node, num_node))

    # evaluate the prediction acc of the adjacent matrix
    mse_adj = np.sum(np.sum(np.sum(np.sum(np.equal(generated_adj, real_adj))))) / (
            num_graph * num_time * num_node * num_node)
    # evaluate the KLD of graphs

    return mse_adj


def generation_evaluation(generated_adj, generated_nodes, generated_spatial, real_adj, real_nodes, real_spatial,
                          dataset):
    real_adj = real_adj.reshape((-1, real_adj.shape[1], real_adj.shape[2]))
    mmd_cls = compute_KLD('avg_clustering_dist', generated_adj, real_adj)
    mmd_density = compute_KLD('density', generated_adj, real_adj)
    mmd_connectivity = compute_KLD('avg_node_connectivity', generated_adj, real_adj)
    return mmd_cls, mmd_density, mmd_connectivity


# def disentangle_evaluation(z_s, z_g, z_sg, z_tv, f_s, f_g, f_sg, f_tv, dataset):
#     latent_list = [np.array(z_s), np.array(z_g), np.array(z_sg)]
#     factor_list = [f_s, f_g, f_sg, f_tv]
#     latent_list = [np.tile(np.expand_dims(latent_list[i], axis=2), (1, 1, FLAGS.timestep, 1)) for i in range(3)]
#     latent_list.append(np.array(z_tv))
#     # print (np.array(z_g)[0,0,:,0])
#     MI_matrix(latent_list, factor_list)
#     exit(0)
#     # z_s, z_g, z_sg time invariant
#     # print (np.equal(z_sg,np.array(z_tv).squeeze()))
#     if dataset in ['synthetic2', 'synthetic1']:
#         gold_standard = np.identity(4)
#         pred = np.zeros((4, 4))
#         for i in range(4):
#             for j in range(4):
#                 for k in range(FLAGS.timestep):
#                     pred[i, j] += MI(latent_list[i][:, :, k, :], factor_list[j][:, k])
#                 pred[i, j] = pred[i, j] / FLAGS.timestep
#         plt.imshow(pred)
#         plt.savefig('disentangle_heatmap.png')
#         print(pred)
#         return np.linalg.norm(abs(pred - gold_standard))
#     # if dataset in  ['protein']:
#     #     gold_standard=np.zeros(4)
#     #     gold_standard[1]=1
#     #     pred=np.zeros((4,4))
#     #     for i in range(4):
#     #         for j in range(4):
#     #           for k in range(FLAGS.timestep):
#     #             pred[i,j]=MI(latent_list[i][:,k,:],factor_list[j][:,k,:])
#     if dataset == 'protein':
#         gold_standard = np.zeros(4)
#         gold_standard[1] = 1
#         gold_standard[3] = 0
#         pred = np.zeros(4)
#         for i in range(4):
#             for j in range(FLAGS.timestep):
#                 pred[i] += MI(np.expand_dims(latent_list[i][:, j, :], axis=1), factor_list[1][:, j])
#         pred[i] = pred[i] / FLAGS.timestep
#     if dataset == 'protein':
#         gold_standard = np.zeros(4)
#         gold_standard[1] = 0
#         gold_standard[3] = 1
#         pred_tv = np.zeros(4)
#         for i in range(4):
#             for j in range(FLAGS.timestep):
#                 pred_tv[i] += MI(np.expand_dims(latent_list[i][:, j, :], axis=1), factor_list[3][:, j])
#             pred_tv[i] = pred_tv[i] / FLAGS.timestep
#
#     return np.linalg.norm(abs(pred - gold_standard)), np.linalg.norm(abs(pred_tv - gold_standard))


def load_graph_list(fname):
    with open(fname, "rb") as f:
        graph_list = pickle.load(f)
    return graph_list


def load_combined_graph_list(root_path, graph_name, timestep=8):

    node_size = 0
    generated_graph_list = list()
    for i in range(timestep):
        with open(os.path.join(root_path, 'timestep_' + str(i), graph_name), 'rb') as f:
            temp_graphs = pickle.load(f)
            # print(i, len(temp_graphs))
            temp_adj_list = list()
            for item in temp_graphs:
                temp_adj = np.asarray(nx.to_numpy_matrix(item))
                if node_size == 0:
                    node_size = temp_adj.shape[0]
                temp_adj_list.append(temp_adj)
            temp_adj_list = np.array(temp_adj_list)
            generated_graph_list.append(temp_adj_list)

    #combine graphs into temporal graphs
    whole_graph_list = list()
    for i in range(len(generated_graph_list[0])):

        combined_graph = list()
        for j in range(timestep):
            # combined_graph.append(generated_graph_list[j][i])
            # current_temp_graph = generated_graph_list[j][i]

            current_temp_graph = generated_graph_list[j][i]
            if current_temp_graph.shape[0] > node_size:
                current_temp_graph = current_temp_graph[:node_size, :node_size]
            elif current_temp_graph.shape[0] < node_size:
                padding_width = node_size - current_temp_graph.shape[0]
                current_temp_graph = np.pad(current_temp_graph, ((0, padding_width), (0, padding_width)))
            combined_graph.append(current_temp_graph)

            # print(current_temp_graph)
            # print(type(current_temp_graph))
            # print(current_temp_graph.shape)
            # sys.exit(0)

        combined_graph = np.array(combined_graph)
        # print(combined_graph.shape)
        # if len(combined_graph.shape) < 3:
        #     continue
        whole_graph_list.append(combined_graph)

    return np.array(whole_graph_list)


def load_combined_graph_vae(root_path, graph_name, timestep=8, re=''):

    node_size = 0
    generated_graph_list = list()
    for i in range(timestep):
        with open(os.path.join(root_path, str(i) + re, graph_name), 'rb') as f:

            temp_adj_list = np.load(f)
            node_size = temp_adj_list.shape[2]
            generated_graph_list.append(temp_adj_list)

    #combine graphs into temporal graphs
    whole_graph_list = list()
    for i in range(len(generated_graph_list[0])):

        combined_graph = list()
        for j in range(timestep):
            # combined_graph.append(generated_graph_list[j][i])
            # current_temp_graph = generated_graph_list[j][i]

            current_temp_graph = generated_graph_list[j][i]
            # if current_temp_graph.shape[0] > node_size:
            #     current_temp_graph = current_temp_graph[:node_size, :node_size]
            # elif current_temp_graph.shape[0] < node_size:
            #     padding_width = node_size - current_temp_graph.shape[0]
            #     current_temp_graph = np.pad(current_temp_graph, ((0, padding_width), (0, padding_width)))
            combined_graph.append(current_temp_graph)

            # print(current_temp_graph)
            # print(type(current_temp_graph))
            # print(current_temp_graph.shape)
            # sys.exit(0)

        combined_graph = np.array(combined_graph)
        # print(combined_graph.shape)
        # if len(combined_graph.shape) < 3:
        #     continue
        whole_graph_list.append(combined_graph)

    return np.array(whole_graph_list)


def load_data_syn(type_, mode, path):
    if type_ == 'train':
        adj = np.load(path + '/train/2D_adj_' + mode + '.npy', allow_pickle=True)
        node = np.load(path + '/train/2D_node_' + mode + '.npy', allow_pickle=True) / 120
        spatial = np.load(path + '/train/2D_geometry_' + mode + '.npy', allow_pickle=True) / 600
        rel = np.load(path + '/train/2D_rel_' + mode + '.npy', allow_pickle=True) / 600
        factor = np.load(path + '/train/2D_prop_' + mode + '.npy', allow_pickle=True)
        index = [i for i in range(len(node))]  # randomly shuffle the dataset
        np.random.shuffle(index)
        adj = adj[index]
        node = node[index]
        spatial = spatial[index]
        rel = rel[index]
        factor = factor[index]

    if type_ in ['test_generation', 'test_disentangle', 'test_reconstruct', 'test']:
        adj = np.load(path + '/test/2D_adj_' + mode + '.npy', allow_pickle=True)
        node = np.load(path + '/test/2D_node_' + mode + '.npy', allow_pickle=True) / 120
        spatial = np.load(path + '/test/2D_geometry_' + mode + '.npy', allow_pickle=True) / 600
        rel = np.load(path + '/test/2D_rel_' + mode + '.npy', allow_pickle=True) / 600
        factor = np.load(path + '/train/2D_prop_' + mode + '.npy', allow_pickle=True)
        index = [i for i in range(len(node))]  # randomly shuffle the dataset
        np.random.shuffle(index)
        adj = adj[index]
        node = node[index]
        spatial = spatial[index]
        rel = rel[index]
        factor = factor[index]

    adj_tmp = []
    for n in range(adj.shape[0]):
        adj_time = []
        for k in range(adj.shape[1]):
            adj_time.append(scipy.sparse.csr_matrix.todense(adj[n][k]))
        adj_tmp.append(adj_time)
    adj = np.array(adj_tmp)
    new_adj = []
    for n in range(adj.shape[0]):
        new_adj.append(adj[n])
        for k in range(adj.shape[1]):
            for i in range(adj.shape[2]):
                new_adj[n][k][i, i] = 0
                for j in range(adj.shape[2]):
                    assert new_adj[n][k][i, j] == new_adj[n][k][j, i]

    # new_adj=[]
    # for n in range(len(adj)):
    #    new_adj.append(adj[n])
    #    for i in range(len(new_adj[n])):
    #      new_adj[n][i,i]=0
    #      for j in range(len(new_adj[n])):
    #         assert new_adj[n][i,j]==new_adj[n][j,i] #check whether undirected graph

    # print (factor.shape)
    # exit(0)
    return node, spatial, np.array(new_adj), rel, factor



def main():
    generated_graph_list = []
    real_graph_list = []
    timestep = 8
    test_size = 1000

    dataset_name = 'synthetic_2'
    root_path = '/home/qli10/hengning/models/graph-generation/baselines/graphvae/graphs/' + dataset_name.upper() + '_full'
    graph_name = 'feature.npy'

    generated_feature_list = load_combined_graph_vae(root_path, graph_name, re='_re')

    if 'synthetic' in dataset_name:

        # load synthetic
        real_node, real_spatial, _, _, _ = load_data_syn('test', 'small_constant_more_data', './dataset/raw_datasets/' + dataset_name)

        print(real_node.shape)
        print(real_spatial.shape)

        generated_spatial = generated_feature_list[:, :, :, :2]
        generated_node = generated_feature_list[:, :, :, 2:]

        print(generated_spatial.shape)
        print(generated_node.shape)

        mse_node = mean_squared_error(real_node.reshape(-1), generated_node.reshape(-1))
        mse_spatial = mean_squared_error(real_spatial.reshape(-1), generated_spatial.reshape(-1))

        print(dataset_name)
        print('node', mse_node)
        print('spatial', mse_spatial)

    else:

        # load protein
        root_path = '/home/qli10/hengning/models/graph-generation/dataset/raw_datasets/' + dataset_name + '/'
        # graph_name = '2D_geometry_small_constant_more_data.npy'
        graph_name = 'node_test.npy'
        with open(root_path + graph_name, 'rb') as f:
            real_feature_list = np.load(f)

        print(generated_feature_list.shape, real_feature_list.shape)

        real_feature_list = real_feature_list.reshape(-1, 8, 8, 3)[: len(generated_feature_list)]

        print(generated_feature_list.shape, real_feature_list.shape)

        print(real_feature_list[0, 0])
        print(generated_feature_list[0, 0])
        mse_spatial = mean_squared_error(real_feature_list.reshape(-1), generated_feature_list.reshape(-1))
        mse_spatial = mse_spatial / test_size

        print(dataset_name, mse_spatial)


    sys.exit(0)



    # st_protein synthetic_1
    dataset_name = 'st_protein'
    root_path = '/home/qli10/hengning/models/graph-generation/baselines/graphvae/graphs/' + dataset_name.upper() + '_full'
    graph_name = 'adj.npy'

    generated_graph_list = load_combined_graph_vae(root_path, graph_name)

    root_path = '/home/qli10/hengning/models/graph-generation/dataset/' + dataset_name.upper() + '_full'
    graph_name = 'train.dat'

    real_graph_list = load_combined_graph_list(root_path, graph_name)

    print(generated_graph_list.shape, real_graph_list.shape)

    # rec_result = reconstruct_evaluation_simple(generated_graph_list, real_graph_list)

    # print(dataset_name, rec_result)

    # generated_graph_list = load_combined_graph_list(root_path, graph_name)

    # root_path = '/home/qli10/hengning/models/graph-generation/dataset/' + dataset_name.upper() + '_full'
    # graph_name = 'train.dat'

    # real_graph_list = load_combined_graph_list(root_path, graph_name)

    # print(len(generated_graph_list), len(real_graph_list))

    # real_graph_list = real_graph_list[: len(generated_graph_list)]

    # rec_result = reconstruct_evaluation_simple(generated_graph_list, real_graph_list)

    # print(dataset_name, rec_result)

    # sys.exit(0)


    # synthetic_1 0.5524453125
    # synthetic_2 0.853213671875


    # st_protein synthetic_1 synthetic_2
    # dataset_name = 'synthetic_1'
    # root_path = '/home/qli10/hengning/models/graph-generation/graphs/graphrnn_rnn/' + dataset_name + '_full'
    # graph_name = 'GraphRNN_RNN_st_protein_4_128_pred_3000_1.dat'
    # for i in range(timestep):
    #     with open(os.path.join(root_path, 'timestep_' + str(i), graph_name), 'rb') as f:
    #         temp_graphs = pickle.load(f)
    #         generated_graph_list += temp_graphs

    # root_path = '/home/qli10/hengning/models/graph-generation/dataset/' + dataset_name.upper() + '_full'
    # graph_name = 'train.dat'
    # for i in range(timestep):
    #     temp_graphs = load_graph_list(os.path.join(root_path, 'timestep_' + str(i), 'train.dat'))
    #     real_graph_list += temp_graphs

    generated_graph_list = generated_graph_list.reshape((-1, generated_graph_list.shape[2], generated_graph_list.shape[2]))
    real_graph_list = real_graph_list.reshape((-1, real_graph_list.shape[2], real_graph_list.shape[2]))

    temp_index = [i for i in range(len(real_graph_list))]
    np.random.shuffle(temp_index)
    real_graph_list = [real_graph_list[i] for i in temp_index]

    temp_index = [i for i in range(len(generated_graph_list))]
    np.random.shuffle(temp_index)
    generated_graph_list = [generated_graph_list[i] for i in temp_index]

    if test_size == -1:
        min_len = min(len(real_graph_list), len(generated_graph_list))
        real_graph_list = real_graph_list[: min_len]
        generated_graph_list = generated_graph_list[: min_len]
    else:
        real_graph_list = real_graph_list[: test_size*timestep]
        generated_graph_list = generated_graph_list[: test_size*timestep]

    print(len(real_graph_list), len(generated_graph_list))

    # round1
    # rnn protein test_size=1000s
    # avg_clustering_dist 1.0452435165946623
    # density 1.1547208534368776
    # avg_node_connectivity 2.0553869403589604

    # rnn synthetic_1 test_size=100
    # avg_clustering_dist 1.9665763840622126
    # density 2.5017898253812407
    # avg_node_connectivity 0.5145501284777848

    # rnn synthetic_2 test_size=100
    # avg_clustering_dist 0.5748978020860074
    # density 1.2421273892357148
    # avg_node_connectivity 2.8806394550942627

    # round2
    # rnn protein test_size=-1
    # avg_clustering_dist 1.00
    # density 1.16
    # avg_node_connectivity 

    # rnn synthetic_1 test_size=-1
    # avg_clustering_dist 2.70
    # density 3.06
    # avg_node_connectivity 

    # rnn synthetic_2 test_size=-1
    # avg_clustering_dist 0.50
    # density 1.55
    # avg_node_connectivity 

    c_metric = 'avg_clustering_dist'
    result1 = compute_KLD(c_metric, generated_graph_list, real_graph_list)
    print(root_path)
    print(c_metric)
    print('test_size', test_size)
    print(result1)

    c_metric = 'density'
    result2 = compute_KLD(c_metric, generated_graph_list, real_graph_list)
    print(root_path)
    print(c_metric)
    print('test_size', test_size)
    print(result2)

    c_metric = 'avg_node_connectivity'
    result3 = compute_KLD(c_metric, generated_graph_list, real_graph_list)
    print(root_path)
    print(c_metric)
    print('test_size', test_size)
    print(result3)

    # def reconstruct_evaluation(generated_adj, generated_nodes, generated_spatial, real_adj, real_nodes, real_spatial,
    #                            dataset):

    # generated_graph_adj = []
    # real_graph_adj = []
    #
    # for item in generated_graph_list:
    #     temp_adj = np.asarray(nx.to_numpy_matrix(item))
    #     generated_graph_adj.append(temp_adj)
    #
    # for item in sample(real_graph_list, len(generated_graph_list)):
    #     temp_adj = np.asarray(nx.to_numpy_matrix(item))
    #     real_graph_adj.append(temp_adj)
    #
    # generated_graph_adj = np.array(generated_graph_adj)
    # real_graph_adj = np.array(real_graph_adj)
    #
    # mse_node = reconstruct_evaluation_simple(generated_graph_adj, real_graph_adj)
    # print(mse_node)


if __name__ == '__main__':
    main()
