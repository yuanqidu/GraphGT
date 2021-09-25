import networkx as nx
import numpy as np
import math
import os
import sys
from tqdm import tqdm
import eval.stats
import pickle
import scipy.io as scio
import scipy.stats
import scipy.sparse
from queue import Queue


def load_data_protein(path):
    time_step = 8
    adj = np.load(path + '/edge_train.npy')
    spatial = np.load(path + '/node_train.npy')
    node = np.ones(shape=(spatial.shape[0], spatial.shape[1]), dtype=int)
    # rel = cal_rel_dist(spatial)
    factor = np.array(range(1, 1001)).reshape((1, 1000))
    factor = np.tile(factor, 38).reshape(-1)
    factor = np.reshape(factor, (factor.shape[0] // time_step, time_step))
    node = np.reshape(node, (node.shape[0] // time_step, time_step, node.shape[1]))
    spatial = np.reshape(spatial,
                         (spatial.shape[0] // time_step, time_step, spatial.shape[1], spatial.shape[-1]))
    # rel = np.reshape(rel, (rel.shape[0] // time_step, time_step, rel.shape[1], rel.shape[1]))
    adj = np.reshape(adj, (adj.shape[0] // time_step, time_step, adj.shape[1], adj.shape[1]))
    index = [i for i in range(len(node))]  # randomly shuffle the dataset
    np.random.shuffle(index)
    adj = adj[index]
    node = node[index]
    spatial = spatial[index]
    # rel = rel[index]
    factor = factor[index]

    new_adj = []
    for n in range(adj.shape[0]):
        new_adj.append(adj[n])
        for k in range(adj.shape[1]):
            for i in range(adj.shape[2]):
                new_adj[n][k][i, i] = 0
                for j in range(len(new_adj[n][k])):
                    assert new_adj[n][k][i, j] == new_adj[n][k][j, i]  # check whether undirected graph
    return node, spatial, np.array(new_adj, dtype=int), factor


def temporal_correlation(graph):
    a = 1
    """
    calculate temporal correlation of a dynamic network
    Input:  n * 3 matrix encoding contacts between node i, j and time t by (i, j, t)
            the graph is undirected
    Output: C = temporal correlation coefficient of the network
            C_vec = temporal correlation of each node
    """

    # print(graph.shape)
    # print(graph)

    time_size = graph.shape[0]
    node_size = graph.shape[1]

    C_vec = [0.0] * node_size
    C = 0

    for i in range(node_size):

        C_i = 0.0

        for t in range(1, time_size):
            den = np.dot(graph[t, i, :], graph[t - 1, i, :])
            num = math.sqrt(np.sum(graph[t, i, :]) * np.sum(graph[t - 1, i, :]))

            C_i += den / num

        C_i /= (time_size - 1)
        C_vec[i] = C_i

    C = sum(C_vec)

    return C, C_vec


def forward_latency(graph, time, start, end):
    time_len = len(graph)

    def fl_helper(t, x, y):
        # print(t, x, y)
        a = 1
        if t >= time_len:
            return 10000

        min_value = 10000
        for index, item in enumerate(graph[t, x, :]):
            if item <= 0:
                continue
            if index == y:
                return 1
            c_result = fl_helper(t + 1, index, y) + 1
            if c_result < min_value:
                min_value = c_result

        return min_value

    return fl_helper(time, start, end)


def closeness_centrality(graph, time, i):
    result = 0
    for j in range(len(graph[0][0])):
        if j == i:
            continue
        result += 1.0 / forward_latency(graph, time, i, j)

    result /= len(graph[0]) - 1

    return result


def shortest_path(graph, time, start, end, target):
    time_len = len(graph)

    sp_q = Queue()
    sp_q.put([0, start, False])

    result = list()

    end_value = -1
    while not sp_q.empty():

        top_item = sp_q.get()
        c_time_stamp = top_item[0] + time
        c_x = top_item[1]
        c_if_cross = top_item[2]

        if 0 <= end_value < c_time_stamp:
            break

        for index, value in enumerate(graph[c_time_stamp, c_x, :]):
            if value <= 0:
                continue
            if index == end:
                result.append([top_item[0] + 1, c_if_cross])
                end_value = c_time_stamp
                continue

            temp_if_cross = c_if_cross
            if index == target:
                temp_if_cross = True

            if top_item[0] + 1 + time < time_len:
                sp_q.put([top_item[0] + 1, index, temp_if_cross])

    n_sp = len(result)
    n_sp_target = 0

    for item in result:
        if item[1]:
            n_sp_target += 1

    return n_sp, n_sp_target


def betweenness_centrality(graph, time, i):
    result = 0.0
    node_size = len(graph[0])

    for j in range(node_size):
        if j == i:
            continue

        for k in range(j + 1, node_size):
            if k == i:
                continue

            n_sp, n_sp_target = shortest_path(graph, time, j, k, i)
            if n_sp > 0:
                result += n_sp_target / n_sp

    return result


def make_discretizer(target, num_bins=2):
    """Wrapper that creates discretizers."""
    return np.digitize(target, np.histogram(target, num_bins)[1][:-1])


def compute_kld_(generated, real):
    generated = [i / generated.count(i) for i in generated]
    real = [i / real.count(i) for i in real]

    return scipy.stats.entropy(generated, real)


def compute_KLD_from_graph(generated_graph_list, real_graph_list):
    generated_discrete = make_discretizer(generated_graph_list, num_bins=5)
    real_discrete = make_discretizer(real_graph_list, num_bins=5)
    kld = compute_kld_(list(generated_discrete), list(real_discrete))
    return kld


def load_mat(fpath):
    timestep = 8
    raw_data = scio.loadmat(fpath)
    adj_key_name = 'adjDsbm_fake_graphs'
    raw_adj = raw_data[adj_key_name]

    raw_adj = raw_adj.reshape((-1, timestep, raw_adj.shape[0], raw_adj.shape[1]))

    return raw_adj


def compute_kld(metric_name, generate_graph_list, real_graph_list, test_size=800):
    generate_graph_list = generate_graph_list[: test_size]
    real_graph_list = real_graph_list[: test_size]

    fake = list()
    real = list()
    if metric_name == 'degree':
        return eval.stats.degree_stats(generate_graph_list,real_graph_list)
    elif metric_name == 'cluster':
        return eval.stats.clustering_stats(generate_graph_list,real_graph_list)
    elif metric_name == 'orbit':
        return eval.stats.orbit_stats_all(generate_graph_list,real_graph_list)
    # if metric_name == 'degree':
    #     for item in generate_graph_list:
    #         deg = eval.stats.degree_stats(item)
    #         fake.append(deg)
    #     for item in real_graph_list:
    #         deg = eval.stats.degree_stats(item)
    #         real.append(deg)
    # elif metric_name == 'cluster':
    #     for item in generate_graph_list:
    #         deg = eval.stats.clustering_stats(item)
    #         fake.append(deg)
    #     for item in real_graph_list:
    #         deg = eval.stats.clustering_stats(item)
    #         real.append(deg)
    # elif metric_name == 'orbit':
    #     for item in generate_graph_list:
    #         deg = eval.stats.orbit_stats_all(item)
    #         fake.append(deg)
    #     for item in real_graph_list:
    #         deg = eval.stats.orbit_stats_all(item)
    #         real.append(deg)

    # if metric_name == 'temporal':
    #     for item in generate_graph_list:
    #         tc, tc_vec = temporal_correlation(item)
    #         fake.append(tc)
    #     fake = np.nan_to_num(fake, nan=0)
    #     for item in real_graph_list:
    #         tc, tc_vec = temporal_correlation(item)
    #         real.append(tc)
    #     real = real[:len(fake)]
    #     real = np.nan_to_num(real, nan=0)
    # elif metric_name == 'between':
    #     for item in tqdm(generate_graph_list):
    #         graph_sum = 0
    #         for i in range(item.shape[0] - 1):
    #             for j in range(item.shape[1]):
    #                 temp_closeness = betweenness_centrality(item, i, j)
    #                 graph_sum += temp_closeness
    #         fake.append(graph_sum)

    #     for item in tqdm(real_graph_list):
    #         graph_sum = 0
    #         for i in range(item.shape[0] - 1):
    #             for j in range(item.shape[1]):
    #                 temp_closeness = betweenness_centrality(item, i, j)
    #                 graph_sum += temp_closeness
    #         real.append(graph_sum)
    # elif metric_name == 'close':
    #     for item in tqdm(generate_graph_list):
    #         graph_sum = 0
    #         for j in range(item.shape[1]):
    #             temp_closeness = closeness_centrality(item, 0, j)
    #             graph_sum += temp_closeness
    #
    #         fake.append(graph_sum)
    #
    #     for item in tqdm(real_graph_list[:len(fake)]):
    #         graph_sum = 0
    #         for j in range(item.shape[1]):
    #             temp_closeness = closeness_centrality(item, 0, j)
    #             graph_sum += temp_closeness
    #
    #         real.append(graph_sum)

    result = compute_KLD_from_graph(fake, real)
    print(metric_name, result)

    return result


def load_data_synthetic(load_path):
    p_data = np.load(load_path, allow_pickle=True)
    p_data_tmp = list()
    for n in range(p_data.shape[0]):
        adj_time = []
        for k in range(p_data.shape[1]):
            adj_time.append(scipy.sparse.csr_matrix.todense(p_data[n][k]))
        p_data_tmp.append(adj_time)
    p_data = np.array(p_data_tmp)

    return p_data


def compute_kld_all(generate_graph_list, real_graph_list, test_size=800):
    result = list()

    result.append(('degree', compute_kld('degree', generate_graph_list, real_graph_list, test_size)))
    print (result)
    result.append(('cluster', compute_kld('cluster', generate_graph_list, real_graph_list, test_size)))
    print (result)
    result.append(('orbit', compute_kld('orbit', generate_graph_list, real_graph_list, test_size)))
    print(result)

    return result


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


if __name__ == '__main__':
    PATH = './graphs/graphrnn_rnn/traffic_bay/GraphRNN_RNN_traffic_bay_4_128_pred_1000_1.dat'
    PATH = '../graphgmg/graphs/baseline_dgmg/n_body_charged/Baseline_DGMG_n_body_charged_64_pred_1.dat'
    PATH = '../baselines/graphvae/pred_n_body_spring.dat'
    generate_graph_list = list()
    generate_graph_list_graph = list()
    with open(os.path.join(PATH), 'rb') as f:
        generate_graphs = pickle.load(f)
        adj_list = list()
        for item in generate_graphs:
            item.remove_edges_from(nx.selfloop_edges(item))
            generate_graph_list_graph.append(item)
            adj = np.asarray(nx.to_numpy_matrix(item))
            # print (adj)
            adj_list.append(adj)
        adj_list = np.array(adj_list)
        generate_graph_list = adj_list
    print(generate_graph_list.shape)

    count = 0
    PATH = '../n_body_spring/N_BODY_SPRING_full/timestep_0/train.dat'
    # PATH = '../baselines/graphvae/profold.dat'
    train_graph_list = list()
    train_graph_list_graph = list()
    with open(os.path.join(PATH), 'rb') as f:
        train_graphs = pickle.load(f)
        adj_list = list()
        for item in train_graphs:
            item.remove_edges_from(nx.selfloop_edges(item))
            train_graph_list_graph.append(item)
            adj = np.asarray(nx.to_numpy_matrix(item))
            np.fill_diagonal(adj, 0)
            # adj[adj>0] = 1
            # print (adj)
            adj_list.append(adj)
        adj_list = np.array(adj_list)
        train_graph_list = adj_list
    print(train_graph_list.shape)
    # train_graph_list_graph
    compute_kld_all(generate_graph_list_graph,train_graph_list_graph,test_size=len(generate_graph_list_graph))
    # compute_kld_all(generate_graph_list, train_graph_list, test_size=generate_graph_list.shape[0])
    exit(0)

    # load generated graph
    dataset_name = 'synthetic_1'
    generated_graph_list = list()
    root_path = '/home/qli10/hengning/models/graph-generation/graphs/graphrnn_rnn/' + dataset_name + '_full'
    graph_name = 'GraphRNN_RNN_st_protein_4_128_pred_3000_1.dat'
    for i in range(timestep):
        with open(os.path.join(root_path, 'timestep_' + str(i), graph_name), 'rb') as f:
            temp_graphs = pickle.load(f)
            # print(i, len(temp_graphs))
            temp_adj_list = list()
            for item in temp_graphs:
                temp_adj = np.asarray(nx.to_numpy_matrix(item))
                temp_adj_list.append(temp_adj)
            temp_adj_list = np.array(temp_adj_list)
            generated_graph_list.append(temp_adj_list)

    # combine graphs into temporal graphs
    whole_graph_list = list()
    for i in range(len(generated_graph_list[0])):

        combined_graph = list()
        for j in range(timestep):
            # combined_graph.append(generated_graph_list[j][i])
            # current_temp_graph = generated_graph_list[j][i]

            current_temp_graph = generated_graph_list[j][i]
            if current_temp_graph.shape[0] > 25:
                current_temp_graph = current_temp_graph[:25, :25]
            elif current_temp_graph.shape[0] < 25:
                padding_width = 25 - current_temp_graph.shape[0]
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

    print('real graph list size', len(p_data))
    print('generated graph list size', len(whole_graph_list))

    exit(0)

    dataset_name = 'synthetic_2'
    root_path = '/home/qli10/hengning/models/graph-generation/baselines/graphvae/graphs/' + dataset_name.upper() + '_full'
    graph_name = 'adj.npy'

    generated_graph_list = load_combined_graph_vae(root_path, graph_name)

    root_path = '/home/qli10/hengning/models/graph-generation/dataset/' + dataset_name.upper() + '_full'
    graph_name = 'train.dat'

    real_graph_list = load_combined_graph_list(root_path, graph_name)

    print(generated_graph_list.shape, real_graph_list.shape)

    total = 10
    test_size = 10
    for i in range(total):
        temp_index = [i for i in range(len(real_graph_list))]
        np.random.shuffle(temp_index)
        real_graph_list = [real_graph_list[i] for i in temp_index]

        temp_index = [i for i in range(len(generated_graph_list))]
        np.random.shuffle(temp_index)
        generated_graph_list = [generated_graph_list[i] for i in temp_index]

        # compute_kld('temporal', generated_graph_list, real_graph_list, test_size=test_size)

        compute_kld_all(generated_graph_list, real_graph_list, test_size=test_size)


    sys.exit(0)

    # protein
    # [('temporal', 1.9048786032203253), ('between', 1.4801613289813784)]
    # s1
    # [('temporal', 0.8459702573789273), ('between', 0.9071965267796737)]
    # s2
    # [('temporal', 3.077644886216146), ('between', 1.602028821327814)]


    # load raw protein
    # PATH = './dataset/raw_datasets/protein'
    # p_data = load_data_protein(PATH)[2]

    # load raw synthethic
    PATH = './dataset/raw_datasets/synthetic_1/train/2D_adj_small_constant_more_data.npy'
    p_data = np.load(PATH, allow_pickle=True)

    p_data_tmp = list()
    for n in range(p_data.shape[0]):
        adj_time = []
        for k in range(p_data.shape[1]):
            adj_time.append(scipy.sparse.csr_matrix.todense(p_data[n][k]))
        p_data_tmp.append(adj_time)
    p_data = np.array(p_data_tmp)

    a = 1

    # load generated graph
    timestep = 8
    dataset_name = 'synthetic_1'
    generated_graph_list = list()
    root_path = '/home/qli10/hengning/models/graph-generation/graphs/graphrnn_rnn/' + dataset_name + '_full'
    graph_name = 'GraphRNN_RNN_st_protein_4_128_pred_3000_1.dat'
    for i in range(timestep):
        with open(os.path.join(root_path, 'timestep_' + str(i), graph_name), 'rb') as f:
            temp_graphs = pickle.load(f)
            # print(i, len(temp_graphs))
            temp_adj_list = list()
            for item in temp_graphs:
                temp_adj = np.asarray(nx.to_numpy_matrix(item))
                temp_adj_list.append(temp_adj)
            temp_adj_list = np.array(temp_adj_list)
            generated_graph_list.append(temp_adj_list)

    # combine graphs into temporal graphs
    whole_graph_list = list()
    for i in range(len(generated_graph_list[0])):

        combined_graph = list()
        for j in range(timestep):
            # combined_graph.append(generated_graph_list[j][i])
            # current_temp_graph = generated_graph_list[j][i]

            current_temp_graph = generated_graph_list[j][i]
            if current_temp_graph.shape[0] > 25:
                current_temp_graph = current_temp_graph[:25, :25]
            elif current_temp_graph.shape[0] < 25:
                padding_width = 25 - current_temp_graph.shape[0]
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

    print('real graph list size', len(p_data))
    print('generated graph list size', len(whole_graph_list))

    # t_data = p_data[5]
    # temp1 = [[1, 2, 0, 1], [2, 1, 3, 4], [0, 3, 1, 0], [1, 4, 0, 1]]
    # temp2 = [[1, 2, 0, 1], [2, 1, 3, 4], [0, 3, 1, 2], [1, 4, 2, 1]]
    #
    # t_data = np.array([temp1, temp2])

    # temp1 = [[1, 2, 1], [0, 3, 2], [1, 2, 1]]
    # temp2 = [[1, 2, 1], [1, 1, 2], [1, 2, 1]]
    #
    # t_data = np.array([temp1, temp2])

    # print(t_data.shape)

    # cc = betweenness_centrality(t_data, 0, 1)
    # print(cc)

    # temporal correlation
    # fake = list()
    # for item in whole_graph_list:
    #     # if len(item.shape) < 3:
    #     #     continue
    #     tc, tc_vec = temporal_correlation(item)
    #     fake.append(tc)
    # fake = np.nan_to_num(fake, nan=0)
    #
    # print('fake len', len(fake))
    # real = list()
    # for item in p_data:
    #     # print(item.shape)
    #     tc, tc_vec = temporal_correlation(item)
    #     real.append(tc)
    # real = real[:len(fake)]
    # real = np.nan_to_num(real, nan=0)
    #
    # result = compute_KLD_from_graph(fake, real)
    # print(result)

    # fake = list()
    # for item in whole_graph_list:
    #     graph_sum = 0
    #     for i in range(item.shape[0] - 1):
    #         for j in range(item.shape[1]):
    #             temp_closeness = betweenness_centrality(item, i, j)
    #             graph_sum += temp_closeness
    #     fake.append(graph_sum)
    #
    # real = list()
    # for item in p_data[:len(whole_graph_list)]:
    #     graph_sum = 0
    #     for i in range(item.shape[0] - 1):
    #         for j in range(item.shape[1]):
    #             temp_closeness = betweenness_centrality(item, i, j)
    #             graph_sum += temp_closeness
    #     real.append(graph_sum)
    #
    # result = compute_KLD_from_graph(fake, real)
    # print(result)

    # betweenness centrality

    fake = list()
    for item in tqdm(whole_graph_list):
        graph_sum = 0

        # for i in range(item.shape[0] - 1):
        #     for j in range(item.shape[1]):
        #         temp_closeness = betweenness_centrality(item, i, j)
        #         graph_sum += temp_closeness

        for j in range(item.shape[1]):
            temp_closeness = betweenness_centrality(item, 0, j)
            graph_sum += temp_closeness

        fake.append(graph_sum)

    print(fake)

    real = list()
    for item in tqdm(p_data[:len(fake)]):
        graph_sum = 0

        # for i in range(item.shape[0] - 1):
        #     for j in range(item.shape[1]):
        #         temp_closeness = betweenness_centrality(item, i, j)
        #         graph_sum += temp_closeness

        for j in range(item.shape[1]):
            temp_closeness = betweenness_centrality(item, 0, j)
            graph_sum += temp_closeness

        real.append(graph_sum)

    print(real)

    # closeness centrality
    # fake = list()
    # for item in tqdm(whole_graph_list):
    #     if len(item.shape) < 3:
    #         continue
    #     graph_sum = 0

    #     for j in range(item.shape[1]):
    #         temp_closeness = closeness_centrality(item, 0, j)
    #         graph_sum += temp_closeness

    #     fake.append(graph_sum)

    # real = list()
    # for item in tqdm(p_data[:len(fake)]):
    #     graph_sum = 0

    #     for j in range(item.shape[1]):
    #         temp_closeness = closeness_centrality(item, 0, j)
    #         graph_sum += temp_closeness

    #     real.append(graph_sum)

    result = compute_KLD_from_graph(fake, real)
    print(result)

    # protein
    # temporal 0.46594444312521444
    # closeness 0.48813692603010495

    # synthetic_1
    # temporal 0.7930774319290594
    # between 0.9245186297092411
