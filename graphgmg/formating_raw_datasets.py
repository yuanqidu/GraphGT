import numpy as np
import networkx as nx
import pickle
import os
import scipy


def cal_rel_dist(coords):
    rel = np.ones(shape=(coords.shape[0], coords.shape[1], coords.shape[1]), dtype=float)
    for i in range(coords.shape[0]):
        for j in range(coords.shape[1]):
            for k in range(coords.shape[1]):
                rel[i][j][k] = ((coords[i][j][0] - coords[i][k][0]) ** 2 + (coords[i][j][1] - coords[i][k][1]) ** 2 + (
                            coords[i][j][2] - coords[i][k][2]) ** 2) ** .5
    return rel


def save_graph_list(G_list, fname):
    with open(fname, "wb") as f:
        pickle.dump(G_list, f)


def save_graph_dataset(root_path, name, graphs_list):

    dataset_dir = os.path.join(root_path, name.upper() + '_full')
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)
    # assert os.path.exists(dataset_dir)
    for index, item in enumerate(graphs_list):
        file_dir = os.path.join(dataset_dir, 'timestep_' + str(index))
        os.mkdir(file_dir)
        save_graph_list(item[0], os.path.join(file_dir, 'train.dat'))
        save_graph_list(item[1], os.path.join(file_dir, 'test.dat'))

    print("done")


def formatting_graph(train_adj, train_spatial, test_adj, test_spatial, timestep):

    graphs_list = list()
    for i in range(timestep):
        train_adj_slide = train_adj[:, i, :, :]
        train_spatial_slide = train_spatial[:, i, :, :]

        test_adj_slide = test_adj[:, i, :, :]
        test_spatial_slide = test_spatial[:, i, :, :]

        train_graphs = list()
        for item in zip(train_adj_slide, train_spatial_slide):
            temp_graph = nx.from_numpy_matrix(item[0])

            for j, spatial_info in enumerate(item[1]):
                temp_graph.node[j]['feature'] = spatial_info

            train_graphs.append(temp_graph)

        test_graphs = list()
        for item in zip(test_adj_slide, test_spatial_slide):
            temp_graph = nx.from_numpy_matrix(item[0])

            for j, spatial_info in enumerate(item[1]):
                temp_graph.node[j]['feature'] = spatial_info

            test_graphs.append(temp_graph)

        graphs_list.append((train_graphs, test_graphs))

    return graphs_list


def formatting_graph_flatten(train_adj, train_spatial, test_adj, test_spatial):

    graphs_list = list()

    # train_adj_slide = train_adj.reshape(-1 ,train_adj.shape[2], train_adj.shape[3])
    # train_spatial_slide = train_spatial.reshape(-1 ,train_spatial.shape[2], train_spatial.shape[3])

    # test_adj_slide = test_adj.reshape(-1 ,test_adj.shape[2], test_adj.shape[3])
    # test_spatial_slide = test_spatial.reshape(-1 ,test_spatial.shape[2], test_spatial.shape[3])

    # print (train_adj_slide.shape, train_spatial_slide.shape, test_adj_slide.shape, test_spatial_slide.shape)

    train_graphs = list()
    for item in zip(train_adj, train_spatial):
        temp_graph = nx.from_numpy_matrix(item[0])

        for j, spatial_info in enumerate(item[1]):
            temp_graph.nodes[j]['feature'] = spatial_info

        train_graphs.append(temp_graph)

    test_graphs = list()
    for item in zip(test_adj, test_spatial):
        temp_graph = nx.from_numpy_matrix(item[0])

        for j, spatial_info in enumerate(item[1]):
            temp_graph.nodes[j]['feature'] = spatial_info

        test_graphs.append(temp_graph)

    graphs_list.append((train_graphs, test_graphs))

    return graphs_list



def generate_dataset_synthetic(root_path, name):

    train_node, train_spatial, train_adj = load_data_syn(root_path)
    total_len = train_adj.shape[0]
    split = (total_len//10)*7
    test_adj = train_adj[split:].squeeze()
    test_spatial = train_spatial[split:]
    test_node = train_node[split:]
    train_adj = train_adj[:split].squeeze()
    train_spatial = train_spatial[:split]
    train_node = train_node[:split]

    print (train_node.shape,train_spatial.shape,train_adj.shape,test_node.shape,test_spatial.shape,test_adj.shape)

    train_spatial = np.concatenate((train_spatial, train_node), axis=2)
    test_spatial = np.concatenate((test_spatial, test_node), axis=2)

    graphs_list = formatting_graph_flatten(train_adj, train_spatial, test_adj, test_spatial)
    save_graph_dataset(root_path, name, graphs_list)


def load_data_syn(path):
    adj = np.load(path + '/adj.npy', allow_pickle=True)
    node = np.load(path + '/node_feat.npy', allow_pickle=True)
    spatial = np.load(path + '/spatial.npy', allow_pickle=True)
    index = [i for i in range(len(node))]  # randomly shuffle the dataset
    np.random.shuffle(index)
    adj = adj[index]
    node = node[index]
    spatial = spatial[index]

    return node, spatial, adj


def generate_dataset_protein(root_path):
    ROOT_PATH = root_path

    train_adj = np.load(os.path.join(ROOT_PATH, 'edge_feat.npy'))
    train_spatial = np.load(os.path.join(ROOT_PATH, 'node_feat.npy'))

    total_len = train_adj.shape[0]
    split = (total_len//10)*7
    test_adj = train_adj[split:].squeeze()
    test_spatial = train_spatial[split:]
    train_adj = train_adj[:split].squeeze()
    train_spatial = train_spatial[:split]
    print (train_adj.shape, train_spatial.shape, test_adj.shape, test_spatial.shape)

    graphs_list = formatting_graph_flatten(train_adj, train_spatial, test_adj, test_spatial)

    save_graph_dataset(root_path, 'ST_PROTEIN', graphs_list)


if __name__ == '__main__':
    # root_path = '../profold'
    # generate_dataset_protein(root_path)
    # root_path = '../waxman'
    # generate_dataset_synthetic(root_path, name='waxman')
    root_path = '../random_geometry'
    generate_dataset_synthetic(root_path, name='random_geometry')
