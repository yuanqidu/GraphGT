import numpy as np
import networkx as nx
import pickle
import os
import json
import scipy

from utils import *
from data import *


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

def save_graph_dataset_single(root_path, name, graphs_list):

    dataset_dir = os.path.join(root_path, name.upper() + '_full')
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)
    # assert os.path.exists(dataset_dir)
    # for index, item in enumerate(graphs_list):
    #     print (index)
    #     file_dir = os.path.join(dataset_dir, 'timestep_' + str(index))
    #     if not os.path.exists(file_dir):
    #         os.mkdir(file_dir)
    file_dir = os.path.join(dataset_dir, 'timestep_' + str(0))
    save_graph_list(graphs_list, os.path.join(file_dir, 'train.dat'))

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
    print (adj[0],adj[1])
    node = np.load(path + '/node_feat.npy', allow_pickle=True)
    spatial = np.load(path + '/spatial.npy', allow_pickle=True)
    print (adj.shape,node.shape,spatial.shape)
    index = [i for i in range(len(node))]  # randomly shuffle the dataset
    np.random.shuffle(index)
    adj = adj[index]
    node = node[index]
    spatial = spatial[index]

    return node, spatial, adj

def load_data_pointcloud(path):
    adj = np.load(path + '/adj.npy', allow_pickle=True)
    print (adj.shape)
    exit(0)
    node = np.load(path + '/node_feat.npy', allow_pickle=True)
    spatial = np.load(path + '/spatial.npy', allow_pickle=True)
    print (adj.shape,node.shape,spatial.shape)
    index = [i for i in range(len(node))]  # randomly shuffle the dataset
    np.random.shuffle(index)
    adj = adj[index]
    node = node[index]
    spatial = spatial[index]

    return node, spatial, adj

def generate_dataset_pointcloud(root_path, name):
    ROOT_PATH = root_path

    train_spatial = np.load(os.path.join(ROOT_PATH, 'spatial.npy'))[:,:3].reshape(-1, 25, 3)
    train_adj = np.expand_dims(np.load(os.path.join(ROOT_PATH, 'adj.npy')),axis=0)
    train_adj = np.tile(train_adj, (train_spatial.shape[0],1,1))
    print (train_adj.shape, train_spatial.shape)

    total_len = train_adj.shape[0]
    split = (total_len//10)*7
    test_adj = train_adj[split:].squeeze()
    test_spatial = train_spatial[split:]
    train_adj = train_adj[:split].squeeze()
    train_spatial = train_spatial[:split]
    print (train_adj.shape, train_spatial.shape, test_adj.shape, test_spatial.shape)

    graphs_list = formatting_graph_flatten(train_adj, train_spatial, test_adj, test_spatial)

    save_graph_dataset(root_path, name, graphs_list)

def generate_dataset_protein_enzyme(root_path, name):
    graphs = Graph_load_batch(min_num_nodes=10, name='ENZYMES')
    print (graphs)
    # print (graphs)
    save_graph_dataset_single(root_path, name, graphs)
#     ROOT_PATH = root_path

#     train_node = np.load(os.path.join(ROOT_PATH, 'node_feat.npy'),allow_pickle=True)#[:,:3].reshape(-1, 25, 3)
#     print (train_node.shape)
#     train_adj = np.expand_dims(np.load(os.path.join(ROOT_PATH, 'adj.npy'),allow_pickle=True),axis=0)
#     train_adj = np.tile(train_adj, (train_node.shape[0],1,1))
#     print (train_adj.shape, train_spatial.shape)
#     exit(0)

#     total_len = train_adj.shape[0]
#     split = (total_len//10)*7
#     test_adj = train_adj[split:].squeeze()
#     test_spatial = train_spatial[split:]
#     train_adj = train_adj[:split].squeeze()
#     train_spatial = train_spatial[:split]
#     print (train_adj.shape, train_spatial.shape, test_adj.shape, test_spatial.shape)

#     graphs_list = formatting_graph_flatten(train_adj, train_spatial, test_adj, test_spatial)

#     save_graph_dataset(root_path, name, graphs_list)

# def generate_dataset_pointcloud_kinetics(root_path, name):
#     ROOT_PATH = root_path

#     train_spatial = np.load(os.path.join(ROOT_PATH, 'spatial.npy'))[:,:3].reshape(-1, 18, 2)
#     train_adj = np.expand_dims(np.load(os.path.join(ROOT_PATH, 'adj.npy')),axis=0)
#     train_node = np.load(os.path.join(ROOT_PATH, 'node_feat.npy'))[:,:3].reshape(-1, 18, 1)
#     train_adj = np.tile(train_adj, (train_spatial.shape[0],1,1))
#     print (train_adj.shape, train_spatial.shape, train_node.shape)

#     total_len = train_adj.shape[0]
#     split = (total_len//10)*7
#     test_adj = train_adj[split:].squeeze()
#     test_spatial = train_spatial[split:]
#     test_node = train_node[split:]
#     train_node = train_node[:split]
#     train_adj = train_adj[:split].squeeze()
#     train_spatial = train_spatial[:split]

#     train_spatial = np.concatenate((train_spatial, train_node), axis=2)
#     test_spatial = np.concatenate((test_spatial, test_node), axis=2)
    
#     print (train_adj.shape, train_spatial.shape, test_adj.shape, test_spatial.shape)

    # graphs_list = formatting_graph_flatten(train_adj, train_spatial, test_adj, test_spatial)

    

def generate_dataset_traffic(root_path, name):
    ROOT_PATH = root_path

    train_node = np.load(os.path.join(ROOT_PATH, 'node_feat.npy'))
    print (train_node.shape)
    train_spatial = np.expand_dims(np.load(os.path.join(ROOT_PATH, 'spatial.npy')),axis=0)
    train_spatial = np.tile(train_spatial, (train_node.shape[0],1,1))
    print (train_spatial.shape)
    train_adj = np.expand_dims(np.load(os.path.join(ROOT_PATH, 'adj.npy')),axis=0)
    train_adj = np.tile(train_adj, (train_spatial.shape[0],1,1))
    print (train_adj.shape, train_spatial.shape, train_node.shape)

    total_len = train_adj.shape[0]
    split = (total_len//10)*7
    test_adj = train_adj[split:].squeeze()
    test_spatial = train_spatial[split:]
    test_node = train_node[split:]
    train_node = train_node[:split]
    train_adj = train_adj[:split].squeeze()
    train_spatial = train_spatial[:split]

    train_spatial = np.concatenate((train_spatial, train_node), axis=2)
    test_spatial = np.concatenate((test_spatial, test_node), axis=2)
    
    print (train_adj.shape, train_spatial.shape, test_adj.shape, test_spatial.shape)

    graphs_list = formatting_graph_flatten(train_adj, train_spatial, test_adj, test_spatial)

    save_graph_dataset(root_path, name, graphs_list)

def generate_dataset_n_body(root_path, name):
    ROOT_PATH = root_path

    train_adj = np.load(os.path.join(ROOT_PATH, 'adj.npy'))
    train_spatial = np.load(os.path.join(ROOT_PATH, 'spatial.npy'))
    train_node = np.load(os.path.join(ROOT_PATH, 'node_feat.npy'))
    print (train_adj.shape, train_spatial.shape, train_node.shape)

    total_len = train_adj.shape[0]
    split = int(total_len * 0.1)
    split2 = int(total_len * 0.11)
    test_adj = train_adj[split:split2].squeeze()
    test_spatial = train_spatial[split:split2]
    test_node = train_node[split:split2]
    train_adj = train_adj[:split].squeeze()
    train_spatial = train_spatial[:split]
    train_node = train_node[:split]
    
    
    print (train_adj.shape, train_spatial.shape, test_adj.shape, test_spatial.shape)

    train_spatial = np.concatenate((train_spatial, train_node), axis=2)
    test_spatial = np.concatenate((test_spatial, test_node), axis=2)

    graphs_list = formatting_graph_flatten(train_adj, train_spatial, test_adj, test_spatial)

    save_graph_dataset(root_path, name, graphs_list)

def generate_dataset_colab(root_path, name):
    ROOT_PATH = root_path

    train_adj = np.load(os.path.join(ROOT_PATH, 'adj.npy'), allow_pickle=True)
    train_spatial = np.load(os.path.join(ROOT_PATH, 'spatial.npy'), allow_pickle=True)
    train_node = np.load(os.path.join(ROOT_PATH, 'node_feat.npy'), allow_pickle=True)
    print (train_adj.shape, train_spatial.shape, train_node.shape)

    total_len = train_adj.shape[0]
    split = int(total_len * 0.1)
    split2 = int(total_len * 0.11)
    test_adj = train_adj[split:split2].squeeze()
    test_spatial = train_spatial[split:split2]
    test_node = train_node[split:split2]
    train_adj = train_adj[:split].squeeze()
    train_spatial = train_spatial[:split]
    train_node = train_node[:split]
    
    
    print (train_adj.shape, train_spatial.shape, test_adj.shape, test_spatial.shape)

    train_spatial = np.concatenate((train_spatial, train_node), axis=2)
    test_spatial = np.concatenate((test_spatial, test_node), axis=2)

    graphs_list = formatting_graph_flatten(train_adj, train_spatial, test_adj, test_spatial)

    save_graph_dataset(root_path, name, graphs_list)

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

    save_graph_dataset(root_path, 'NTU', graphs_list)

def citeseer_ego():
    _, _, G = data.Graph_load(dataset='citeseer')
    G = max((G.subgraph(c) for c in nx.connected_components(G)), key=len)
    G = nx.convert_node_labels_to_integers(G)
    graphs = []
    for i in range(G.number_of_nodes()):
        G_ego = nx.ego_graph(G, i, radius=3)
        if G_ego.number_of_nodes() >= 50 and (G_ego.number_of_nodes() <= 400):
            graphs.append(G_ego)
    save_graph_dataset_single('../ego', 'EGO', graphs)

def community():
    num_communities = 4
    print('Creating dataset with ', num_communities, ' communities')
    c_sizes = np.random.choice([12, 13, 14, 15, 16, 17], num_communities)
    # c_sizes = [15] * num_communities
    graphs = []
    for k in range(3000):
        graphs.append(n_community(c_sizes, p_inter=0.01))
    max_prev_node = 80
    save_graph_dataset_single('../community', 'Community', graphs)

def clevr():
    size = 10
    spatial, node, adj = [], [], []
    node_feature = ['shape']
    rel_feature = ['right', 'behind', 'front', 'left']
    rel_feature_dou = [['12','21'], ['13','31'], ['24','42'], ['34','43']]
    color_feature = ['blue', 'gray', 'brown', 'purple', 'yellow', 'green', 'cyan', 'red']
    size_feature = ['large', 'small']
    shape_feature = ['sphere', 'cylinder', 'cube']
    material_feature = ['rubber', 'metal']
    features = [shape_feature]
    size_feature = [3]
    f = open('../clevr/CLEVR_train_scenes.json', 'r')
    data = json.load(f)
    length = len(data['scenes'])
    for i in range(length):
        len_obj = len(data['scenes'][i]['objects'])
        if len_obj != size: continue
        spatial_sub = []
        node_sub = []
        for j in range(len_obj):
            coord = data['scenes'][i]['objects'][j]['3d_coords']
            spatial_sub.append(coord)
            node_sub_sub = np.array([[]])
            for feature in node_feature:
                node_sub_sub = np.concatenate((node_sub_sub,to_one_hot(np.array([features[node_feature.index(feature)].index(data['scenes'][i]['objects'][j][feature])]), size_feature[node_feature.index(feature)])), axis=1)
            node_sub.append(node_sub_sub)
        node.append(node_sub)
        spatial.append(spatial_sub)
        adj_sub = np.zeros(shape=(size,size),dtype=int)
        relationship = data['scenes'][i]['relationships']
        merge_adj_sub = np.empty(shape=(size,size),dtype=object)
        merge_adj_sub[:,:] = ''
        for direction in relationship:
            for k in range(len(relationship[direction])):
                for kl in range(len(relationship[direction][k])):
                    # adj_sub i, j means i is of *feature* j
                    merge_adj_sub[relationship[direction][k][kl]][k] += str(rel_feature.index(direction)+1)
                    adj_sub[relationship[direction][k][kl]][k] = rel_feature.index(direction)+1
        for direction in relationship:
            for k in range(len(relationship[direction])):
                for kl in range(len(relationship[direction][k])):
                    # adj_sub i, j means i is of *feature* j
                    for ls in rel_feature_dou:
                        if merge_adj_sub[relationship[direction][k][kl]][k] in ls:
                            adj_sub[relationship[direction][k][kl]][k] = rel_feature_dou.index(ls)+1
        adj.append(adj_sub)

    adj=np.array(adj)
    spatial=np.array(spatial)
    node=np.array(node)
    print (adj.shape, spatial.shape, node.shape)
    node = np.squeeze(node)
    train_adj, train_spatial, train_node = adj, spatial, node 
    train_spatial = np.concatenate((train_spatial, train_node), axis=-1) 

    size = 10
    spatial, node, adj = [], [], []
    node_feature = ['shape']
    rel_feature = ['right', 'behind', 'front', 'left']
    rel_feature_dou = [['12','21'], ['13','31'], ['24','42'], ['34','43']]
    color_feature = ['blue', 'gray', 'brown', 'purple', 'yellow', 'green', 'cyan', 'red']
    size_feature = ['large', 'small']
    shape_feature = ['sphere', 'cylinder', 'cube']
    material_feature = ['rubber', 'metal']
    features = [shape_feature]
    size_feature = [3]
    f = open('../clevr/CLEVR_val_scenes.json', 'r')
    data = json.load(f)
    length = len(data['scenes'])
    for i in range(length):
        len_obj = len(data['scenes'][i]['objects'])
        if len_obj != size: continue
        spatial_sub = []
        node_sub = []
        for j in range(len_obj):
            coord = data['scenes'][i]['objects'][j]['3d_coords']
            spatial_sub.append(coord)
            node_sub_sub = np.array([[]])
            for feature in node_feature:
                node_sub_sub = np.concatenate((node_sub_sub,to_one_hot(np.array([features[node_feature.index(feature)].index(data['scenes'][i]['objects'][j][feature])]), size_feature[node_feature.index(feature)])), axis=1)
            node_sub.append(node_sub_sub)
        node.append(node_sub)
        spatial.append(spatial_sub)
        adj_sub = np.zeros(shape=(size,size),dtype=int)
        relationship = data['scenes'][i]['relationships']
        merge_adj_sub = np.empty(shape=(size,size),dtype=object)
        merge_adj_sub[:,:] = ''
        for direction in relationship:
            for k in range(len(relationship[direction])):
                for kl in range(len(relationship[direction][k])):
                    # adj_sub i, j means i is of *feature* j
                    merge_adj_sub[relationship[direction][k][kl]][k] += str(rel_feature.index(direction)+1)
                    adj_sub[relationship[direction][k][kl]][k] = rel_feature.index(direction)+1
        for direction in relationship:
            for k in range(len(relationship[direction])):
                for kl in range(len(relationship[direction][k])):
                    # adj_sub i, j means i is of *feature* j
                    for ls in rel_feature_dou:
                        if merge_adj_sub[relationship[direction][k][kl]][k] in ls:
                            adj_sub[relationship[direction][k][kl]][k] = rel_feature_dou.index(ls)+1
        adj.append(adj_sub)

    adj=np.array(adj)
    spatial=np.array(spatial)
    node=np.array(node)
    node = np.squeeze(node)
    test_adj, test_spatial, test_node = adj, spatial, node 
    test_spatial = np.concatenate((test_spatial, test_node), axis=-1) 
    graphs_list = formatting_graph_flatten(train_adj, train_spatial, test_adj, test_spatial)

    save_graph_dataset('../clevr', 'CLEVR', graphs_list)


def to_one_hot(data, num_classes):
    one_hot = np.zeros(list(data.shape) + [num_classes])
    one_hot[np.arange(len(data)),data] = 1
    return one_hot


if __name__ == '__main__':
    # root_path = '../profold'
    # generate_dataset_protein(root_path)
    # root_path = '../waxman'
    # generate_dataset_synthetic(root_path, name='waxman')
    # root_path = '../ntu'
    # generate_dataset_pointcloud(root_path, name='ntu')
    # root_path = '../kinetics'
    # generate_dataset_pointcloud_kinetics(root_path, name='kinetics')
    # root_path = '../traffic_la'
    # generate_dataset_traffic(root_path, name='traffic_la')
    # root_path = '../traffic_bay'
    # generate_dataset_traffic(root_path, name='traffic_bay')
    # root_path = '../collab'
    # generate_dataset_colab(root_path, name='collab')
    root_path = '../enzymes_submit_no_padding'
    generate_dataset_protein_enzyme(root_path, name='enzymes')
    # root_path = '../proteins_submit_no_padding'
    # generate_dataset_protein_enzyme(root_path, name='protein')
    citeseer_ego()
    community()
    # clevr()

