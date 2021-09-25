import argparse
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
from random import shuffle
import pickle
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR

# import data
from model_vae import GraphVAE
from data_vae import GraphAdjSampler

CUDA = 2 

LR_milestones = [500, 1000]

def save_graph_list(G_list, fname):
    with open(fname, "wb") as f:
        pickle.dump(G_list, f)


def build_model(args, max_num_nodes):
    out_dim = max_num_nodes * (max_num_nodes + 1) // 2
    if args.feature_type == 'id':
        input_dim = max_num_nodes
    elif args.feature_type == 'deg':
        input_dim = 1
    elif args.feature_type == 'struct':
        input_dim = 2
    elif args.feature_type == 'custom':
        input_dim = args.feature_dim

    print('input_dim', input_dim)
    model = GraphVAE(input_dim, 32, 64, max_num_nodes)
    return model


def train(args, dataloader, model, save_path):
    epoch = 1
    optimizer = optim.Adam(list(model.parameters()), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=LR_milestones, gamma=args.lr)

    model.train()
    for epoch in range(1):
        for batch_idx, data in enumerate(dataloader):
            if batch_idx == 10000:
                break
            model.zero_grad()
            features = data['features'].float()
            adj_input = data['adj'].float()

            features = Variable(features).cuda()
            adj_input = Variable(adj_input).cuda()
            # print (features.shape,adj_input.shape)

            # features = Variable(features)
            # adj_input = Variable(adj_input)

            loss = model(features, adj_input)
            print('Epoch: ', epoch, ', Iter: ', batch_idx, ', Loss: ', loss.item())
            loss.backward()

            optimizer.step()
            scheduler.step()


    G_pred_list = []
    for i in range(1000):    
        graphs = model.sample()
        G_pred_list.append(graphs)
    save_graph_list(G_pred_list, 'pred_'+save_path)
    torch.save(model.state_dict(), save_path)
            # break
        


def arg_parse():
    parser = argparse.ArgumentParser(description='GraphVAE arguments.')
    io_parser = parser.add_mutually_exclusive_group(required=False)
    io_parser.add_argument('--dataset', dest='dataset',
                           help='Input dataset.')

    parser.add_argument('--lr', dest='lr', type=float,
                        help='Learning rate.')
    parser.add_argument('--batch_size', dest='batch_size', type=int,
                        help='Batch size.')
    parser.add_argument('--num_workers', dest='num_workers', type=int,
                        help='Number of workers to load data.')
    parser.add_argument('--max_num_nodes', dest='max_num_nodes', type=int,
                        help='Predefined maximum number of nodes in train/test graphs. -1 if determined by \
                  training data.')
    parser.add_argument('--feature', dest='feature_type',
                        help='Feature used for encoder. Can be: id, deg')

    parser.add_argument('--feature_dim', dest='feature_dim')

    parser.set_defaults(dataset='grid',
                        feature_type='custom',
                        feature_dim=4,
                        lr=0.001,
                        batch_size=1,
                        num_workers=4,
                        max_num_nodes=-1)
    return parser.parse_args()


def load_graph_list(fname):
    with open(fname, "rb") as f:
        graph_list = pickle.load(f)
    return graph_list


def main():
    prog_args = arg_parse()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(2)
    # print('CUDA', CUDA)

    # torch.device('cpu')

    ### running log

    # if prog_args.dataset == 'enzymes':
    #     graphs = data.Graph_load_batch(min_num_nodes=10, name='ENZYMES')
    #     num_graphs_raw = len(graphs)
    # elif prog_args.dataset == 'grid':
    #     graphs = []
    #     for i in range(2, 3):
    #         for j in range(2, 3):
    #             graphs.append(nx.grid_2d_graph(i, j))
    #     num_graphs_raw = len(graphs)
    #
    # if prog_args.max_num_nodes == -1:
    #     max_num_nodes = max([graphs[i].number_of_nodes() for i in range(len(graphs))])
    # else:
    #     max_num_nodes = prog_args.max_num_nodes
    #     # remove graphs with number of nodes greater than max_num_nodes
    #     graphs = [g for g in graphs if g.number_of_nodes() <= max_num_nodes]
    #
    # graphs_len = len(graphs)
    # print('Number of graphs removed due to upper-limit of number of nodes: ',
    #       num_graphs_raw - graphs_len)
    # graphs_test = graphs[int(0.8 * graphs_len):]
    # # graphs_train = graphs[0:int(0.8*graphs_len)]
    # graphs_train = graphs
    #
    # print('total graph num: {}, training set: {}'.format(len(graphs), len(graphs_train)))
    # print('max number node: {}'.format(max_num_nodes))

    # temp_path = './dataset/ST_PROTEIN_full/timestep_0'
    temp_path = '../../n_body_charged/N_BODY_CHARGED_full/timestep_0'
    dataset_name = 'n_body_charged'
    max_num_nodes = 5

    graphs_train = load_graph_list(os.path.join(temp_path, 'train.dat'))
    graphs_test = load_graph_list(os.path.join(temp_path, 'test.dat'))

    dataset = GraphAdjSampler(graphs_train, max_num_nodes, features=prog_args.feature_type)
    # sample_strategy = torch.utils.data.sampler.WeightedRandomSampler(
    #        [1.0 / len(dataset) for i in range(len(dataset))],
    #        num_samples=prog_args.batch_size, 
    #        replacement=False)
    dataset_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=prog_args.batch_size,
        num_workers=prog_args.num_workers)
    model = build_model(prog_args, max_num_nodes).cuda()
    # model = build_model(prog_args, max_num_nodes)

    print('start training')

    train(prog_args, dataset_loader, model, save_path=dataset_name+'.dat')


if __name__ == '__main__':
    main()
