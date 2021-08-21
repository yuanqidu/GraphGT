import pandas as pd
import numpy as np
import sys

from .meta import *
from .utils import *

class DataLoader:
    def __init__(self, name, save_path='./', format='numpy'):
        '''
            Arguments:
            numpy: return np.arrays
            returns:
            self.node_feat: node features of the graphs
            self.adj: adjacency matrices of the graphs
            self.edge_feat: edge features of the graphs
            self.spatial: spatial locations for each node of the graphs
            self.label: labels of the graphs
        '''
        self.name = name
        if format == 'numpy':
            if self.name in dataset['single']:
                self.adj, self.node_feat, self.edge_feat, self.spatial, self.label = load_single_dataset(name, save_path)
            elif self.name in dataset['pair']:
                self.input_adj, self.input_node_feat, self.input_edge_feat, self.input_spatial, self.target_adj, self.target_node_feat, self.target_edge_feat, self.target_spatial, self.label = load_pair_dataset(name, save_path)
            else:
                raise AttributeError("Please use the provided dataset name")
        else:
            raise AttributeError("Please use the correct format input")

    def get_data(self):
        if self.name in dataset['single']:
            return self.adj, self.node_feat, self.edge_feat, self.spatial, self.label
        elif self.name in dataset['pair']:
            return self.input_adj, self.input_node_feat, self.input_edge_feat, self.input_spatial, self.target_adj, self.target_node_feat, self.target_edge_feat, self.target_spatial, self.label

    def __len__(self):
        return self.adj.shape[0]
